"""
Unified Answer Evaluation Module

This module integrates answer extraction, normalization, and grading functionalities
for mathematical problem evaluation. It combines code from the evaluation pipeline
and the math500 grading system.

Main functions:
- extract_boxed_answer: Extract answers from \\boxed{...} format
- normalize_answer: Normalize mathematical expressions for comparison
- grade_answer: Grade answers using multiple evaluation strategies
- evaluate_answer: Unified interface for answer evaluation
"""

import re
import sympy
from typing import Optional, Tuple, Union
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


# Constants for sympy evaluation safety
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()\[\]"


def extract_boxed_answer(text: str) -> Tuple[str, bool]:
    """
    Extract answer from the last \\boxed{...} format commonly used in mathematical solutions.
    
    Args:
        text (str): The text containing potential boxed answer
        
    Returns:
        Tuple[str, bool]: (extracted_answer, has_boxed_format)
            - extracted_answer: The content inside the last \\boxed{} or empty string if not found
            - has_boxed_format: True if \\boxed{} pattern was found, False otherwise
    """
    # Find all occurrences of \\boxed{
    boxed_positions = []
    pos = 0
    while True:
        boxed_start = text.find("\\boxed{", pos)
        if boxed_start == -1:
            break
        boxed_positions.append(boxed_start)
        pos = boxed_start + 1
    
    if not boxed_positions:
        return "", False
    
    # Process from the last occurrence backwards to find the last valid match
    for boxed_start in reversed(boxed_positions):
        # Start counting braces after the opening brace of \\boxed{
        start_pos = boxed_start + len("\\boxed{")
        brace_count = 1
        pos = start_pos
        
        # Find the matching closing brace
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        
        # If we found a matching closing brace, this is our answer
        if brace_count == 0:
            content = text[start_pos:pos-1]
            return content.strip(), True
    
    return "", False


def extract_answer_patterns(text: str) -> Tuple[str, str]:
    """
    Extract answers using multiple common patterns in mathematical text.
    
    Args:
        text (str): The text containing potential answers
        
    Returns:
        Tuple[str, str]: (extracted_answer, extraction_method)
            - extracted_answer: The extracted answer
            - extraction_method: Method used for extraction ('boxed', 'final', 'last_number', 'none')
    """
    # Try boxed format first
    boxed_answer, has_boxed = extract_boxed_answer(text)
    if has_boxed:
        return boxed_answer, 'boxed'
    
    # Try "The answer is X" pattern
    final_answer_patterns = [
        r"[Tt]he answer is:?\s*([^\n\.]+)",
        r"[Ss]o,?\s*the answer is:?\s*([^\n\.]+)",
        r"[Ff]inal answer:?\s*([^\n\.]+)",
        r"[Tt]he final answer is:?\s*([^\n\.]+)",
    ]
    
    for pattern in final_answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(), 'final'
    
    # # Try to extract the last number or mathematical expression
    # number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    # numbers = re.findall(number_pattern, text)
    # if numbers:
    #     return numbers[-1], 'last_number'
    
    return "", 'none'


# ===== Math Normalization Functions (from math_normalize.py) =====

def normalize_answer(answer: Optional[str]) -> Optional[str]:
    """
    Normalize mathematical answer using the math500 normalization approach.
    This logic is largely copied from the Hendrycks' MATH release (math_equivalence).
    
    Args:
        answer (Optional[str]): The answer to normalize
        
    Returns:
        Optional[str]: Normalized answer or None if input is None
    """
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _fix_fracs(string):
    """Fix fraction formatting in LaTeX expressions."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    """Convert simple fraction format a/b to LaTeX \\frac{a}{b}."""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    """Remove unit descriptions from the right side of expressions."""
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    """Fix square root formatting to ensure proper LaTeX syntax."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    """Core string normalization function with comprehensive cleaning."""
    # Remove linebreaks
    string = string.replace("\n", "")

    # Remove inverse spaces
    string = string.replace("\\!", "")

    # Replace \\ with \
    string = string.replace("\\\\", "\\")

    # Replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs
    string = string.replace("\\$", "")

    # Remove units (on the right)
    string = _remove_right_units(string)

    # Remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." 
    # Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    # If empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # Remove variable assignments like "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # Fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # Remove spaces
    string = string.replace(" ", "")

    # Fix fractions: \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}
    string = _fix_fracs(string)

    # Manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Convert X/Y to \frac{X}{Y} in simple cases
    string = _fix_a_slash_b(string)

    return string


# ===== Grading Functions (from grader.py) =====

def _sympy_parse(expr: str):
    """Parse an expression with sympy, handling common mathematical notation."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Parse LaTeX mathematical expressions to sympy-readable format."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace specific mathematical symbols
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    """Check if string represents a valid float."""
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    """Check if float value is effectively an integer."""
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    """Check if expression is in simple fraction format."""
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    """Check if string represents an integer (handling commas)."""
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> int:
    """Convert string to integer (handling commas)."""
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evaluable.
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    """Remove properly formatted commas from numbers while preserving tuple commas."""
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize_grader(expr: str) -> str:
    """
    Normalize answer expressions for grading comparison.
    This is more comprehensive than the basic math normalization.
    """
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    # Handle large number descriptions
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    # Remove common units
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute", 
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    # Remove outer braces
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    
    # Convert float to int if it's a whole number
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    
    # Parse LaTeX if present
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # Handle negative signs in mixed numbers
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # Remove remaining LaTeX braces
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # Case insensitive for text answers
    expr = expr.lower()

    # Convert to integer string if applicable
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    """Count unknown variables in mathematical expression."""
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    """
    Determine if expression is safe for sympy evaluation.
    Avoid parsing unknown text or functions with too many variables.
    """
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    """Check if two normalized expressions are mathematically equivalent using sympy."""
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split elements in a tuple/interval while handling well-formatted commas in large numbers.
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    Grade answer using comprehensive mathematical equivalence checking.
    
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer, OR
    (b) sympy can simplify the difference between the expressions to 0
    
    Args:
        given_answer (str): The answer provided by the model
        ground_truth (str): The correct answer
        
    Returns:
        bool: True if answer is correct, False otherwise
    """
    if given_answer is None:
        return False

    # First try basic math normalization (more lenient)
    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    # Then try more comprehensive grader normalization
    ground_truth_normalized = _normalize_grader(ground_truth)
    given_normalized = _normalize_grader(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    # Handle tuple/interval answers
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    # Check tuple structure consistency
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        # Check each element
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # For fractions, require exact match (no reduction allowed)
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # Type mismatch between integer and non-integer
                is_correct = False
            else:
                # Use sympy for mathematical equivalence
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            
            if not is_correct:
                break

    return is_correct


# ===== Unified Evaluation Interface =====

def evaluate_answer(
    response_text: str, 
    ground_truth: str, 
    extraction_method: str = 'auto',
    grading_method: str = 'comprehensive'
) -> dict:
    """
    Unified interface for answer evaluation combining extraction and grading.
    
    Args:
        response_text (str): The full response text from the model
        ground_truth (str): The correct answer
        extraction_method (str): Method for answer extraction
            - 'auto': Try multiple patterns automatically
            - 'boxed': Only look for \\boxed{} format
            - 'pattern': Use pattern matching for common answer formats
        grading_method (str): Method for answer grading
            - 'comprehensive': Use full mathematical equivalence checking
            - 'basic': Use basic string normalization only
            - 'exact': Require exact string match after normalization
            
    Returns:
        dict: Evaluation results containing:
            - 'extracted_answer': The extracted answer string
            - 'extraction_method_used': Method that successfully extracted the answer
            - 'has_answer': Whether any answer was found
            - 'is_correct': Whether the answer is correct
            - 'ground_truth_normalized': Normalized ground truth
            - 'extracted_normalized': Normalized extracted answer
    """
    result = {
        'extracted_answer': '',
        'extraction_method_used': 'none',
        'has_answer': False,
        'is_correct': False,
        'ground_truth_normalized': '',
        'extracted_normalized': ''
    }
    
    # Extract answer based on specified method
    if extraction_method == 'boxed':
        extracted_answer, has_boxed = extract_boxed_answer(response_text)
        result['extracted_answer'] = extracted_answer
        result['has_answer'] = has_boxed
        result['extraction_method_used'] = 'boxed' if has_boxed else 'none'
    elif extraction_method == 'pattern':
        extracted_answer, method_used = extract_answer_patterns(response_text)
        result['extracted_answer'] = extracted_answer
        result['has_answer'] = bool(extracted_answer)
        result['extraction_method_used'] = method_used
    else:  # 'auto'
        # Try boxed first, then patterns
        extracted_answer, has_boxed = extract_boxed_answer(response_text)
        if has_boxed:
            result['extracted_answer'] = extracted_answer
            result['has_answer'] = True
            result['extraction_method_used'] = 'boxed'
        else:
            extracted_answer, method_used = extract_answer_patterns(response_text)
            result['extracted_answer'] = extracted_answer
            result['has_answer'] = bool(extracted_answer)
            result['extraction_method_used'] = method_used
    
    # Grade answer if one was extracted
    if result['has_answer']:
        if grading_method == 'comprehensive':
            result['is_correct'] = grade_answer(result['extracted_answer'], ground_truth)
        elif grading_method == 'basic':
            result['ground_truth_normalized'] = normalize_answer(ground_truth)
            result['extracted_normalized'] = normalize_answer(result['extracted_answer'])
            result['is_correct'] = (result['ground_truth_normalized'] == result['extracted_normalized'])
        elif grading_method == 'exact':
            result['is_correct'] = (result['extracted_answer'].strip() == ground_truth.strip())
        
        # Always provide normalized versions for inspection
        if not result['ground_truth_normalized']:
            result['ground_truth_normalized'] = normalize_answer(ground_truth)
        if not result['extracted_normalized']:
            result['extracted_normalized'] = normalize_answer(result['extracted_answer'])
    
    return result


# ===== Convenience Functions =====

def simple_grade(predicted_answer: str, correct_answer: str) -> bool:
    """
    Simple grading function that mimics the original eval_unified.py logic.
    
    Args:
        predicted_answer (str): The predicted answer
        correct_answer (str): The correct answer
        
    Returns:
        bool: True if answers match exactly, False otherwise
    """
    if not predicted_answer:
        return False
    return predicted_answer.strip() == correct_answer.strip()


def extract_and_grade_boxed(response_text: str, ground_truth: str) -> Tuple[str, bool, bool]:
    """
    Extract boxed answer and grade it, mimicking original evaluation logic.
    
    Args:
        response_text (str): The full response text
        ground_truth (str): The correct answer
        
    Returns:
        Tuple[str, bool, bool]: (extracted_answer, has_boxed_answer, is_correct)
    """
    extracted_answer, has_boxed = extract_boxed_answer(response_text)
    is_correct = simple_grade(extracted_answer, ground_truth) if has_boxed else False
    return extracted_answer, has_boxed, is_correct

