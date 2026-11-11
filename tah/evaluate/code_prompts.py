"""TaH2's reasoning prompt and offline code extraction, without TaH2 imports.

Copied from tah2/evaluate/{codeeval,local_eval}.py. The harness variant stays
in codeeval.py; keep this path separate to preserve existing evaluations.
"""
from __future__ import annotations

import re


MBPP_QUERY_TEMPLATE = """
Please solve the programming task with test cases below in Python. Make sure your code satisfies the following requirements:
1. The function name and signature must match exactly as specified in the test cases.
2. Your code should be wrapped in a markdown code block without including any test cases.

Task:
{Question}

Test Cases:
```python
{TestCases}
```
""".strip()

HUMANEVAL_QUERY_TEMPLATE = """
Please solve the programming task below in Python. Code should be wrapped in a markdown code block.

```python
{Question}
```
""".strip()


def make_tah2_prompt(task_prompt: str, assertion: str, answer_type: str, tokenizer) -> str:
    """Match TaH2's make_raw_chat_prompt_for_code_evaluation(reasoning=True)."""
    if tokenizer.chat_template is None:
        return task_prompt
    if answer_type == "mbpp":
        content = MBPP_QUERY_TEMPLATE.format(Question=task_prompt, TestCases=assertion)
    elif answer_type == "humaneval":
        content = HUMANEVAL_QUERY_TEMPLATE.format(Question=task_prompt)
    else:
        raise ValueError(f"Unsupported code answer type: {answer_type}")
    # No enable_thinking override or assistant priming in the authors' path.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    )


_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"\bdef\s+\w+\s*\(")


def extract_tah2_code(text: str) -> str:
    """Match TaH2 local_eval.process_batch_items' extract_code(output_text).

    That call omits entry_point: take the first Python block containing a
    function, otherwise the last Python block, or empty if none. It neither
    strips thinking nor applies EvalPlus sanitize before submitting solutions.
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        return ""
    for block in blocks:
        if _DEF_RE.search(block):
            return block
    return blocks[-1]
