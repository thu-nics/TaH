import re
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from latex2sympy2_extended import NormalizationConfig


class MathEvaluator:

    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        raise NotImplementedError

    def extract_after_think(self, text: str, truncate_length: int = 1000, finish_generation: bool = True) -> str:
        pattern = r"</think>(.*)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if (match and finish_generation) else text[-truncate_length:]



class AIMEEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)


class GSM8KEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[ExprExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)


class MATH500Evaluator(MathEvaluator):
    evaluation_method = "math_verify"

    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)


class AMCEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)

class OlympiadBenchEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        if not ground_truth.startswith("$"):
            ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[LatexExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
                ExprExtractionConfig(),
            ],
            extraction_mode="first_match",
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)


# class MBPPEvaluator(Evaluator):
#     def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
#         return True, "No extracted answer"

#     def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
#         solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
#         return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
# Model-generated answer: {solution_str}
# Reference answer: {ground_truth}""".strip()


# class HUMANEVALEvaluator(Evaluator):
#     def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
#         return True, "No extracted answer"

#     def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
#         solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
#         return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
# Model-generated answer: {solution_str}
# Reference answer: {ground_truth}""".strip()


evaluator_map = {
    "gsm8k": GSM8KEvaluator(),
    "math500": MATH500Evaluator(),
    "amc23": AMCEvaluator(),
    "olympiadbench": OlympiadBenchEvaluator(),
    "aime25": AIMEEvaluator(),
    "aime26": AIMEEvaluator(),
}
