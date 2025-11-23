import re
from openai import OpenAI
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from latex2sympy2_extended import NormalizationConfig
import os

class MathEvaluator:
    
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        raise NotImplementedError

    def extract_after_think(self, text: str, truncate_length: int = 1000, finish_generation: bool = True) -> str:
        pattern = r"</think>(.*)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if (match and finish_generation) else text[-truncate_length:]
    
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> str:
        raise NotImplementedError

    def get_llm_judge_prompt_not_finished(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> str:
        return f"""Please determine whether the final answer in the model-generated response was already correctly derived early in the reasoning process, and that the subsequent content consists mainly of unnecessary verification, overthinking, or repetitive reasoning. If correct is derived early, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Reference answer: {ground_truth}
Model-generated response: {solution_str}
""".strip()

    def llm_judge(self, solution_str: str, ground_truth: str, extracted_answer: str = "", finish_generation: bool = True) -> bool:
        global OPENAI_CLIENT, MODEL_NAME
        def get_inputs(scene_description):
            body = [
                {"role": "user", "content": scene_description},
            ]
            return body

        def run_api(inputs):
            completion = OPENAI_CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=inputs
            )
            return completion.choices[0].message.content.strip()
        if finish_generation:
            scene_description = self.get_llm_judge_prompt(solution_str, ground_truth, extracted_answer, finish_generation)
        else:
            scene_description = self.get_llm_judge_prompt_not_finished(solution_str, ground_truth, extracted_answer, finish_generation)
        inputs = get_inputs(scene_description)
        response = run_api(inputs)

        return "YES" in response


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

    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


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

    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response with rule-based extracted answer is equivalent to the reference answer from a math question. The final answer may either be enclosed in the \\boxed{{}} or appear after the "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.

1. The reference answer does not include percentage signs, units or time formats (e.g., am, pm), but the Model-generated answer may include them.
For example, 1 is equivalent to 1 %, 1 kg, 1 am, 1 pm, 1:00 am, 1:00 pm, etc.
Model-generated answer: 1%
Reference answer: 1
Your output: YES

Model-generated answer: 1 kg
Reference answer: 1
Your output: YES

Model-generated answer: 1:00 pm
Reference answer: 1
Your output: YES

2. The reference answer only includes one single number, but the Model-generated answer may include multiple numbers.
For example, 10 is equivalent to \\boxed{{(4, 6)}}, etc.
Model-generated answer: 5, 5
Reference answer: 10
Your output: YES

Model-generated answer: 4, 6
Reference answer: 10
Your output: YES

Model-generated answer: 86, 42
Reference answer: 128
Your output: YES

Now let's try a real example.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}
""".strip()


class MATH500Evaluator(MathEvaluator):
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
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()
    
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
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()

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
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a math question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()

class GPQAEvaluator(MathEvaluator):
    def rule_judge(self, solution_str: str, ground_truth: str, finish_generation: bool = True) -> bool:
        # if not ground_truth.startswith("$"):
        #     ground_truth = f"${ground_truth}$"
        gold = parse(
            ground_truth,
            extraction_config=[StringExtractionConfig()],
        )
        answer = parse(
            solution_str,
            extraction_config=[
                StringExtractionConfig(),
            ]
        )
        if len(answer) == 0:
            return False, "No extracted answer"
        else:
            return verify(gold, answer), str(answer)
        
    def get_llm_judge_prompt(self, solution_str: str, ground_truth: str, extract_answer: str = "", finish_generation: bool = True) -> str:
        solution_str = self.extract_after_think(solution_str, finish_generation=finish_generation)
        return f"""Please determine whether the final answer provided in the model-generated response is equivalent to the reference answer from a multiple choice question. The final answer may either be enclosed in \\boxed{{}} or appear after "Answer:". If they are equivalent, return "YES"; if they are not, return "NO". Only return "YES" or "NO", and do not generate any other content.
Model-generated answer: {solution_str}
Reference answer: {ground_truth}""".strip()


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
    "aime24": AIMEEvaluator(),
    "aime25": AIMEEvaluator(),
    "brumo25": AIMEEvaluator(),
    "chmath": AMCEvaluator(),
    "gsm8k": GSM8KEvaluator(),
    "math500": MATH500Evaluator(),
    "amc23": AMCEvaluator(),
    "olympiadbench": OlympiadBenchEvaluator(),
    "gpqa": GPQAEvaluator(),
    "minerva": AMCEvaluator(),
    "mmlu_stem": GPQAEvaluator(),
    "mmlu_redux": GPQAEvaluator(),
    "arc_e": GPQAEvaluator(),
    "arc_c": GPQAEvaluator(),
}

API_BASE = None
DEPLOYMENT_NAME = None
API_VERSION = None
CONSTRUCTED_URL = None
API_KEY = None
HEADERS = None
OPENAI_CLIENT = None
MODEL_NAME = None

def set_client(api_base=None, deployment_name=None, api_version=None, api_key=None, model_name="gpt-4.1-2025-04-14"):
    global API_BASE, DEPLOYMENT_NAME, API_VERSION, CONSTRUCTED_URL, API_KEY, HEADERS, MODEL_NAME, OPENAI_CLIENT

    API_BASE = api_base
    DEPLOYMENT_NAME = deployment_name
    API_VERSION = api_version
    CONSTRUCTED_URL = f"{api_base}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    API_KEY = api_key or os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = model_name
    HEADERS = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    if API_KEY:
        print(f"Using API key: {API_KEY}")
        OPENAI_CLIENT = OpenAI(api_key=API_KEY)
    else:
        OPENAI_CLIENT = None
    