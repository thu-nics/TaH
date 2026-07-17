"""Per-dataset rule-based math graders.

Every benchmark in ``eval_configs/dataset_configs.json`` whose ``answer_type``
isn't a code answer maps here through :data:`evaluator_map`. The runner calls
``evaluator_map[dataset].rule_judge(output_text, ground_truth)`` and gets back
``(is_correct: bool, predicted: str)`` (or ``(False, "No extracted answer")``
when nothing matches).

Three grading shapes cover all configured datasets — they only differ in how
ground truth and the model output are extracted before being compared via
``math_verify.verify``:

* ``expr``    — AIME / GSM8K. Ground truth is a bare expression; output is
  matched as latex-or-expr.
* ``latex``   — MATH500 / AMC / OlympiadBench. Ground truth is wrapped in
  ``$...$`` if not already, then parsed as latex; output as latex-or-expr.
* ``string``  — GPQA / MMLU / ARC. Both sides are matched as strings.

Public TaH also shipped LLM-judge plumbing (``llm_judge`` / ``set_client``
/ a per-evaluator ``get_llm_judge_prompt``); the eval driver only ever
calls ``rule_judge``, so the LLM-judge surface is removed.
"""
from __future__ import annotations

from typing import Tuple

from latex2sympy2_extended import NormalizationConfig
from math_verify import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
    parse,
    verify,
)


# Latex+Expr config used by both expr-mode and latex-mode for the model output.
_OUTPUT_EXTRACTION = [
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
]


class MathEvaluator:
    """Grader for one of three answer shapes (``expr``/``latex``/``string``).

    Construct with the ground-truth shape; :meth:`rule_judge` does the
    parse+verify dance with that shape.
    """

    def __init__(self, mode: str):
        if mode not in {"expr", "latex", "string"}:
            raise ValueError(f"unsupported grader mode {mode!r}")
        self.mode = mode

    def rule_judge(self, solution: str, ground_truth: str) -> Tuple[bool, str]:
        if self.mode == "expr":
            gold_cfg = [ExprExtractionConfig()]
            answer_cfg = _OUTPUT_EXTRACTION
        elif self.mode == "latex":
            if not ground_truth.startswith("$"):
                ground_truth = f"${ground_truth}$"
            gold_cfg = [LatexExtractionConfig()]
            answer_cfg = _OUTPUT_EXTRACTION
        else:  # string
            gold_cfg = [StringExtractionConfig()]
            answer_cfg = [StringExtractionConfig()]

        gold = parse(ground_truth, extraction_config=gold_cfg)
        # Match main's per-mode behavior: math (expr/latex) takes the first match;
        # string/multiple-choice uses math_verify's default extraction_mode
        # ("any_match"), exactly as main's GPQAEvaluator did (it passed no
        # extraction_mode kwarg). Using first_match for string mode would diverge.
        if self.mode == "string":
            answer = parse(solution, extraction_config=answer_cfg)
        else:
            answer = parse(solution, extraction_config=answer_cfg, extraction_mode="first_match")
        if not answer:
            return False, "No extracted answer"
        return bool(verify(gold, answer)), str(answer)


# Dataset → grader mapping. Keys must match
# ``eval_configs/dataset_configs.json`` benchmark names.
_EXPR = MathEvaluator("expr")
_LATEX = MathEvaluator("latex")
_STRING = MathEvaluator("string")

evaluator_map = {
    # Bare-expression ground truth.
    "aime24": _EXPR,
    "aime25": _EXPR,
    "brumo25": _EXPR,
    "gsm8k": _EXPR,
    # Latex-wrapped ground truth.
    "math500": _LATEX,
    "amc23": _LATEX,
    "chmath": _LATEX,
    "olympiadbench": _LATEX,
    "minerva": _LATEX,
    # String ground truth (multiple-choice).
    "gpqa": _STRING,
    "mmlu_stem": _STRING,
    "mmlu_redux": _STRING,
    "arc_e": _STRING,
    "arc_c": _STRING,
}
