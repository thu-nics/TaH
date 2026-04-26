"""Unit test for ``tah.evaluate.jobs.run_single_job`` with a stubbed backend.

Validates the per-job orchestration (output dir layout, prompt assembly,
score-and-save loop, stats write) without loading a real model. Real
end-to-end coverage of the per-backend setup lives in the live
``test_released_checkpoint`` smoke; this test exercises the surrounding
glue, which is the part most commonly broken by surgical refactors.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tokenizer_stub():
    """Bare-minimum tokenizer interface that ``run_single_job`` touches.

    * ``apply_chat_template`` — returns the message content (so ``_make_prompt``
      gets a deterministic string).
    * ``encode``              — counts characters; good enough for the
      input/output token-count fields in the per-row CSV.
    * ``pad_token`` — set, so the constructor's ``if pad_token is None``
      branch isn't hit.
    """

    class T:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            return f"PROMPT::{messages[0]['content']}"

        def encode(self, text, **kwargs):
            return list(text)  # 1 token per character

    return T()


def _fake_backend_factory(echo_prefix: str = "OUT::"):
    """Returns a setup_backend stand-in that produces a tokenizer-free model
    + an inference fn whose outputs are a deterministic function of the input."""

    def _setup(backend, config, model_path, tokenizer, tp_size=1):
        del backend, config, model_path, tokenizer, tp_size  # unused

        def infer(prompts):
            return [(f"{echo_prefix}{p} \\boxed{{42}}", 0.001) for p in prompts]

        return object(), infer

    return _setup


def _fake_score_one(output_text: str, correct: str, dataset_name: str, *, is_code: bool):
    """Mark "boxed{N}" as correct iff N == correct, else wrong."""
    del dataset_name, is_code
    if "\\boxed{" in output_text:
        ans = output_text.split("\\boxed{")[1].split("}")[0]
        return ans, True, ans == correct
    return "", False, False


def test_run_single_job_writes_expected_files(tmp_path, tokenizer_stub):
    """End-to-end: 2 problems × 2 samples = 4 inferences with the fake backend.
    Verify CSV row count, per-sample json files, stats summary."""
    from tah.evaluate import jobs

    problems_data = [
        {"id": "math_1", "question": "What's 41+1?", "answer": "42", "_original_id": "1"},
        {"id": "math_2", "question": "What's 100-58?", "answer": "42", "_original_id": "2"},
    ]
    field_mapping = {
        "id_field": "id", "question_field": "question", "answer_field": "answer",
        "answer_type": "boxed", "prompt_template": "{question}",
    }
    config = {
        "repeat_size": 2, "batch_size": 2, "temperature": 0.0,
        "max_new_tokens": 16, "top_p": 1.0,
    }

    with patch.object(jobs, "setup_backend", _fake_backend_factory()), \
         patch.object(jobs, "_score_one", _fake_score_one), \
         patch.object(jobs, "AutoTokenizer", create=True) if False else patch(
             "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer_stub,
         ), \
         patch.object(jobs, "cleanup", lambda *a, **kw: None):
        jobs.run_single_job(
            config=config, combined_dataset_name="math",
            output_dir=str(tmp_path), timestamp="20260101",
            model_path="dummy", job_id=0, job_nums=1, start_idx=0, end_idx=2,
            tp_size=1, backend="hf", data_range=None,
            problems_data=problems_data, field_mapping=field_mapping,
            unified_code_solutions_file=None,
        )

    job_dir = tmp_path / "math_hf" / "20260101" / "job_0"
    assert job_dir.exists(), f"job dir not created: {sorted(tmp_path.rglob('*'))}"

    # detailed_results.csv: header + 4 rows (2 problems × repeat 2).
    rows = list(csv.DictReader(open(job_dir / "detailed_results.csv")))
    assert len(rows) == 4, rows
    assert all(r["is_correct"] == "True" for r in rows), rows
    assert sorted(set(r["problem_id"] for r in rows)) == ["math_1", "math_2"]
    assert sorted(set(r["sample_idx"] for r in rows)) == ["0", "1"]

    # Per-sample JSON files written under details/<problem_id>/sample_<idx>.json.
    for pid in ("math_1", "math_2"):
        d = job_dir / "details" / pid
        files = sorted(d.glob("sample_*.json"))
        assert len(files) == 2, files
        sample = json.load(open(files[0]))
        assert sample["correct_answer"] == "42"
        assert sample["predicted_answer"] == "42"
        assert sample["is_correct"]

    # evaluation_stats.csv: per-problem rows + a "Total Accuracy" row.
    stat_rows = list(csv.reader(open(job_dir / "evaluation_stats.csv")))
    header = stat_rows[0]
    body = [r for r in stat_rows[1:] if r and r[0] not in ("", "Total Accuracy")]
    total = next(r for r in stat_rows if r and r[0] == "Total Accuracy")
    assert "accuracy" in header
    assert len(body) == 2  # one row per problem
    assert total[1] == "1.000"  # everyone correct
    assert int(total[3]) == 4   # total samples


def test_build_prompts_repeat_and_mapping(tokenizer_stub):
    from tah.evaluate.jobs import _build_prompts

    problems = [
        {"problem_text": "Q1", "problem_id": "p1"},
        {"problem_text": "Q2", "problem_id": "p2"},
    ]
    prompts, mapping = _build_prompts(problems, tokenizer_stub, repeat_size=3, is_code=False)
    assert len(prompts) == 6
    assert prompts[0] == prompts[1] == prompts[2] == "PROMPT::Q1"
    assert prompts[3] == prompts[4] == prompts[5] == "PROMPT::Q2"
    assert mapping == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
