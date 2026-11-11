"""CPU/tokenizer-only parity with the authors' actual TaH2 functions/loaders."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tah.evaluate import codeeval, jobs
from tah.evaluate.code_prompts import extract_tah2_code, make_tah2_prompt
from tah.evaluate.datasets import load_combined_dataset


ROOT = Path(__file__).resolve().parents[4]
# Downloaded from TaH-plus/eval_science_code at this immutable revision; the
# complete files (including all assertions) matched our existing local files.
SOURCE_REVISION = "8737a89ea8d141dae003d33ca1f818de58b237b1"
SOURCE_SHA256 = {
    "humaneval": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
    "mbpp": "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad",
}
# Captured from _make_prompt before adding the variant, using the checkpoint
# tokenizer. Hash json.dumps(list_of_prompts, ensure_ascii=False).encode().
HARNESS_SHA256 = {
    "humaneval": "dde2c61c7c2f1ce753f249daee2cb9d4da417f122e44edaa7ecbb07a6c781b2c",
    "mbpp": "8b74e5cdd7a75da09866ce9bf4ed25e88f079eadfbe4abec79d17bcc00317c79",
}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    path = Path(os.environ.get("TAH_CODE_TOKENIZER", ROOT / "models/TaH-plus-1.7B"))
    if not path.exists():
        pytest.skip("set TAH_CODE_TOKENIZER to the local TaH-plus-1.7B tokenizer")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


@pytest.mark.parametrize("name,count", [("humaneval", 164), ("mbpp", 378)])
def test_all_prompts_match_tah2_and_harness_snapshot(name, count, tokenizer, monkeypatch, tmp_path):
    # These imports work in .venv-tah: compare the real authors' implementation,
    # not another copy of our templates. No model initialization or network.
    from tah2.evaluate import codeeval as authors
    from tah2.evaluate.common import load_datasets_with_config as authors_load
    import huggingface_hub

    source = ROOT / f"data/initial_data/eval/{name}.jsonl"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256[name]

    def cached_source(repo_id, repo_type, filename):
        assert repo_id == "TaH-plus/eval_science_code" and repo_type == "dataset"
        assert filename == name + ".jsonl"
        return str(source)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", cached_source)
    reference, _ = authors_load(name)
    rows, mapping = load_combined_dataset(name)
    assert len(reference) == len(rows) == count
    prompts = []
    for index, (row, author_row) in enumerate(zip(rows, reference)):
        problem = jobs._build_problem(row, index, mapping, tmp_path)
        assert row["_original_id"] == author_row["_original_id"]
        # local_eval strips question a second time and leaves assertion intact.
        assert problem["problem_text"] == author_row["question"].strip()
        assert problem["assertion"] == author_row["assertion"]
        expected = authors.make_raw_chat_prompt_for_code_evaluation(
            author_row["question"].strip(), author_row["assertion"], name, True, tokenizer,
        )
        actual = jobs._make_prompt(problem["problem_text"], tokenizer, is_code=True,
                                  code_prompt="tah2", answer_type=name, assertion=problem["assertion"])
        assert actual == expected, row["id"]
        assert actual.endswith("<|im_start|>assistant\n")
        default = jobs._make_prompt(problem["problem_text"], tokenizer, is_code=True)
        assert default == jobs._make_prompt(problem["problem_text"], tokenizer, is_code=True,
                                            code_prompt="harness")
        assert default == codeeval.make_raw_chat_prompt_for_code_evaluation(
            problem["problem_text"], False, tokenizer)
        prompts.append(default)
    assert hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest() == HARNESS_SHA256[name]


@pytest.mark.parametrize("text,expected", [
    ("<think>Plan.</think>\n```python\ndef f():\n    return 1\n```", "def f():\n    return 1\n"),
    ("<think>Plan.</think>\n```python\ndef f(): pass\n```\nExample:\n```python\nprint(f())\n```",
     "def f(): pass\n"),
    ("```python\nx = 1\n```\n```python\ny = 2\n```", "y = 2\n"),
    ("<think>```python\ndef draft(): pass\n```</think>```python\ndef f(): pass\n```",
     "def draft(): pass\n"),
    ("```python\ndef wrong(): pass\n```\n```python\ndef f(): pass\n```", "def wrong(): pass\n"),
    ("<think>unfinished reasoning", ""),
    ("def f(): pass", ""),
    ("```\ndef f(): pass\n```", ""),
    ("```python\ndef f(): pass", ""),
])
def test_reasoning_extraction_matches_actual_tah2(text, expected):
    from tah2.evaluate.local_eval import extract_code
    assert extract_tah2_code(text) == extract_code(text) == expected


def test_chat_arguments_and_no_template_fallback():
    calls = []
    def apply(messages, **kwargs):
        calls.append((messages, kwargs))
        return "rendered"
    tokenizer = SimpleNamespace(chat_template="template", apply_chat_template=apply)
    assert make_tah2_prompt(" task ", "\nassert f()\n", "mbpp", tokenizer) == "rendered"
    messages, kwargs = calls.pop()
    assert [m["role"] for m in messages] == ["user"]
    assert "\n task \n" in messages[0]["content"]
    assert "\n\nassert f()\n\n" in messages[0]["content"]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}
    tokenizer.chat_template = None
    assert make_tah2_prompt(" task ", "", "mbpp", tokenizer) == " task "
    tokenizer.chat_template = "template"
    with pytest.raises(ValueError, match="Unsupported code answer type"):
        make_tah2_prompt("", "", "other", tokenizer)
    with pytest.raises(ValueError, match="code_prompt"):
        jobs._make_prompt("", tokenizer, is_code=True, code_prompt="typo")
    jobs._make_prompt("math", tokenizer, is_code=False, code_prompt="tah2")
    assert calls[-1][0] == [{"role": "user", "content": "math"}]


@pytest.mark.parametrize("variant", [None, "harness", "tah2"])
def test_worker_uses_variant_for_prompts_and_persisted_solutions(variant, tokenizer, monkeypatch, tmp_path):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a: tokenizer)
    rows, mapping = load_combined_dataset("humaneval,mbpp")
    rows = [rows[0], rows[164]]
    output = "<think>Plan.</think>\n```python\ndef wrong(): pass\n```\n```python\nprint('demo')\n```"
    captured = []
    # Assert that the TaH2 path does not invoke the harness sanitizer at all.
    def sanitize(text, entry_point):
        assert variant != "tah2"
        assert text == output
        return "legacy sanitized " + entry_point
    monkeypatch.setattr(codeeval, "sanitize", sanitize)
    config = dict(batch_size=16, repeat_size=2, grading_workers=0)
    if variant is not None:
        config["code_prompt"] = variant
    # Two repeats for each of two datasets; callback deliberately out of order.
    def repeated_infer(prompts, on_result):
        captured.extend(prompts)
        for index in (3, 1, 2, 0):
            on_result(index, (output, 0.1))
    repeated_infer.supports_callback = True
    monkeypatch.setattr(jobs, "setup_backend", lambda *a, **kw: (None, repeated_infer))
    jobs.run_single_job(config=config, combined_dataset_name="humaneval_mbpp",
        dataset_names=["humaneval", "mbpp"], output_dir=str(tmp_path), timestamp="test",
        model_path="stub", job_id=0, job_nums=1, start_idx=0, end_idx=2, tp_size=1,
        backend="tah_engine", data_range=None, problems_data=rows, field_mapping=mapping)
    assert len(captured) == 4 and captured[0] == captured[1] and captured[2] == captured[3]
    for index, row in enumerate(rows):
        ds = row["_source_dataset"]
        expected = jobs._make_prompt(row["question"].strip(), tokenizer, is_code=True,
            code_prompt=variant or "harness", answer_type=ds,
            assertion=row.get("_original_assertion", row.get("_original_test", "")))
        assert captured[index * 2] == expected
        expected_code = "def wrong(): pass\n" if variant == "tah2" else "legacy sanitized " + row["entry_point"]
        solutions = [json.loads(line) for line in
                     (tmp_path / f"{ds}_tah_engine/test/code_solutions.jsonl").read_text().splitlines()]
        assert solutions == [{"task_id": row["_original_id"], "solution": expected_code}] * 2
        for path in (tmp_path / f"{ds}_tah_engine").rglob("sample_*.json"):
            detail = json.loads(path.read_text())
            assert detail["output"] == output and detail["extracted_code"] == expected_code


def test_amendment_recipe_settings():
    directory = ROOT / "script/tah/recipes/qwen3_1.7"
    recipe = yaml.safe_load((directory / "protocol_paper_code_tah2.yaml").read_text())
    base = yaml.safe_load((directory / recipe["eval_config"]).read_text())
    assert recipe["backends"] == ["tah_engine"]
    assert [g["datasets"] for g in recipe["groups"]] == [["humaneval"], ["mbpp"]]
    for group in recipe["groups"]:
        assert group["repeat_size"] == 1 and group["max_new_tokens"] == 8192
        assert group["overrides"] == {"code_prompt": "tah2"}
    assert (base["temperature"], base["top_p"], base["top_k"]) == (0.6, 0.95, 20)
