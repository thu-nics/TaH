"""Harness boundaries: indexed completion, shared engine and durable grading."""
import csv
import json
from types import SimpleNamespace
import runpy

import pytest
import torch

from tah.evaluate import backends, jobs


class Tokenizer:
    pad_token = "eos"
    eos_token = "eos"

    def encode(self, text):
        return list(range(len(text))) or [1]

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


def test_engine_adapter_one_instance_and_prompt_cap(monkeypatch, tmp_path):
    import tah.engine as api
    instances = []

    class Engine:
        def __init__(self, path, config_override, **options):
            instances.append(self)
            self.options = options
            self.max_batch_size = options["max_batch_size"]
            self.seed = options["seed"]
            self.device = torch.device("cuda")
            self.memory_budget = {}
            self.calls = []

        def generate(self, requests, sampling, on_result=None):
            self.calls.append(requests)
            self.last_run = dict(wall_seconds=1., generated_tokens=len(requests))
            results = [api.Result(i, [3], "answer", [1], [2], "length", dict(total_seconds=.5))
                       for i in range(len(requests))]
            if on_result:
                for r in reversed(results):
                    on_result(r)
            return results

    monkeypatch.setattr(api, "TaHEngine", Engine)
    monkeypatch.setattr(api, "check_environment", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 20)
    model, infer = backends.setup_backend("tah_engine", dict(temperature=.6, top_p=.95,
        top_k=20, max_new_tokens=17, _prompt_count=2, engine=dict(max_batch_size=64,
        graph_batch_sizes=[1, 2, 64]), _random_seed=42), str(tmp_path), Tokenizer())
    completed = []
    infer(["math", "gpqa"], on_result=lambda i, r: completed.append((i, r)))
    assert len(instances) == 1 and len(model.calls) == 2
    assert model.options["max_batch_size"] == 2
    assert model.options["graph_batch_sizes"] == [1, 2]
    assert [r.seed for r in model.calls[-1]] == [42, 43]
    assert [r.max_new_tokens for r in model.calls[-1]] == [17, 17]
    assert [i for i, r in completed] == [1, 0]
    assert completed[0][1]["pass_counts"] == [2]
    assert infer.telemetry["peak_gpu_allocated_bytes"] == 10


def test_environment_error_points_to_reference(monkeypatch, tmp_path):
    import tah.engine as api
    def fail():
        raise RuntimeError("missing toolkit")
    monkeypatch.setattr(api, "check_environment", fail)
    with pytest.raises(RuntimeError, match="--backend tah"):
        backends.setup_tah_engine({}, str(tmp_path), Tokenizer())
    with pytest.raises(ValueError, match="one GPU"):
        backends.setup_tah_engine({}, str(tmp_path), Tokenizer(), tp_size=2)


def test_playground_colours_and_eval_default(capsys):
    script = runpy.run_path("script/tah/playground/inference_example.py", run_name="playground_test")
    script["_print_coloured"]([10, 11], [1, 2],
        SimpleNamespace(decode=lambda ids: str(ids[0])))
    assert capsys.readouterr().out == "\033[0m10\033[0m\033[92m11\033[0m\n"
    parser = runpy.run_path("script/tah/evaluation/eval.py", run_name="eval_test")["_build_parser"]()
    args = parser.parse_args(["--eval_config", "c", "--model_path", "m", "--dataset_name", "math500"])
    assert args.backend == "tah_engine"
    assert backends._SETUP_BY_NAME["tah"] is backends.setup_tah


@pytest.mark.parametrize("workers", [0, 2])
def test_mixed_callback_harness_durable_results(monkeypatch, tmp_path, workers):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a: Tokenizer())
    calls = []

    def infer(prompts, on_result):
        calls.append(prompts)
        for i in (1, 0):
            on_result(i, dict(text="\\boxed{2}" if i == 0 else "Answer: A", seconds=.2,
                pass_counts=[1, 2], token_ids=[3, 4], prompt_pass_counts=[1],
                finish_reason="length", timing={}))
            raw = list(tmp_path.rglob("sample_0.json"))
            assert len(raw) == (1 if i == 1 else 2)
            assert all(json.loads(p.read_text())["output"] for p in raw)
    infer.supports_callback = True
    monkeypatch.setattr(jobs, "setup_backend", lambda *a, **kw: (None, infer))
    data = [dict(id=f"{ds}_0", question="question", answer=answer,
                 _source_dataset=ds, _source_answer_type="boxed")
            for ds, answer in [("math500", "2"), ("gpqa", "A")]]
    jobs.run_single_job(config=dict(batch_size=16, repeat_size=1, grading_workers=workers),
        combined_dataset_name="math500_gpqa", dataset_names=["math500", "gpqa"],
        output_dir=str(tmp_path), timestamp="test", model_path="stub", job_id=0,
        job_nums=1, start_idx=0, end_idx=2, tp_size=1, backend="tah_engine",
        data_range=None, problems_data=data,
        field_mapping=dict(id_field="id", question_field="question", answer_field="answer"))
    assert len(calls) == 1 and len(calls[0]) == 2
    for path in tmp_path.rglob("detailed_results.csv"):
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1 and rows[0]["is_correct"] == "True"
        assert rows[0]["mean_passes_per_token"] == "1.500000"
    for path in tmp_path.rglob("sample_0.json"):
        raw = json.loads(path.read_text())
        assert raw["token_ids"] == [3, 4] and raw["pass_counts"] == [1, 2]
        assert "grading_status" not in raw
    metrics = list(tmp_path.rglob("job_0.metrics.json"))
    assert len(metrics) == 2 and json.loads(metrics[0].read_text())["grading_workers"] == workers
