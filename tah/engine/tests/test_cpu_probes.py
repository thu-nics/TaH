"""Opt-in CPU probes; these never establish production gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tah.engine.generate_fixtures import labels_for, load_reference
from tah.engine.model import EagerTaHModel
from tah.engine.parity import compare_output
from tah.engine.reference_capture import capture


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(4)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU probes may not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def probe(name, root, ids, labels, filename):
    recipe = Path("script/tah/recipes/qwen3_1.7/eval_tah.yaml")
    if not root.exists():
        pytest.skip(f"prepared checkpoint missing: {root}")
    engine = EagerTaHModel.from_pretrained(root, yaml.safe_load(recipe.read_text()) if name == "real" else None)
    oracle = load_reference(SimpleNamespace(device="cpu", eval_config=recipe), name, root)
    traces, logits = capture(oracle, [ids], labels=[labels] if labels is not None else None, chunk_size=0)
    out = engine(ids, iter_count_labels=labels, hidden_positions=traces[0]["hidden_positions"])
    metrics = compare_output(traces[0], out, logits[0], float(engine.iter_decider.threshold))
    torch.testing.assert_close(out.logits, logits[0], rtol=0, atol=0)
    torch.testing.assert_close(out.decider_hidden, traces[0]["decider_hidden"], rtol=0, atol=0, equal_nan=True)
    assert metrics["decision_mismatches"] == 0
    assert metrics["pass_count_mismatches"] == 0
    destination = Path("data/engine/wp3/cpu-pytest")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / filename).write_text(json.dumps({"production_gates_eligible": False,
        "gate": "CPU development only", "length": len(ids), "metrics": metrics}, indent=2))
    assert not torch.cuda.is_initialized()


@pytest.mark.slow
@pytest.mark.parametrize("name", ["real", "mlp3", "mlp1", "label3"])
def test_short_production_weights_on_cpu(name):
    root = Path("models/TaH-plus-1.7B") if name == "real" else Path("data/engine/fixture-models/full") / name
    ids = [3, 7, 11, 13, 17, 19, 23, 29]
    labels = [1, 2, 3, -100, 1, 2, 3, 1] if name == "label3" else None
    probe(name, root, ids, labels, f"production-{name}.json")


@pytest.mark.slow
def test_long_tiny_uninitialized_reference_on_cpu():
    ids = (torch.arange(4096) % 94 + 3).tolist()
    probe("label3", Path("data/engine/fixture-models/tiny/label3"), ids, labels_for(ids), "long-label3.json")
