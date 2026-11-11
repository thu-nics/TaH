"""CPU development parity, configuration/weight integrity, and layout checks."""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, Qwen3Config

from tah.engine.config import resolve_tah_config
from tah.engine.fixtures import FixtureSuite
from tah.engine.generate_fixtures import labels_for, load_reference
from tah.engine.model import EagerTaHModel
from tah.engine.parity import assert_gtf, compare_output, require_production
from tah.engine.reference_capture import capture, summarize
from tah.engine.tests.test_gtf import replay_layout
from tah.model.tah_config import TaHConfig
from tah.model.tah_model import TaHForCausalLM, _resolve_tah_config


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    initialized_before = torch.cuda.is_initialized()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError("WP3 CPU tests must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert torch.cuda.is_initialized() == initialized_before, "CPU test changed CUDA initialization state"


def reference(tmp_path, *, max_iter=3, label=True, normalize=False, targets=None,
              patterns=False, bias=False, tied=True, rope=None, layers=2, activation="silu", head_dim=8):
    torch.manual_seed(711)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, head_dim=head_dim,
                      num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=layers,
                      attention_bias=bias, tie_word_embeddings=tied, rope_scaling=rope,
                      max_position_embeddings=16, hidden_act=activation)
    base = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    adapter = dict(r=4, lora_alpha=12, lora_dropout=.1, use_rslora=True,
                   target_modules=targets or ["q_proj", "v_proj", "down_proj"], task_type="CAUSAL_LM")
    if patterns:
        adapter.update(rank_pattern={"self_attn.q_proj": 2}, alpha_pattern={"self_attn.q_proj": 7})
    if bias:
        adapter["bias"] = "all"
    decider = dict(max_iter=max_iter) if label else dict(topk=8, hidden_states_size=32,
        hidden_states_layer_nums=[0, 1, layers], hidden_dims=[12, 24, 12],
        expansion_factor=2, dropout_rate=.1, normalize_input=normalize, threshold=-1., max_iter=max_iter)
    config = TaHConfig(max_iter=max_iter, iter_decider="IterLabelDecider" if label else "MLPIterDecider",
        iter_decider_kwargs=decider, input_updater_kwargs={"topk": 100}, adapter_kwargs=adapter,
        train_loss="NextTokenPredLoss", eval_loss="NextTokenPredLoss")
    model = TaHForCausalLM(base, config).to(dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora" in name:
                param.normal_(0, .07)
    model.save_pretrained(str(tmp_path))
    # The persisted constructor/reload is the oracle used by WP2.
    oracle = TaHForCausalLM.from_pretrained(str(tmp_path), device_map="cpu", torch_dtype=torch.bfloat16,
                                          attn_implementation="sdpa").to(dtype=torch.bfloat16).eval()
    return oracle


@pytest.mark.parametrize("name", ["mlp3", "mlp1", "label3"])
def test_all_tiny_teacher_forced_records(name, tmp_path):
    root = Path(os.environ.get("TAH_CPU_FIXTURES", "data/engine/fixtures/cpu-tiny"))
    if not root.exists():
        pytest.skip("WP2 cpu-tiny fixtures not present")
    suite = FixtureSuite(root)
    assert suite.manifest["production_gates_eligible"] is False
    model = EagerTaHModel.from_pretrained(suite.manifest["checkpoints"][name]["path"])
    oracle = load_reference(SimpleNamespace(device="cpu", eval_config=Path("script/tah/recipes/qwen3_1.7/eval_tah.yaml")),
                            name, Path(suite.manifest["checkpoints"][name]["path"]))
    sequences = json.loads((suite.root / "inputs.json").read_text())["sequences"]
    rows = []
    for rec in suite.records(kind="teacher_forced", checkpoint=name):
        t = suite.load(rec)
        out = model(t["input_ids"], iter_count_labels=t.get("iter_count_labels"), hidden_positions=t["hidden_positions"])
        # Development tolerances chosen a priori: exact decisions/pass
        # counts, top-1 and retained logits/hidden inputs. Tiny CPU SDPA and
        # identical BF16 operators have no measured short-fixture noise.
        assert torch.equal(out.visited, t["visited"]), rec
        assert torch.equal(out.decision, t["decision"]), rec
        assert torch.equal(out.iter_count, t["pass_count"]), rec
        assert torch.equal(out.logits.argmax(-1).int(), t["final_top1"]), rec
        assert torch.equal(out.logits[t["logit_positions"]], t["final_logits_sample"]), rec
        assert torch.equal(out.logits.gather(-1, t["final_topk_indices"].long()), t["final_topk_values"]), rec
        torch.testing.assert_close(out.decider_hidden, t["decider_hidden"], rtol=0, atol=0, equal_nan=True)
        item = next(item for item in sequences if item["id"] == rec["sequence_id"])
        batch, target, side, extra = replay_layout(sequences, item, rec)
        trace, full_logits = capture(oracle, batch, side=side, extra=extra,
            labels=[labels_for(tokens) for tokens in batch] if name == "label3" else None,
            chunk_size=rec["chunk_size"])
        torch.testing.assert_close(out.logits, full_logits[target], rtol=0, atol=0)
        rows.append(compare_output(t, out, full_logits[target], suite.manifest["checkpoints"][name]["threshold"]))
    assert len(list(suite.records(kind="teacher_forced", checkpoint=name))) == 45
    destination = Path("data/engine/wp3/cpu-pytest")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}.json").write_text(json.dumps({"production_gates_eligible": False,
        "gate": "CPU development only", "summary": summarize(rows)}, indent=2))


CASES = [
    dict(max_iter=1, label=False, tied=False), dict(max_iter=2, label=False),
    dict(max_iter=5, label=False, normalize=True, layers=3),
    dict(max_iter=5, patterns=True, bias=True, targets="all-linear", tied=False),
    dict(max_iter=4, targets=["k_proj", "o_proj", "gate_proj", "up_proj"]),
    dict(max_iter=3, rope={"rope_type": "linear", "factor": 2.}),
    dict(max_iter=3, rope={"rope_type": "dynamic", "factor": 2.}),
    dict(max_iter=3, rope={"rope_type": "yarn", "factor": 2.}),
    dict(max_iter=3, rope={"rope_type": "longrope", "factor": 2., "short_factor": [1.]*4, "long_factor": [2.]*4}),
    dict(max_iter=3, rope={"rope_type": "llama3", "factor": 2., "low_freq_factor": 1., "high_freq_factor": 4., "original_max_position_embeddings": 8}),
    dict(max_iter=3, activation="gelu"), dict(max_iter=3, activation="gelu_new"),
    dict(max_iter=3, activation="gelu_pytorch_tanh"), dict(max_iter=3, activation="relu"),
]


@pytest.mark.parametrize("options", CASES)
def test_constructor_generality(tmp_path, options):
    oracle = reference(tmp_path, **options)
    engine = EagerTaHModel.from_pretrained(tmp_path)
    ids = torch.tensor([3, 7, 11, 13, 17, 19, 23, 29, 31, 5, 8, 12, 9, 40, 51, 69, 70, 81, 90, 2, 12])
    labels = torch.tensor([1, 2, 3, -100, 5, 4, 2] * 3)
    traces, logits = capture(oracle, [ids.tolist()], labels=[labels.tolist()], chunk_size=0)
    out = engine(ids, iter_count_labels=labels, hidden_positions=traces[0]["hidden_positions"])
    assert torch.equal(out.iter_count, traces[0]["pass_count"])
    assert torch.equal(out.decision, traces[0]["decision"])
    torch.testing.assert_close(out.logits, logits[0], atol=0, rtol=0)
    torch.testing.assert_close(out.decider_hidden, traces[0]["decider_hidden"], atol=0, rtol=0, equal_nan=True)
    # No hidden persistent KV/history or pass-dependent adapter toggle.
    repeated = engine(ids, iter_count_labels=labels)
    assert torch.equal(repeated.logits, out.logits)


def test_override_precedence_external_decider_and_eval_label(tmp_path):
    reference(tmp_path, label=False)
    raw = json.loads((tmp_path / "tah_config.json").read_text())
    raw["old_unused_public_field"] = "ignored by reference"
    (tmp_path / "tah_config.json").write_text(json.dumps(raw))
    kwargs = dict(raw["iter_decider_kwargs"], threshold=.9)
    override = TaHConfig(max_iter=4, iter_decider_kwargs=kwargs, eval_iter_decider="iter_decider")
    assert asdict(resolve_tah_config(tmp_path, override)) == asdict(_resolve_tah_config(str(tmp_path), override))
    # Checkpoint threshold is deliberately different; constructor args win.
    payload = torch.load(tmp_path / "iter_decider.bin", weights_only=True)
    payload["state_dict"]["threshold"] = torch.tensor(.1)
    torch.save(payload, tmp_path / "iter_decider.bin")
    engine = EagerTaHModel.from_pretrained(tmp_path, override)
    assert float(engine.iter_decider.threshold) == .8984375
    assert engine.iter_decider.max_iter == 3  # explicit decider kwargs win over wrapper max_iter=4
    external = tmp_path / "external"
    external.mkdir()
    torch.save(payload, external / "iter_decider.bin")
    kwargs["load_path"] = str(external)
    engine = EagerTaHModel.from_pretrained(tmp_path, {"iter_decider_kwargs": kwargs,
        "eval_iter_decider": "IterLabelDecider", "eval_iter_decider_kwargs": {"max_iter": 2}})
    assert engine([3, 4, 5], iter_count_labels=[3, 1, 2]).iter_count.tolist() == [2, 1, 2]
    assert engine([3, 4, 5]).iter_count.tolist() == [1, 1, 1]


def test_weight_shards_bin_and_rejection(tmp_path):
    reference(tmp_path)
    normal = EagerTaHModel.from_pretrained(tmp_path)([3, 4, 5], iter_count_labels=[3, 1, 2]).logits
    state = load_file(tmp_path / "model.safetensors")
    keys = sorted(state)
    mapping = {}
    for i, group in enumerate((keys[::2], keys[1::2])):
        filename = f"model-{i}.safetensors"
        save_file({k: state[k] for k in group}, tmp_path / filename)
        mapping.update({k: filename for k in group})
    (tmp_path / "model.safetensors").unlink()
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    assert torch.equal(EagerTaHModel.from_pretrained(tmp_path)([3, 4, 5], iter_count_labels=[3, 1, 2]).logits, normal)
    index.unlink()
    torch.save(state, tmp_path / "pytorch_model.bin")
    assert torch.equal(EagerTaHModel.from_pretrained(tmp_path)([3, 4, 5], iter_count_labels=[3, 1, 2]).logits, normal)
    del state[keys[0]]
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="missing Qwen3"):
        EagerTaHModel.from_pretrained(tmp_path)


@pytest.mark.parametrize("field,value", [("use_dora", True), ("modules_to_save", ["lm_head"]), ("fan_in_fan_out", True)])
def test_unsupported_adapter_fails(tmp_path, field, value):
    reference(tmp_path)
    path = tmp_path / "lora/adapter_config.json"
    cfg = json.loads(path.read_text())
    cfg[field] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unsupported LoRA"):
        EagerTaHModel.from_pretrained(tmp_path)


def test_embedding_target_fails_loudly_pending_semantics_resolution(tmp_path):
    oracle = reference(tmp_path, max_iter=1, targets=["embed_tokens"])
    with pytest.raises(ValueError, match="embedding-target LoRA"):
        EagerTaHModel.from_pretrained(tmp_path)
    ids = torch.tensor([[3, 7, 11, 13, 17, 19]])
    with torch.no_grad():
        first = oracle(input_ids=ids).logits.clone()
        second = oracle(input_ids=ids).logits.clone()
    difference = (first.float() - second.float()).abs()
    # Reference embeds before _set_lora_enabled(False). First call starts
    # enabled; max_iter=1 leaves it disabled for the second identical call.
    assert difference.max() > 0
    destination = Path("data/engine/wp3")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "reference-embedding-conflict.json").write_text(json.dumps({
        "production_gates_eligible": False, "max_iter": 1,
        "target_modules": ["embed_tokens"], "input_ids": ids.tolist(),
        "first_vs_second_max_abs": float(difference.max()),
        "first_vs_second_mae": float(difference.mean()),
        "top1_mismatches": int((first.argmax(-1) != second.argmax(-1)).sum()),
        "cause": "reference embeds before disabling LoRA, so pass-1 input depends on prior adapter state",
    }, indent=2))


def test_lm_head_target_is_rejected(tmp_path):
    reference(tmp_path, targets=["lm_head", "q_proj"])
    with pytest.raises(ValueError, match="lm_head LoRA is out of scope"):
        EagerTaHModel.from_pretrained(tmp_path)


def test_row_masked_delta_and_runtime_imports(tmp_path):
    reference(tmp_path)
    engine = EagerTaHModel.from_pretrained(tmp_path)
    layer = engine.model.layers[0].self_attn.q_proj
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    mask = torch.tensor([True, False, False, True])
    actual = layer(x, mask)
    assert torch.equal(actual[mask], layer(x[mask], True))
    assert torch.equal(actual[~mask], layer(x[~mask], False))
    code = """import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('transformers', 'peft') or name.startswith('tah.model'):
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tah.engine.model import EagerTaHModel
import torch
m = EagerTaHModel.from_pretrained(sys.argv[1])
assert m([3,4,5]).logits.shape == (3,97)
assert not any(k.startswith(('transformers','peft','tah.model')) for k in sys.modules)
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"))


def test_fresh_engine_import_and_existing_public_exports():
    code = """import sys
import tah.engine
assert not any(k.startswith(('transformers', 'peft', 'tah.model')) for k in sys.modules)
import tah
expected = ['IterDeciderLoss', 'IterLabelDecider', 'MLPIterDecider', 'NextTokenPredLoss',
            'TaHCache', 'TaHCausalLMOutputWithPast', 'TaHConfig', 'TaHForCausalLM']
assert tah.__all__ == expected
from tah import TaHForCausalLM
from tah.model.tah_model import TaHForCausalLM as reference
assert TaHForCausalLM is reference
import importlib
for name in expected:
    assert getattr(tah, name) is getattr(importlib.import_module(tah._EXPORTS[name]), name)
    assert name in dir(tah)
namespace = {}
exec('from tah import *', namespace)
assert all(namespace[name] is getattr(tah, name) for name in expected)
import torch
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"))


@pytest.mark.parametrize("path", ["script/tah/train/SFT_TaH.py", "script/tah/playground/inference_example.py"])
def test_existing_entrypoint_imports(path):
    code = """import runpy, sys, torch
def forbidden(*args, **kwargs):
    raise AssertionError('entrypoint import initialized CUDA')
torch.cuda._lazy_init = forbidden
runpy.run_path(sys.argv[1], run_name='wp3_import_check')
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", code, path], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"))


def test_eval_entrypoint_help():
    subprocess.run([sys.executable, "script/tah/evaluation/eval.py", "--help"],
        check=True, capture_output=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2"))


def test_binding_gate_rejects_tiny_and_zero_noise_has_no_floor():
    suite = type("Suite", (), {"manifest": {"production_gates_eligible": False}})()
    with pytest.raises(AssertionError, match="production fixtures"):
        require_production(suite)
    # A real comparison exercises full-vocabulary metrics and strict limits.
    # Reuse an on-disk tiny trace only as a unit test of gate arithmetic.
    root = Path("data/engine/fixtures/cpu-tiny")
    if not root.exists():
        return
    fixtures = FixtureSuite(root)
    t = fixtures.load(next(fixtures.records(kind="teacher_forced", checkpoint="mlp1")))
    engine = EagerTaHModel.from_pretrained(fixtures.manifest["checkpoints"]["mlp1"]["path"])
    out = engine(t["input_ids"], hidden_positions=t["hidden_positions"])
    row = compare_output(t, out, out.logits.clone(), .5)
    noise = summarize([row])
    assert_gtf([row], noise)
    altered = out.logits.clone()
    altered[0, altered[0].argmin()] += .125
    bad = compare_output(t, out, altered, .5)
    with pytest.raises(AssertionError):
        assert_gtf([bad], noise)
