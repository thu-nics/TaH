"""CPU tests for observational capture, checkpoint round trips and metrics.

Run with CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python -m pytest
tah/engine/tests/test_fixtures.py. Tests never require generated artifacts.
"""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen3Config

from tah.engine.fixtures import FixtureSuite, SCHEMA_VERSION
from tah.engine.generate_fixtures import synthetic_config
from tah.engine.reference_capture import capture, compare, divergence, pack, summarize
from tah.model.tah_model import TaHForCausalLM


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)

    def forbidden(*args, **kwargs):
        raise AssertionError("fixture tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def build(*, max_iter=3, label=False):
    torch.manual_seed(91)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, tie_word_embeddings=max_iter != 1,
                         attention_dropout=0, use_sliding_window=False)
    base = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16,
                                           attn_implementation="sdpa")
    model = TaHForCausalLM(base, synthetic_config(SimpleNamespace(profile="tiny"),
                                                 max_iter=max_iter, label=label))
    model.to(dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.normal_(0, 0.03)
        if not label:
            model.iter_decider.threshold.fill_(-1)
    return model


def test_hooks_are_observational_and_hidden_index_zero_is_embedding():
    model = build()
    sequences = [[3, 4, 5, 6, 7], [12, 15]]
    inputs, columns = pack(sequences, 0, "cpu", extra=2)
    with torch.no_grad():
        original = model(**inputs)
    traces, logits = capture(model, sequences, extra=2, chunk_size=0)
    for row, col in enumerate(columns):
        assert torch.equal(logits[row], original.logits[row, col])
        assert torch.equal(traces[row]["pass_count"], original.iter_count[row, col])
        assert traces[row]["visited"].all()
        assert traces[row]["decision"][:2].all()
        assert not traces[row]["decision"][2].any()
        assert not traces[row]["continue_probability"][2].any()
        positions = traces[row]["hidden_positions"]
        expected = model.embed_tokens(torch.tensor(sequences[row]))[positions]
        assert torch.equal(traces[row]["decider_hidden"][0, :, 0], expected)
    assert not model.iter_decider._forward_hooks


def test_chunked_capture_matches_unchunked_and_padding():
    model = build(label=True)
    seq = [3, 7, 11, 13, 17, 19, 23, 29, 31]
    labels = [1, 2, 3, -100, 1, 3, 2, 1, 3]
    baseline, base_logits = capture(model, [seq], labels=[labels], chunk_size=0)
    candidate, candidate_logits = capture(model, [[14, 16], seq],
                                           labels=[[2, 1], labels], extra=4, chunk_size=3)
    assert torch.equal(base_logits[0], candidate_logits[1])
    assert torch.equal(baseline[0]["decision"], candidate[1]["decision"])
    assert baseline[0]["pass_count"].tolist() == [1, 2, 3, 1, 1, 3, 2, 1, 3]
    assert baseline[0]["decider_hidden"].shape[-2] == 0
    stopped, _ = capture(model, [seq], chunk_size=3)
    assert stopped[0]["pass_count"].tolist() == [1] * len(seq)


@pytest.mark.parametrize("max_iter,label", [(1, False), (3, False), (3, True)])
def test_reference_save_reload_retains_logits_and_random_lora(tmp_path, max_iter, label):
    model = build(max_iter=max_iter, label=label)
    # Keep a real positive threshold consistent with saved constructor kwargs.
    if not label:
        model.iter_decider.threshold.data.fill_(0.5)
    sequences = [[3, 8, 12, 18]]
    labels = [[3, 1, 2, -100]] if label else None
    traces, logits = capture(model, sequences, labels=labels)
    model.save_pretrained(str(tmp_path))
    restored = TaHForCausalLM.from_pretrained(str(tmp_path), device_map="cpu",
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(dtype=torch.bfloat16).eval()
    after, after_logits = capture(restored, sequences, labels=labels)
    assert torch.equal(logits[0], after_logits[0])
    assert torch.equal(traces[0]["decision"], after[0]["decision"])
    for name, parameter in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert torch.equal(parameter, dict(restored.named_parameters())[name])


def test_noise_counts_missing_depths_and_uses_full_vocab():
    model = build(label=True)
    base, logits = capture(model, [[3, 7, 11]], labels=[[3, 2, 1]])
    candidate = {key: value.clone() for key, value in base[0].items()}
    candidate["visited"][2, 0] = False
    candidate["decision"][0, 1] = False
    altered = logits[0].clone()
    # Change a low-ranked vocabulary coordinate, not just the saved top-k.
    index = int(altered[0].argmin())
    altered[0, index] += 0.125
    stats = compare(base[0], candidate, logits[0], altered, threshold=0.5)
    assert stats["decision_mismatches"] == 2
    assert stats["visited_mismatches"] == 1
    assert stats["logit_max_abs"] > 0
    summary = summarize([stats])
    assert summary["g_tf_logit_mae_limit"] == 1.5 * stats["logit_mae"]
    assert summary["retained_dense_error"]["logit_mae"] > 0


def test_loader_and_eos_divergence(tmp_path):
    model = build(max_iter=1)
    traces, _ = capture(model, [[3, 4, 5]])
    save_file(traces[0], str(tmp_path / "record.safetensors"))
    record = {"file": "record.safetensors", "kind": "teacher_forced", "checkpoint": "test"}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "status": "complete", "records": [record]}))
    suite = FixtureSuite(tmp_path)
    assert len(list(suite.records(checkpoint="test"))) == 1
    assert torch.equal(suite.load(record)["input_ids"], traces[0]["input_ids"])
    assert divergence([1, 2], [1, 2]) is None
    assert divergence([1, 2], [1, 2, 3]) == 2
    assert divergence([1, 2], [1, 5]) == 1
