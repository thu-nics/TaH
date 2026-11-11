"""float64 SDPA workaround for PyTorch #143644 (CPU semantics; one short CUDA check)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from script.tah.validation import capture
from script.tah.validation.common import load_reference, write_json
from script.tah.validation.fp64_attention import POLICY, fp64_attention_guard
from script.tah.validation.reference_run import ReferenceRunner

MODEL = Path("data/engine/fixture-models/tiny/mlp3")
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="tiny fixture checkpoint absent")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA check")
MIN = torch.finfo(torch.float64).min


def qkv(length, rows=5, heads=3, batch=2, kv_heads=None, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, rows, 16, generator=g, dtype=torch.float64)
    k = torch.randn(batch, kv_heads or heads, length, 16, generator=g, dtype=torch.float64)
    v = torch.randn(batch, kv_heads or heads, length, 16, generator=g, dtype=torch.float64)
    return q.to(device), k.to(device), v.to(device)


def additive(rows, length, batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch, 1, rows, length, dtype=torch.float64)
    mask[torch.rand(batch, 1, rows, length, generator=g) < .3] = MIN
    mask[:, :, :, 0] = 0.
    return mask


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("length", [7, 513, 739])
def test_padding_preserves_attention_on_cpu(length):
    native = F.scaled_dot_product_attention
    q, k, v = qkv(length)
    tri = torch.ones(5, length, dtype=torch.bool).tril(length - 5)
    masks = dict(additive=additive(5, length), boolean=additive(5, length) == 0, two_dim=tri,
                 broadcast=additive(5, 1), none=None)
    with fp64_attention_guard() as guard:
        for mask in masks.values():
            close(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), native(q, k, v, attn_mask=mask))
        for rows in (length, 3):  # SDPA is_causal is the top-left triangle, also for rows < length
            qq = q[:, :, :1].expand(-1, -1, rows, -1).contiguous() + torch.arange(rows).double()[:, None]
            close(F.scaled_dot_product_attention(qq, k, v, is_causal=True), native(qq, k, v, is_causal=True))
        close(F.scaled_dot_product_attention(q, k, v, attn_mask=additive(5, length), scale=.3),
              native(q, k, v, attn_mask=additive(5, length), scale=.3))
        qg, kg, vg = qkv(length, heads=4, kv_heads=2)
        close(F.scaled_dot_product_attention(qg, kg, vg, enable_gqa=True), native(qg, kg, vg, enable_gqa=True))
        assert guard.padded == guard.calls == 9
    assert F.scaled_dot_product_attention is native


def test_fully_masked_rows_keep_native_semantics():
    native = F.scaled_dot_product_attention
    q, k, v = qkv(9)
    mask = additive(5, 9)
    mask[:, :, 2] = MIN  # equal scores: uniform over the original slots, not the padded one
    boolean = mask == 0
    boolean[:, :, 3] = False
    with fp64_attention_guard():
        close(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), native(q, k, v, attn_mask=mask))
        close(F.scaled_dot_product_attention(q, k, v, attn_mask=boolean), native(q, k, v, attn_mask=boolean))


def test_other_dtypes_and_even_lengths_pass_through_unchanged(monkeypatch):
    seen = []

    def spy(query, key, value, **kwargs):
        seen.append((query, key, value, kwargs))
        return query
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    q, k, v = qkv(8)
    mask = additive(5, 8)
    odd = [t.float() for t in qkv(7)]
    with fp64_attention_guard() as guard:
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        F.scaled_dot_product_attention(*odd, is_causal=True)
        F.scaled_dot_product_attention(*[t.bfloat16() for t in odd])
    assert F.scaled_dot_product_attention is spy
    assert (guard.calls, guard.padded) == (1, 0)
    assert seen[0][1] is k and seen[0][2] is v and seen[0][3]["attn_mask"] is mask
    assert seen[1][1] is odd[1] and seen[1][3]["is_causal"] is True
    assert seen[2][1].dtype == torch.bfloat16 and seen[2][1].shape[-2] == 7


def test_guard_restores_on_error_and_is_reentrant():
    native = F.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match="boom"):
        with fp64_attention_guard() as outer:
            with fp64_attention_guard() as inner:
                assert inner is outer
            assert F.scaled_dot_product_attention is outer
            raise RuntimeError("boom")
    assert F.scaled_dot_product_attention is native
    with pytest.raises(ValueError, match="exclusive"):
        with fp64_attention_guard():
            F.scaled_dot_product_attention(*qkv(7), attn_mask=additive(5, 7), is_causal=True)


@needs_model
def test_reference_runner_applies_guard_to_float64_only(monkeypatch):
    torch.set_num_threads(2)
    native = F.scaled_dot_product_attention
    seen = []

    def spy(query, key, value, **kwargs):
        seen.append((query.dtype, key.shape[-2]))
        return native(query, key, value, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    seq = dict(id="pad", prompt=[5, 9, 13, 21], forced=[40, 41, 42, 43, 44])
    for dtype in ("fp64", "fp32"):
        runner = ReferenceRunner(load_reference(MODEL, None, dtype, "cpu"))
        seen.clear()
        runner.run([seq], layout="chunk", chunk=3)
        assert seen and F.scaled_dot_product_attention is spy
        lengths = {length for _, length in seen}
        if dtype == "fp64":
            assert all(d == torch.float64 and length % 2 == 0 for d, length in seen)
        else:
            assert any(length % 2 for length in lengths)  # untouched native shapes
    monkeypatch.setattr(runner, "_run", lambda seqs, **kwargs: [])
    runner.model.double()
    with pytest.raises(AssertionError, match="bypassed"):
        runner.run([seq], layout="chunk", chunk=3)


def test_target_refuses_float64_oracle_without_workaround(tmp_path):
    config = dict(impl="reference", mode="free", layout="chunk", chunk=128, dtype="fp64")
    write_json(tmp_path / "old" / "run.json", dict(config=config))
    with pytest.raises(ValueError, match="#143644"):
        capture.Target(tmp_path / "old", "cpu")
    write_json(tmp_path / "new" / "run.json", dict(config=dict(config, fp64_attention=POLICY)))
    capture.Target(tmp_path / "new", "cpu")
    write_json(tmp_path / "fp32" / "run.json", dict(config=dict(config, dtype="fp32")))
    capture.Target(tmp_path / "fp32", "cpu")
    assert json.loads((tmp_path / "new" / "run.json").read_text())["config"]["fp64_attention"] == POLICY


def predicted_faulty(width):
    """Native torch<2.7 trigger: odd width 513..1023 whose 32-rounded block exceeds width + 1."""
    return width % 2 == 1 and 513 <= width <= 1023 and width % 32 != 31


@needs_cuda
def test_cuda_guarded_attention_matches_cpu_float64():
    native = F.scaled_dot_product_attention
    widths = list(range(505, 1101, 3)) + [739, 741, 767, 1023, 1024, 1025, 7281]
    faulty = []
    with fp64_attention_guard() as guard:
        for width in widths:
            q, k, v = qkv(width, rows=26, heads=4, batch=1, seed=width)
            mask = additive(26, width, batch=1, seed=width)
            expected = native(q, k, v, attn_mask=mask)
            gpu = [t.cuda() for t in (q, k, v, mask)]
            close(F.scaled_dot_product_attention(*gpu[:3], attn_mask=gpu[3]).cpu(), expected)
            if predicted_faulty(width):
                scores = (q @ k.transpose(-1, -2) * q.shape[-1] ** -.5 + mask).reshape(-1, width)
                wrong = (scores.cuda().softmax(-1).cpu() - scores.softmax(-1)).abs().max() > 1e-12
                faulty.append(bool(wrong))
        assert guard.padded == sum(w % 2 for w in widths)
    saved = Path("data/tmp/wp23/replay-diagnostic-deep-v2/standalone-a/layer0-sdpa.pt")
    if saved.exists():  # the real failing layer-0 inputs
        data = torch.load(saved, map_location="cpu")
        args = [data[name] for name in ("query", "key", "value", "mask")]
        kwargs = dict(scale=data["scale"], dropout_p=data["dropout_p"], is_causal=data["is_causal"])
        expected = native(*args[:3], attn_mask=args[3], **kwargs)
        with fp64_attention_guard():
            gpu = [a.cuda() for a in args]
            close(F.scaled_dot_product_attention(*gpu[:3], attn_mask=gpu[3], **kwargs).cpu(), expected)
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 7):
        assert np.mean(faulty) > .9, "native float64 softmax fault not reproduced; recheck the workaround premise"
