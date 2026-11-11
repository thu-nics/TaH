"""CPU distribution/semantics checks; GPU tests are in test_sampling_gpu.py."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from tah.engine.sampling import GraphSampler, RequestRNG, SamplingParams, request_uniform
from tah.model.utils import sample_next_token


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    initialized_before = torch.cuda.is_initialized()
    torch.set_num_threads(4)

    def forbidden(*args, **kwargs):
        raise AssertionError("sampling CPU tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert torch.cuda.is_initialized() == initialized_before, "CPU test changed CUDA initialization state"


def exact_target(logits, params):
    """Independent target with score then token-ID ordering for cutoff ties."""
    if params.greedy:
        return torch.zeros_like(logits.float()).scatter(-1, logits.argmax(-1, keepdim=True), 1)
    p = torch.softmax(logits.float() / params.temperature, -1)
    if params.min_p > 0:
        p[p < params.min_p * p.max(-1, keepdim=True).values] = 0
        p /= p.sum(-1, keepdim=True)
    # Deterministically order equal raw scores by token index, then keep the
    # same prefix for top-k and stable nucleus ties.
    raw_order = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)
    p = p.gather(-1, raw_order)
    p, prob_order = torch.sort(p, dim=-1, descending=True, stable=True)
    token_order = raw_order.gather(-1, prob_order)
    if params.top_k > 0:
        positions = torch.arange(p.shape[-1], device=p.device)[None, :]
        p = torch.where(positions < min(params.top_k, p.shape[-1]), p, 0)
        p /= p.sum(-1, keepdim=True)
    if params.top_p < 1:
        remove = p.double().cumsum(-1) > params.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        p[remove] = 0
        p /= p.sum(-1, keepdim=True)
    return torch.zeros_like(p).scatter(-1, token_order, p)


def exact_uniform_top_p_target(vocab, k, top_p, *, device="cpu"):
    """FP64 nucleus oracle for a uniform top-k tie under the documented rule."""
    top_p = float(torch.tensor(top_p, dtype=torch.float32))
    ranked = torch.full((1, k), 1.0 / k, dtype=torch.float64, device=device)
    remove = ranked.cumsum(-1) > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    ranked[remove] = 0
    ranked /= ranked.sum(-1, keepdim=True)
    return torch.zeros((1, vocab), dtype=torch.float32, device=device).scatter(
        -1, torch.arange(k, device=device)[None, :], ranked.float())


CASES = [
    ([2., 1., 0., -1., -2., -3.], SamplingParams()),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(temperature=.6)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(temperature=2)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(min_p=.2)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(top_k=2)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(top_k=999)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(top_p=.7)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(temperature=.8, min_p=.05, top_k=4, top_p=.8)),
    # k larger than the surviving min-p candidates, not merely larger than V.
    ([4., 2., 0., -2., -4., -6.], SamplingParams(min_p=.2, top_k=5, top_p=.9)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(top_p=0)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(top_p=1e-8)),
    ([2., 1., 0., -1., -2., -3.], SamplingParams(top_p=1 - 1e-6)),
    ([0., 0., 0., 0., 0., 0.], SamplingParams(top_k=2)),
    ([0., 0., 0., 0., 0., 0.], SamplingParams(top_k=2, top_p=.4)),
    ([2., 1., 1., 1., 0., 0.], SamplingParams(top_k=2, top_p=.8)),
    ([2., 1., 1., 1., 0., 0.], SamplingParams(min_p=1)),
    ([2., 2., 1., 0., -1., -2.], SamplingParams(temperature=0, top_p=0)),
    ([2., 2., 1., 0., -1., -2.], SamplingParams(do_sample=False, min_p=1)),
    ([-float("inf"), 1., -float("inf"), 0., -5., -float("inf")], SamplingParams(top_p=.9)),
    ([10000., 9999., -10000., -10000., -10000., -10000.], SamplingParams()),
]


@pytest.mark.parametrize("case_index", range(len(CASES)))
def test_target_and_reference_filter_order(case_index):
    values, params = CASES[case_index]
    logits = torch.tensor([values])
    sampler = GraphSampler(logits.shape[-1])
    batch = sampler.make_batch([params])
    expected = exact_target(logits, params)
    torch.testing.assert_close(sampler.probabilities(logits, batch), expected, atol=0, rtol=0)
    if params.greedy:
        assert torch.equal(sample_next_token(logits, do_sample=False), expected.argmax(-1))
    elif case_index not in {12, 13, 14}:
        # The unchanged reference keeps every probability tie at top-k. The
        # simplified engine uses its documented exact-k tie rule instead.
        seen = []

        def intercept(p, num_samples):
            seen.append(p.clone())
            return p.argmax(-1, keepdim=True)

        with patch("torch.multinomial", intercept):
            sample_next_token(logits, **params.__dict__)
        torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)


def test_sampling_statistics():
    # Six marginal binomial standard deviations + one count, chosen before
    # running. Zero target mass must receive exactly zero samples. These are
    # CPU development checks, independent of any model parity fixture/gate.
    draws = 65536
    results = []
    for i, (values, params) in enumerate(CASES):
        logits = torch.tensor([values]).expand(draws, -1)
        sampler = GraphSampler(len(values))
        batch = sampler.make_batch([params] * draws)
        # Independent per-request seeds; also exercise a consecutive counter stream.
        rng = RequestRNG.from_seeds(list(range(draws)))
        stream = "independent request seeds"
        if i % 2:
            rng.seeds.fill_(1234)
            rng.counters.copy_(torch.arange(draws))
            stream = "consecutive counters of one request"
        sampled = sampler.sample(logits, batch, rng)
        expected = exact_target(logits[:1], params)[0].double()
        actual = torch.bincount(sampled, minlength=len(values)).double() / draws
        limits = 6 * (expected * (1 - expected) / draws).sqrt() + 1 / draws
        assert not actual[expected == 0].any()
        assert torch.all((actual - expected).abs() <= limits), (i, actual, expected, limits)
        results.append({"case": i, "draws": draws, "stream": stream,
                        "max_abs_probability_error": float((actual - expected).abs().max()),
                        "total_variation": float((actual - expected).abs().sum() / 2),
                        "target": expected.tolist(), "observed": actual.tolist()})
    output = Path("data/engine/wp5/cpu-statistics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"production_gates_eligible": False,
                                 "tolerance": "6 binomial SE + 1/draws; zero mass exact",
                                 "cases": results}, indent=2) + "\n")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_per_row_params_greedy_and_inactive(dtype):
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(128, 97, generator=generator).to(dtype)
    logits[0, :2] = 100  # lowest-index argmax tie
    params = [SamplingParams(temperature=0), SamplingParams(do_sample=False)]
    params += [SamplingParams(temperature=.6 + i / 100, min_p=.1 * (i % 3),
                              top_k=[0, 1, 5, 150][i % 4], top_p=[0, .7, 1][i % 3])
               for i in range(126)]
    sampler = GraphSampler(97)
    batch = sampler.make_batch(params)
    expected = torch.cat([exact_target(row[None], p) for row, p in zip(logits, params)])
    torch.testing.assert_close(sampler.probabilities(logits, batch), expected, atol=2e-7, rtol=2e-6)
    rng = RequestRNG.from_seeds(list(range(128)))
    active = torch.arange(128) % 4 != 3
    for _ in range(3):
        ids = sampler.sample(logits, batch, rng, active)
        assert torch.equal(ids[:2], logits[:2].argmax(-1))
        assert torch.all(ids[~active] == -1)
        assert torch.all(expected[torch.arange(128)[active], ids[active]] > 0)
    assert rng.counters[:2].tolist() == [0, 0]
    assert torch.equal(rng.counters[2:], (active[2:] * 3).long())
    assert ids[0] == 0


def splitmix_python(seed, counter):
    mask = (1 << 64) - 1
    z = (seed + 0x9E3779B97F4A7C15 * (counter + 1)) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    z ^= z >> 31
    return ((z >> 12) + .5) * 2 ** -52


def test_rng_known_integer_oracle_and_open_interval():
    seeds = [0, 1, -1, (1 << 63) - 1, -(1 << 63), 81923]
    counters = [0, 1, 27, 9999999, (1 << 63) - 2, 6]
    rng = RequestRNG.from_seeds(seeds)
    rng.counters.copy_(torch.tensor(counters))
    expected = torch.tensor([splitmix_python(s, c) for s, c in zip(seeds, counters)],
                            dtype=torch.float64)
    assert torch.equal(request_uniform(rng), expected)
    assert torch.all((expected > 0) & (expected < 1))


def test_request_reordering_pausing_reset_and_global_rng_isolation():
    generator = torch.Generator().manual_seed(28)
    logits = torch.randn(4, 31, generator=generator)
    params = [SamplingParams(top_k=i + 2, top_p=.8) for i in range(4)]
    sampler = GraphSampler(31, max_top_k=5)
    batch = sampler.make_batch(params)
    rng = RequestRNG.from_seeds([91, 26, 3, -6])
    original_seeds = rng.seeds.clone()
    global_rng = torch.random.get_rng_state()
    baseline = torch.stack([sampler.sample(logits, batch, rng) for _ in range(128)])
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.equal(rng.seeds, original_seeds)
    order = torch.tensor([2, 0, 3, 1])
    reordered_batch = sampler.make_batch([params[i] for i in order.tolist()])
    reordered_rng = RequestRNG(original_seeds[order], torch.zeros(4, dtype=torch.int64))
    candidate = torch.stack([sampler.sample(logits[order], reordered_batch, reordered_rng)
                             for _ in range(128)])
    assert torch.equal(baseline[:, order], candidate)
    rng.counters.zero_()
    assert torch.equal(sampler.sample(logits, batch, rng), baseline[0])
    before = rng.counters.clone()
    assert torch.all(sampler.sample(logits, batch, rng, torch.zeros(4, dtype=torch.bool)) == -1)
    assert torch.equal(before, rng.counters)
    assert torch.equal(sampler.sample(logits, batch, rng), baseline[1])


@pytest.mark.parametrize("k", [1, 2, 20, 128])
def test_bounded_algorithm_against_stable_full_target(k):
    # Run the GPU algorithm's math on CPU; stable full-sort oracle emulates
    # PyTorch 2.6 CUDA large-V sort. This does NOT verify its CUDA implementation.
    gen = torch.Generator().manual_seed(441)
    logits = torch.randn(16, 8193, generator=gen)
    logits[:8] = (logits[:8] * 2).round()  # many ties; k-overflow ties included
    logits[8] = 0  # whole vocabulary is a tie, much larger than the shortlist
    logits[9, 3:] = -float("inf")  # fewer candidates than k
    logits[10] = torch.arange(8193) % 4
    params = [SamplingParams(temperature=.7, min_p=[0, .3, 1][i % 3],
                             top_k=max(1, k - i % k),
                             top_p=[0, 1e-8, .1, .95, 1 - 1e-6, 1][i % 6])
              for i in range(16)]
    sampler = GraphSampler(8193, max_top_k=k)
    batch = sampler.make_batch(params)
    ids, probs = sampler._bounded_distribution(logits, batch)
    actual = torch.zeros_like(logits).scatter(-1, ids, probs.float())
    targets = []
    for row, p in zip(logits, params):
        if p.min_p == 1 and torch.equal(row, row[:1].expand_as(row)):
            # For an exact tied nucleus boundary, use the documented stable
            # score/token rule in FP64 instead of inheriting CPU FP32 drift.
            targets.append(exact_uniform_top_p_target(
                logits.shape[-1], min(p.top_k, logits.shape[-1]), p.top_p))
        else:
            targets.append(exact_target(row[None], p))
    expected = torch.cat(targets)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_bounded_nucleus_at_exact_cdf_boundary():
    logits = torch.zeros(8, 8193)
    pvals = [0, 1 / 8193, 20 / 8193, .5, .75, .999, 1 - 1e-7, 1]
    params = [SamplingParams(top_k=20, top_p=p) for p in pvals]
    sampler = GraphSampler(8193, max_top_k=20)
    batch = sampler.make_batch(params)
    ids, probs = sampler._bounded_distribution(logits, batch)
    actual = torch.zeros_like(logits).scatter(-1, ids, probs.float())
    expected = torch.cat([exact_uniform_top_p_target(logits.shape[-1], 20, p.top_p)
                          for p in params])
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_static_shapes_and_no_host_scalar_operations(monkeypatch):
    sampler = GraphSampler(97, max_top_k=20)
    params = [SamplingParams(top_k=20, top_p=.95), SamplingParams(temperature=0)]
    batch = sampler.make_batch(params)
    rng = RequestRNG.from_seeds([9, 4])
    logits = torch.arange(194, dtype=torch.float32).view(2, 97)

    def forbidden(*args, **kwargs):
        raise AssertionError("decode sampling must not synchronize to host or create ragged shapes")

    for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__int__", "nonzero"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    monkeypatch.setattr(torch, "nonzero", forbidden)
    monkeypatch.setattr(torch, "multinomial", forbidden)
    first = sampler.sample(logits, batch, rng)
    assert first.shape == (2,) and first.dtype == torch.int64
    assert sampler.probabilities(logits, batch).shape == (2, 97)
    bounded_ids, bounded_probs = sampler._bounded_distribution(logits, batch)
    assert bounded_ids.shape == bounded_probs.shape == (2, 20)


@pytest.mark.parametrize("kwargs", [dict(temperature=-1), dict(temperature=float("nan")),
                                   dict(top_p=1.1), dict(min_p=-.1), dict(top_k=-1),
                                   dict(top_k=2.5), dict(top_k=True), dict(do_sample=1)])
def test_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_admission_updates_keep_addresses_and_validate_bounds():
    sampler = GraphSampler(97, max_top_k=20)
    batch = sampler.make_batch([SamplingParams(top_k=20), SamplingParams(temperature=0)])
    addresses = [getattr(batch, key).data_ptr() for key in
                 ("temperature", "min_p", "top_k", "top_p", "do_sample")]
    batch.update([SamplingParams(top_k=3, top_p=.5), SamplingParams(do_sample=False)])
    assert addresses == [getattr(batch, key).data_ptr() for key in
                         ("temperature", "min_p", "top_k", "top_p", "do_sample")]
    for p in (SamplingParams(top_k=0), SamplingParams(top_k=21)):
        with pytest.raises(ValueError, match="bounded sampler"):
            batch.update([p, p])
    with pytest.raises(ValueError, match="shape"):
        batch.update([SamplingParams(top_k=2)])
    with pytest.raises(ValueError, match="seeds"):
        RequestRNG.from_seeds([1 << 63])


def test_general_fallback_bucket_accepts_topk_zero_and_above_bound():
    sampler = GraphSampler(97, max_top_k=20)
    params = [SamplingParams(top_k=0), SamplingParams(top_k=21)]
    batch = sampler.make_batch(params)
    assert not batch.bounded
    logits = torch.randn(2, 97, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(
        sampler.probabilities(logits, batch),
        torch.cat([exact_target(row[None], p) for row, p in zip(logits, params)]),
        atol=2e-7, rtol=2e-6)


def test_tiny_reference_constructor_logits():
    # No fixture dependency; CPU reference constructor through the frozen stack.
    from tah.engine.tests.test_fixtures import build

    model = build(label=True)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[3, 7, 11, 17]]),
                       iter_count_labels=torch.tensor([[1, 2, 3, 1]])).logits[0]
    params = [SamplingParams(temperature=.6, min_p=.1, top_k=20, top_p=.95),
              SamplingParams(temperature=0), SamplingParams(top_p=.5),
              SamplingParams(top_k=200)]
    sampler = GraphSampler(97)
    actual = sampler.probabilities(logits, sampler.make_batch(params))
    expected = torch.cat([exact_target(row[None], p) for row, p in zip(logits, params)])
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_greedy_specialization_is_argmax_without_filter_or_rng(monkeypatch):
    sampler = GraphSampler(151936, greedy_only=True)
    batch = sampler.make_batch([SamplingParams(temperature=0)] * 8)
    logits = torch.randn(8, 151936, generator=torch.Generator().manual_seed(28)).bfloat16()
    logits[0, :2] = 100
    rng = RequestRNG.from_seeds(list(range(8)))

    def forbidden(*args, **kwargs):
        raise AssertionError("greedy specialization should only call argmax")

    monkeypatch.setattr(sampler, "probabilities", forbidden)
    expected = logits.argmax(-1)
    assert torch.equal(sampler.sample(logits, batch, rng), expected)
    assert torch.all(rng.counters == 0)
    with pytest.raises(ValueError, match="greedy_only"):
        batch.update([SamplingParams()] * 8)
