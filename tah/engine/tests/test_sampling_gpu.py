"""Deferred CUDA checks. Hidden CUDA skips before querying/initializing it."""
import os
from pathlib import Path
import json

import pytest
import torch

from tah.engine.sampling import GraphSampler, RequestRNG, SamplingParams, request_uniform
from tah.engine.tests.test_sampling import (CASES, exact_target,
                                            exact_uniform_top_p_target,
                                            splitmix_python)

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        pytest.skip("CPU-only session: CUDA is explicitly hidden")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(4)


def make_logits(batch_size, vocab_size):
    logits = torch.randn(batch_size, vocab_size, generator=torch.Generator().manual_seed(875))
    logits[:batch_size // 2] = (logits[:batch_size // 2] * 2).round()
    logits[0] = 0  # worst-case shortlist overflow: all V tokens tie
    if batch_size > 1:
        logits[1, 3:] = -float("inf")
    return logits.to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("bound", [None, 20])
@pytest.mark.parametrize("batch_size", [1, 8, 128])
def test_gpu_exact_filtered_target_and_ties(bound, batch_size):
    v = 151936
    logits = make_logits(batch_size, v)
    params = [SamplingParams(temperature=.6, min_p=[0, .01, .1, 1][i % 4],
                             top_k=[1, 2, 20, 17][i % 4],
                             top_p=[0, 1e-8, .1, .5, .95, 1 - 1e-6, 1][i % 7])
              for i in range(batch_size)]
    sampler = GraphSampler(v, max_top_k=bound)
    batch = sampler.make_batch(params, device="cuda")
    actual = sampler.probabilities(logits, batch)
    expected = torch.cat([exact_target(row[None], p) for row, p in zip(logits, params)])
    # Support is exact; fp32 probability arithmetic allows only a few ulps.
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_gpu_nucleus_boundary_and_rng_integer_oracle():
    v = 151936
    ps = [SamplingParams(top_k=20, top_p=p)
          for p in [0, 1 / v, 20 / v, .5, .75, .999, 1 - 1e-7, 1]]
    logits = torch.zeros(len(ps), v, device="cuda")
    sampler = GraphSampler(v, max_top_k=20)
    actual = sampler.probabilities(logits, sampler.make_batch(ps, device="cuda"))
    expected = torch.cat([exact_uniform_top_p_target(v, 20, p.top_p, device="cuda")
                          for p in ps])
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    seeds = [0, 1, -1, (1 << 63) - 1, -(1 << 63), 81923]
    counters = [0, 1, 27, 9999999, (1 << 63) - 2, 6]
    rng = RequestRNG.from_seeds(seeds, device="cuda")
    rng.counters.copy_(torch.tensor(counters, device="cuda"))
    expected_u = torch.tensor([splitmix_python(s, c) for s, c in zip(seeds, counters)],
                              dtype=torch.float64)
    assert torch.equal(request_uniform(rng).cpu(), expected_u)


def test_gpu_tied_shortlist_repeatability_across_k_buckets():
    v, top_k = 8193, 8
    logits = torch.zeros(1, v, device="cuda", dtype=torch.bfloat16)
    targets = []
    for bound in (8, 20, 32):
        sampler = GraphSampler(v, max_top_k=bound)
        batch = sampler.make_batch([SamplingParams(top_k=top_k)], device="cuda")
        first_ids, first_probs = sampler._bounded_distribution(logits, batch)
        again_ids, again_probs = sampler._bounded_distribution(logits, batch)
        assert torch.equal(first_ids, again_ids)
        assert torch.equal(first_probs, again_probs)
        target = sampler.probabilities(logits, batch)
        assert torch.equal(
            (target[0] > 0).nonzero().flatten(), torch.arange(top_k, device="cuda"))
        targets.append(target)
    assert all(torch.equal(targets[0], target) for target in targets[1:])


@pytest.mark.parametrize("bound,greedy,v", [(None, False, 97), (None, False, 8193),
                                          (20, False, 151936), (None, True, 151936)])
def test_capture_replay_equals_eager_with_mutable_device_state(bound, greedy, v):
    b = 8
    logits = make_logits(b, v)
    sampler = GraphSampler(v, max_top_k=bound, greedy_only=greedy)
    params = [SamplingParams(top_k=20, top_p=.95, temperature=0 if greedy else .6)
              for _ in range(b)]
    batch = sampler.make_batch(params, device="cuda")
    seeds = list(range(21, 21 + b))
    rng = RequestRNG.from_seeds(seeds, device="cuda")
    active = torch.ones(b, dtype=torch.bool, device="cuda")
    # Warmup on a side stream, then capture. Reset counters afterward so eager
    # and replay begin at the same request state despite warmup draws.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sampler.sample(logits, batch, rng, active)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sampler.sample(logits, batch, rng, active)
    rng.counters.zero_()
    eager_rng = RequestRNG.from_seeds(seeds, device="cuda")
    for step in range(32):
        if step == 8:
            # Exercise parameters, masks and seed changes through fixed buffers.
            params = [SamplingParams(top_k=i + 1, top_p=(i + 1) / 8,
                                     temperature=0 if greedy or i % 3 == 0 else .9)
                      for i in range(b)]
            batch.update(params)
            active.copy_(torch.arange(b, device="cuda") % 2 == 0)
            rng.seeds.add_(17)
            eager_rng.seeds.add_(17)
            logits.copy_(logits.flip(-1))
        expected = sampler.sample(logits, batch, eager_rng, active)
        graph.replay()
        assert torch.equal(captured, expected)
        assert torch.equal(rng.counters, eager_rng.counters)
        if greedy:
            assert torch.equal(captured[active], logits.argmax(-1)[active])
    # Reset demonstrates reusability rather than a frozen captured RNG draw.
    rng.counters.zero_()
    eager_rng.counters.zero_()
    graph.replay()
    assert torch.equal(captured, sampler.sample(logits, batch, eager_rng, active))


def test_gpu_statistical_distributions_graph_replay():
    b, draws, v = 128, 65536, 8193
    results = []
    for i, (values, p) in enumerate(CASES):
        logits = torch.full((b, v), -float("inf"), device="cuda")
        logits[:, :len(values)] = torch.tensor(values, device="cuda")
        # Bound=20 exercises the large-V fast path; k>=V uses the general path.
        sampler = GraphSampler(v, max_top_k=20 if 0 < p.top_k <= 20 else None)
        batch = sampler.make_batch([p] * b, device="cuda")
        rng = RequestRNG.from_seeds(list(range(1000, 1000 + b)), device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sampler.sample(logits, batch, rng)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = sampler.sample(logits, batch, rng)
        rng.counters.zero_()
        samples = torch.empty(draws // b, b, dtype=torch.int64, device="cuda")
        for step in range(draws // b):
            graph.replay()
            samples[step].copy_(captured)
        expected = exact_target(logits[:1], p)[0].double().cpu()
        observed = torch.bincount(samples.flatten(), minlength=v).double().cpu() / draws
        assert not observed[expected == 0].any()
        limit = 6 * (expected * (1 - expected) / draws).sqrt() + 1 / draws
        assert torch.all((observed - expected).abs() <= limit)
        results.append({"case": i, "draws": draws,
                        "max_abs_probability_error": float((observed - expected).abs().max()),
                        "total_variation": float((observed - expected).abs().sum() / 2)})
    root = Path("data/engine/wp5")
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "workload": "synthetic sampler distributions; development diagnostic only",
        "draws_per_case": draws,
        "production_gates_eligible": False,
        "results": results,
    }
    (root / "gpu-statistics.json").write_text(json.dumps(report, indent=2) + "\n")
