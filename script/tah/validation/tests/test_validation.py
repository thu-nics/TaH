"""CPU tests of the oracle measurement tooling on the tiny fixture checkpoints."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from script.tah.validation import analyze, capture
from script.tah.validation.common import inputs_of, load_record, load_reference, reference_distribution
from script.tah.validation.coupling import inverse_cdf_draw, multinomial_patch, uniform, uniforms
from script.tah.validation.reference_run import ReferenceRunner

MODEL = Path("data/engine/fixture-models/tiny/mlp3")
EOS = 2
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="tiny fixture checkpoint absent "
                                 "(python -m tah.engine.generate_fixtures --profile tiny --prepare-only)")


def make_manifest(path, count=36, seed=7):
    rng = random.Random(seed)
    sequences = []
    for i in range(count):
        prompt = [rng.randrange(3, 97) for _ in range(rng.randrange(4, 11))]
        forced = [rng.randrange(3, 97) for _ in range(rng.randrange(2, 18))]
        finish = "eos" if i % 3 == 0 else "length"
        if finish == "eos":
            forced.append(EOS)
        sequences.append(dict(id=f"s{i:03d}", backend="engine" if i % 2 else "reference",
                              dataset="a" if i % 4 < 2 else "b", finish=finish, prompt=prompt, forced=forced))
    manifest = dict(version=1, eos_token_id=EOS, pad_token_id=0, sequences=sequences,
                    subsets=dict(validity=[s["id"] for s in sequences[:8]]))
    path.write_text(json.dumps(manifest))
    return manifest


# ---------------------------------------------------------------- coupling

def test_uniform_matches_engine_counter_rng():
    seeds = [0, 42, -1, -(1 << 63), (1 << 63) - 1, 123456789]
    for seed in seeds:
        expected = uniforms([seed] * 50, list(range(50)))
        assert [uniform(seed, j) for j in range(50)] == expected.tolist()
        assert ((expected > 0) & (expected < 1)).all()


def test_inverse_cdf_reproduces_multinomial_distribution():
    torch.manual_seed(0)
    p = torch.rand(97, dtype=torch.float64) ** 4
    p[torch.randperm(97)[:60]] = 0
    p = p / p.sum()
    n = 200_000
    u = uniforms([42] * n, list(range(n)))
    draws = inverse_cdf_draw(p.expand(n, -1), u)
    freq = torch.bincount(draws, minlength=97).double() / n
    reference = torch.bincount(torch.multinomial(p.float(), n, replacement=True), minlength=97).double() / n
    sigma = (p * (1 - p) / n).sqrt()
    assert (freq[p == 0] == 0).all()
    assert ((freq - p).abs() <= 5 * sigma + 1e-12).all()
    assert ((freq - reference).abs() <= 7 * sigma + 1e-12).all()


def test_inverse_cdf_matches_engine_rule_and_patch():
    from tah.engine.sampling import GraphSampler, RequestRNG, SamplingParams
    torch.manual_seed(1)
    logits = torch.randn(64, 97)
    sampler = GraphSampler(97)
    batch = sampler.make_batch([SamplingParams(temperature=.6, top_k=20, top_p=.95)] * 64)
    seeds = list(range(100, 164))
    rng = RequestRNG.from_seeds(seeds)
    engine_tokens = sampler.sample(logits, batch, rng)
    probs = sampler.probabilities(logits, batch)
    mine = inverse_cdf_draw(probs, uniforms(seeds, [0] * 64))
    # Same distribution and uniform; the fp32 diagnostic copy may move an exact CDF tie.
    assert (mine == engine_tokens).float().mean() > .95
    from tah.model.utils import sample_next_token
    with multinomial_patch(lambda b: uniforms(seeds[:b], [0] * b)):
        patched = sample_next_token(logits, temperature=.6, top_p=.95, top_k=20)
    assert patched.shape == (64,)


# ---------------------------------------------------------------- reference layouts

@needs_model
def test_chunk_runner_equals_reference_capture_in_same_dtype():
    from tah.engine.reference_capture import capture as reference_capture
    model = load_reference(MODEL, None, "bf16", "cpu")
    seq = dict(id="x", prompt=[5, 9, 13, 21, 34], forced=[40, 41, 42, 43, 44, 45, 46, EOS])
    ids, first = inputs_of(seq)
    traces, logits = reference_capture(model, [ids], chunk_size=4)
    rec = ReferenceRunner(model).run([seq], layout="chunk", chunk=4)[0].arrays()
    assert np.array_equal(rec["visited"], traces[0]["visited"].numpy())
    assert np.array_equal(rec["decision"], traces[0]["decision"].numpy())
    prob = traces[0]["continue_probability"].double().numpy()
    prob[-1] = np.nan
    assert np.array_equal(rec["prob"], prob, equal_nan=True)
    assert np.array_equal(rec["top1"], logits[0][first:].float().argmax(-1).numpy())
    # Oracle precision is a different computation of the same model.
    oracle = ReferenceRunner(load_reference(MODEL, None, "fp32", "cpu"))
    assert oracle.threshold == ReferenceRunner(model).threshold
    a = oracle.run([seq], layout="chunk", chunk=4)[0].arrays()
    b = oracle.run([seq], layout="chunk", chunk=4)[0].arrays()
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in ("prob", "gap"))


@needs_model
def test_generate_layout_reproduces_reference_generate_loop():
    """Coupled TaHForCasualLM_generate tokens, then the teacher-forced b3 layout sees
    exactly the distributions and pass counts the generate loop saw."""
    from tah.model.utils import TaHForCasualLM_generate
    from tah.engine.reference_capture import pack
    model = load_reference(MODEL, None, "bf16", "cpu")
    prompts = [[5, 9, 13, 21], [7, 8, 9, 10, 11, 12], [30, 31]]
    seeds, steps, seen = [11, 12, 13], [0], []

    def provider(b):
        u = uniforms(seeds, [steps[0]] * b)
        steps[0] += 1
        return u
    original = torch.multinomial
    with multinomial_patch(provider):
        patched = torch.multinomial

        def spy(p, *a, **k):
            seen.append(p.clone())
            return patched(p, *a, **k)
        torch.multinomial = spy
        try:
            inputs, _ = pack(prompts, 0, "cpu", side="left")
            tokens, _, counts = TaHForCasualLM_generate(
                model, SimpleNamespace(eos_token_id=None, decode=lambda t: ""), inputs, max_new_tokens=6,
                temperature=.6, top_p=.95, top_k=20, verbose=False, return_pass_counts=True)
        finally:
            torch.multinomial = patched
    assert torch.multinomial is original
    seqs = [dict(id=str(i), prompt=p, forced=t) for i, (p, t) in enumerate(zip(prompts, tokens))]
    recs = [r.arrays() for r in ReferenceRunner(model).run(seqs, layout="generate", pad_id=0)]
    for row, rec in enumerate(recs):
        assert rec["visited"][:, rec["first"]:].sum(0).tolist() == counts[row]
        for j in range(len(tokens[row])):
            p = seen[j][row].double()
            p = (p / p.sum()).numpy()
            ids = rec["dist_ids"][rec["ptr"][j]:rec["ptr"][j + 1]]
            assert np.array_equal(np.nonzero(p)[0], np.sort(ids))
            assert np.allclose(p[ids], rec["dist_p"][rec["ptr"][j]:rec["ptr"][j + 1]], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- end-to-end pipeline

def run_capture(*argv):
    capture.main([str(a) for a in argv])


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("tiny fixture checkpoint absent")
    root = tmp_path_factory.mktemp("oracle")
    manifest = root / "manifest.json"
    make_manifest(manifest)
    common = ["--manifest", manifest, "--model", MODEL, "--device", "cpu", "--buffer-dir", root / "buffer"]
    engine = ["--kv-capacity", 8000, "--block", 36, "--max-batch-size", 32, "--prefill-chunk-size", 4]
    runs = dict(oracle=root / "oracle32", ref=root / "ref-b8-forced", ref_free=root / "ref-b8-free",
                engine=root / "engine-forced", engine_free=root / "engine-free",
                engine_small=root / "engine-forced-small", ref_small=root / "ref-b8-forced-small",
                oracle64=root / "oracle64", v32=root / "fp32-vs-64", vref=root / "ref-vs-64")
    run_capture("reference", *common, "--dtype", "fp32", "--layout", "chunk", "--chunk", 5, "--mode", "free",
                "--out", runs["oracle"])
    run_capture("reference", *common, "--mode", "forced", "--target", runs["oracle"], "--out", runs["ref"])
    run_capture("reference", *common, "--mode", "free", "--out", runs["ref_free"])
    run_capture("engine", *common, *engine, "--mode", "forced", "--target", runs["oracle"], "--out", runs["engine"])
    run_capture("engine", *common, *engine, "--mode", "free", "--out", runs["engine_free"])
    # A tiny dump budget forces several replays per block.
    run_capture("engine", *common, *engine, "--mode", "forced", "--target", runs["oracle"],
                "--buffer-gb", 1e-6, "--out", runs["engine_small"])
    run_capture("reference", *common, "--mode", "forced", "--target", runs["oracle"],
                "--buffer-gb", 1e-6, "--out", runs["ref_small"])
    sub = ["--subset", "validity"]
    run_capture("reference", *common, *sub, "--dtype", "fp64", "--layout", "chunk", "--chunk", 5, "--mode", "free",
                "--out", runs["oracle64"])
    run_capture("reference", *common, *sub, "--dtype", "fp32", "--layout", "chunk", "--chunk", 7,
                "--mode", "forced", "--target", runs["oracle64"], "--out", runs["v32"])
    run_capture("reference", *common, *sub, "--mode", "forced", "--target", runs["oracle64"], "--out", runs["vref"])
    return SimpleNamespace(root=root, manifest=manifest, runs=runs,
                           ids=[s["id"] for s in json.loads(manifest.read_text())["sequences"]])


def test_forced_decisions_identical_and_errors_finite(pipeline):
    for name in ("ref", "engine"):
        for sid in pipeline.ids:
            oracle = load_record(pipeline.runs["oracle"], sid)
            rec = load_record(pipeline.runs[name], sid)
            assert np.array_equal(rec["visited"], oracle["visited"])
            assert np.array_equal(rec["decision"], oracle["decision"])
            assert len(rec["top1"]) == len(oracle["top1"]) == len(rec["sq"])
            assert np.isfinite(rec["sq"]).all() and np.isfinite(rec["maxabs"]).all()
            assert (rec["sq"] > 0).any()
            ok = rec["visited"] & ~np.isnan(oracle["prob"])
            assert np.isfinite(rec["prob"][ok]).all()


def test_replays_bitwise_identical(pipeline):
    for big, small in (("engine", "engine_small"), ("ref", "ref_small")):
        info = json.loads((pipeline.runs[small] / "run.json").read_text())
        assert max(b["replays"] for b in info["blocks"]) > 1
        for sid in pipeline.ids:
            a, b = load_record(pipeline.runs[big], sid), load_record(pipeline.runs[small], sid)
            assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


def test_engine_runs_with_32_resident_mixed_depth_rows(pipeline):
    info = json.loads((pipeline.runs["engine"] / "run.json").read_text())
    stats = info["blocks"][0]["stats"]
    assert "32" in stats["rows_histogram"] and stats["steps_mixed_depth"] > 0


def test_free_mode_own_decisions(pipeline):
    flips = 0
    for sid in pipeline.ids:
        rec = load_record(pipeline.runs["engine_free"], sid)
        assert np.array_equal(rec["decision"], rec["own_decision"])
        oracle = load_record(pipeline.runs["oracle"], sid)
        flips += int((rec["decision"] != oracle["decision"]).sum())
    assert flips >= 0


def test_analysis_deterministic_with_validity(pipeline):
    r = pipeline.runs
    argv = ["--manifest", pipeline.manifest, "--oracle", r["oracle"], "--reference", r["ref"],
            "--engine", r["engine"], "--reference-free", r["ref_free"], "--engine-free", r["engine_free"],
            "--validity-oracle", r["oracle64"], "--validity-fp32", r["v32"], "--validity-reference", r["vref"]]
    outputs = []
    for k in range(2):
        out = pipeline.root / f"analysis{k}"
        analyze.main([str(a) for a in argv + ["--output", out]])
        outputs.append(((out / "r1o1.json").read_bytes(), (out / "r1o1.md").read_bytes()))
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0][0])
    for name in ("M1", "M2", "M3"):
        m = report["primary"][name]
        assert np.isfinite([m["engine"], m["reference"], m["ratio"], m["ratio_upper95"]]).all()
        assert m["status"] in ("PASS", "not shown")
    assert report["validity"]["sequences"] == 8
    assert report["conclusion"] in ("PASS", "not shown")


def test_bootstrap_is_seeded():
    a, b = analyze.bootstrap_indices(17), analyze.bootstrap_indices(17)
    assert np.array_equal(a, b) and a.shape == (analyze.RESAMPLES, 17)
    num, den = np.ones(17), np.full(17, 2.)
    assert analyze.ratio_bound(num, den, a) == (.5, .5)


@needs_model
def test_oracle_keeps_bf16_model_constants():
    reference = load_reference(MODEL, None, "bf16", "cpu")
    for dtype in ("fp32", "fp64"):
        oracle = load_reference(MODEL, None, dtype, "cpu")
        ref_buffers = dict(reference.named_buffers())
        inv = [(name, value) for name, value in oracle.named_buffers() if "inv_freq" in name]
        assert inv
        for name, value in inv:
            assert torch.equal(value, ref_buffers[name].to(value.dtype))
            assert torch.equal(value.to(torch.bfloat16).view(torch.int16), ref_buffers[name].view(torch.int16))
        for name, value in oracle.named_parameters():
            assert torch.equal(value, dict(reference.named_parameters())[name].to(value.dtype))
        assert float(oracle._active_iter_decider.threshold) == float(reference._active_iter_decider.threshold)


def test_paired_bias_identical_passes_systematic_bias_fails():
    ref = [dict(m2_n=10, bias_sum=float(x), m2_sum=abs(float(x))) for x in range(-10, 11)]
    idx = analyze.bootstrap_indices(len(ref))
    assert analyze.bias_check(ref, ref, idx)["status"] == "PASS"
    biased = [dict(r, bias_sum=r["bias_sum"] + 100) for r in ref]
    assert analyze.bias_check(biased, ref, idx)["status"] == "not shown"


def test_reference_distribution_is_sparse_and_normalized():
    torch.manual_seed(3)
    rows = reference_distribution(torch.randn(5, 97).to(torch.bfloat16))
    for ids, p in rows:
        assert 1 <= len(ids) <= 20 and abs(p.sum() - 1) < 1e-12


def test_replay_diagnostics_are_exact_and_handle_nan_shape_and_decisions():
    expected = dict(visited=np.ones((2, 4), bool), decision=np.zeros((2, 4), bool),
                    prob=np.array([[.1, .2, .3, .4], [np.nan]*4]), top1=np.array([3, 4]), gap=np.array([.2, .4]))
    replay = {k: v.copy() for k, v in expected.items()}
    assert capture.replay_differences(expected, replay) == {}
    replay['prob'][0, 2] = np.nextafter(replay['prob'][0, 2], np.inf)
    replay['decision'][0, 1] = True
    replay['gap'][0] = np.nan
    replay['top1'] = np.array([3])
    diff = capture.replay_differences(expected, replay)
    assert set(diff) == {'prob', 'decision', 'gap', 'top1'}
    assert diff['prob']['count'] == 1 and diff['prob']['first_index'] == [0, 2]
    assert 0 < diff['prob']['max_abs_finite_difference'] < 1e-15
    assert diff['decision']['count'] == 1
    assert diff['gap']['nonfinite_mismatches'] == 1
    assert diff['top1']['actual_shape'] == [1]


@needs_model
def test_target_replays_free_oracle_and_keeps_strict_identity(tmp_path, monkeypatch):
    from script.tah.validation.common import LogitDump, save_record, write_json
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('CPU replay test initialized CUDA'))
    seq = dict(id='replay', prompt=[5, 9, 13, 21], forced=[40, 41, 42, 43, 44])
    model = load_reference(MODEL, None, 'fp64', 'cpu')
    runner = ReferenceRunner(model)
    original = runner.run([seq], layout='chunk', chunk=3)[0].arrays()
    oracle = tmp_path/'oracle'
    write_json(oracle/'run.json', dict(config=dict(impl='reference', mode='free', layout='chunk', chunk=3)))
    save_record(oracle, seq['id'], original)
    target = capture.Target(oracle, 'cpu')
    target.model = runner
    calls = []
    original_run = runner.run

    def record(*args, **kwargs):
        calls.append(kwargs.get('forced'))
        return original_run(*args, **kwargs)
    monkeypatch.setattr(runner, 'run', record)
    arrays = {seq['id']: {k: v.copy() for k, v in original.items()}}
    dump = LogitDump(tmp_path/'buffer', [seq], model.config.vocab_size, torch.float64)
    try:
        original_run([seq], layout='chunk', chunk=3,
                     consumer=lambda s, indices, logits: dump.write(s['id'], indices, logits))
        target.compare([seq], dump, arrays)
        assert calls == [None]
        assert np.array_equal(arrays[seq['id']]['sq'], np.zeros(len(seq['forced'])))
        # Even one ULP must still fail. The repair changes replay mode, not its acceptance rule.
        altered = {k: v.copy() for k, v in original.items()}
        altered['prob'][0, 0] = np.nextafter(altered['prob'][0, 0], np.inf)
        save_record(oracle, seq['id'], altered)
        with pytest.raises(AssertionError, match='strict oracle replay failed.*prob'):
            target.compare([seq], dump, arrays)
    finally:
        dump.close()
