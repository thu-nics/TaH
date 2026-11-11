"""Tiny CPU trajectories, lifecycle alignment, graders and paired analysis."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from script.tah.validation import coupled_analyze as analysis, coupled_run as runner
from script.tah.validation.common import load_reference, write_json
from script.tah.validation.coupling import uniform, uniforms
from tah.engine import Request, SamplingParams, TaHEngine
from tah.engine.sampling import GraphSampler
from tah.engine.tests.test_model import reference


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU fixture initialised CUDA"))
    torch.set_num_threads(2)


@pytest.fixture
def tiny(tmp_path):
    # Two equiprobable supported tokens: bf16 reference and engine samplers
    # have exactly the same distribution, so identical CPU arithmetic must
    # yield identical trajectories (no bf16-vs-fp64 filtering confound).
    root = tmp_path / "model"
    model = reference(root, max_iter=3, label=False, tied=False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("threshold"):
                p.fill_(-1)
            elif "embed_tokens" in name:
                p.fill_(1)
            elif "norm" in name and name.endswith("weight"):
                p.fill_(1)
            elif "lm_head" in name and name.endswith("weight"):
                p.fill_(-1)
                p[:2].zero_()
            else:
                p.zero_()
    model.save_pretrained(root)
    return root


def fixture_manifest():
    problems = [dict(id=f"p{i}", dataset="math500" if i < 4 else "gpqa", index=i % 4,
                     seed=42 + i, prompt=[3, 7] if i % 2 else [3, 7, 11], answer="A" if i >= 4 else "1")
                for i in range(8)]
    return dict(version=1, protocol="fixture", problems=problems, oracle_ids=[p["id"] for p in problems],
                eos_token_id=None, pad_token_id=0)


def tokenizer(eos=None):
    return SimpleNamespace(eos_token_id=eos, decode=lambda ids: " ".join(map(str, ids)))


def test_real_reference_batch1_batch8_engine_identical(tiny):
    manifest = fixture_manifest()
    problems = manifest["problems"]
    model = load_reference(tiny, None, "bf16", "cpu")
    reference_logits = []
    hook = model.register_forward_hook(lambda m, a, o: reference_logits.append(o.logits[:, -1].clone()))
    b8, _ = runner.reference_batch(model, tokenizer(), problems, 0, 7)
    hook.remove()
    b1 = [runner.reference_batch(model, tokenizer(), [p], 0, 7)[0][0] for p in problems]
    assert b1 == b8
    engine = TaHEngine(tiny, device="cpu", kv_capacity=1000, max_batch_size=8, eos_token_id=None,
                       prefill_chunk_size=3, cuda_graphs=True, overlap=True)
    engine_logits = [[] for _ in problems]

    class Observer:
        def prefill(self, state, start, output):
            if start + len(output.logits) == len(state.prompt):
                engine_logits[state.index].append(output.logits[-1].clone())

        def decode(self, states, bucket, status):
            for state, logits, row in zip(states, bucket.logits, status):
                if not bool(row[0]):
                    engine_logits[state.index].append(logits.clone())

    engine._observer = Observer()
    result = engine.generate([Request(p["prompt"], 7, p["seed"]) for p in problems],
                             SamplingParams(temperature=.6, top_p=.95, top_k=20))
    for row, (e, r) in enumerate(zip(result, b8)):
        assert e.token_ids == r["token_ids"]
        assert e.generated_pass_counts == r["pass_counts"]
        assert e.prompt_pass_counts == r["prompt_pass_counts"]
        assert all(torch.equal(logits, reference_logits[j][row])
                   for j, logits in enumerate(engine_logits[row][:7]))
        assert len(engine_logits[row]) == 7
    assert engine.last_run["overlap"]


@pytest.mark.parametrize("overlap", [False, True])
def test_engine_uniform_position_swap_restart(tiny, monkeypatch, overlap):
    engine = TaHEngine(tiny, device="cpu", kv_capacity=1000, max_batch_size=2, eos_token_id=None,
                       prefill_chunk_size=3, cuda_graphs=True, overlap=overlap)
    requests = [Request([3, 7], 7, 42), Request([5, 8], 7, 43)]
    seen = {42: [], 43: []}
    draws = {42: [], 43: []}
    sampled = GraphSampler.sample
    prefill = engine._prefill
    reset = []

    def observed_sample(self, logits, params, rng, active=None):
        active = torch.ones_like(rng.counters, dtype=torch.bool) if active is None else active
        # Every active engine draw is at the next generated-position index,
        # independent of intermediate passes and overlap host lag/ghost rows.
        for seed, counter, enabled in zip(rng.seeds.tolist(), rng.counters.tolist(), active.tolist()):
            if enabled:
                assert counter == len(seen[seed])
                assert uniforms([seed], [counter]).item() == uniform(seed, len(seen[seed]))
                seen[seed].append(counter)
        result = sampled(self, logits, params, rng, active=active)
        for seed, token, enabled in zip(rng.seeds.tolist(), result.tolist(), active.tolist()):
            if enabled:
                draws[seed].append(token)
        return result

    def observed_prefill(state):
        if state.restarts:
            reset.append((state.seed, seen[state.seed].copy()))
            seen[state.seed].clear()
            draws[state.seed].clear()
            assert state.generated == [] and state.position == 0 and state.depth == 1
        result = prefill(state)
        assert int(engine.counters[state.row]) == 1
        assert int(engine.seeds[state.row]) == state.seed
        return result

    monkeypatch.setattr(GraphSampler, "sample", observed_sample)
    engine._prefill = observed_prefill
    emit = engine._sampled

    def emitted(state, token, count):
        j = len(state.generated)
        assert seen[state.seed][j] == j
        assert draws[state.seed][j] == token
        return emit(state, token, count)

    engine._sampled = emitted
    events = []
    restored = []
    restore = engine.swap.restore

    def observed_restore(state, *args):
        before = [t.clone() for t in (state.snapshot.logits, state.snapshot.embeddings,
                                     state.snapshot.seeds, state.snapshot.counters)]
        result = restore(state, *args)
        assert all(torch.equal(t[state.row], value) for t, value in zip(engine.device_state, before))
        restored.append(state.seed)
        return result

    engine.swap.restore = observed_restore

    def hook(eng, running, swapped, free_rows):
        if len(events) >= 2 or eng.last_run["steps"] < 4:
            return
        if not events:
            victim = running[-1]
            before = [t[victim.row].clone() for t in eng.device_state]
            eng._swap_out(victim, running, swapped, free_rows)
            assert all(torch.equal(t, value) for t, value in zip(
                (victim.snapshot.logits, victim.snapshot.embeddings, victim.snapshot.seeds,
                 victim.snapshot.counters), before))
            events.append(("swap", victim.seed))
            # Restoration happens on the next scheduler turn, after consuming
            # the pending overlap status, as in production.
        elif restored:
            victim = next(s for s in running if s.seed == 43)
            eng._restart(victim, running, swapped, free_rows, eng._restarted)
            events.append(("restart", victim.seed))

    engine._step_hook = hook
    actual = engine.generate(requests, SamplingParams(temperature=.6, top_p=.95, top_k=20))
    assert events == [("swap", 43), ("restart", 43)] and restored == [43] and reset and reset[0][1]
    assert engine.last_run["swap_outs"] == engine.last_run["swap_ins"] == 1
    assert engine.last_run["restarts"] == 1
    # Final attempt consumes every uniform that the reference's step j uses.
    for result, request in zip(actual, requests):
        assert seen[request.seed][:7] == list(range(7))
        assert len(result.token_ids) == 7
    assert engine.pool.audit()["owned"] == 0


def test_reference_eos_last_step_and_trace(tiny):
    model = load_reference(tiny, None, "fp32", "cpu")
    p = fixture_manifest()["problems"][:1]
    first, _ = runner.reference_batch(model, tokenizer(), p, 0, 1, True)
    eos = first[0]["token_ids"][0]
    result, trace = runner.reference_batch(model, tokenizer(eos), p, 0, 1, True)
    assert result[0]["token_ids"] == []
    assert result[0]["sampled_token_ids"] == [eos]
    assert result[0]["finish_reason"] == "eos"
    assert len(trace[0]) == 1 and trace[0][0][-1] == 3


def test_oracle_batch8_batch2_identical_and_resume_guard(tiny, tmp_path):
    manifest = fixture_manifest()
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    saved = []
    base = ["run", "oracle", "--model", str(tiny), "--manifest", str(path),
            "--device", "cpu", "--max-new-tokens", "7"]
    for batch in (8, 2):
        out = tmp_path / f"oracle-b{batch}"
        # Omission exercises the default 8.
        argv = [*base, "--out", str(out), *(["--oracle-batch", str(batch)] if batch != 8 else [])]
        runner.main(argv)
        info = json.loads((out / "run.json").read_text())
        assert info["config"]["oracle_batch"] == info["config"]["batch"] == batch
        assert info["config"]["block_size"] == batch
        assert info["completed_blocks"] == list(range(8 // batch))
        saved.append([json.loads(runner.sample_path(out, p["id"]).read_text()) for p in manifest["problems"]])
        runner.main(argv)
        other = 2 if batch == 8 else 8
        with pytest.raises(ValueError, match="different configuration"):
            runner.main([*argv, "--oracle-batch", str(other)])
    assert saved[0] == saved[1]
    for p in manifest["problems"]:
        with np.load(runner.trace_path(tmp_path / "oracle-b8", p["id"])) as a:
            with np.load(runner.trace_path(tmp_path / "oracle-b2", p["id"])) as b:
                assert np.array_equal(a["details"], b["details"])


@pytest.mark.parametrize("implementation,batch", [("reference", "2"), ("engine", "4"),
                                                 ("oracle", "0"), ("oracle", "-1")])
def test_oracle_batch_rejects_invalid_or_nonoracle_use(implementation, batch):
    with pytest.raises(SystemExit):
        runner.parse(["run", implementation, "--model", "unused", "--manifest", "unused",
                      "--out", "unused", "--oracle-batch", batch])


@pytest.mark.parametrize("dataset,text,answer", [
    ("math500", r"The answer is \boxed{2}", "2"),
    ("gpqa", "The answer is (A).", "A"),
    ("math500", "no answer", "2"),
    ("gpqa", "The answer is (B).", "A"),
])
def test_same_harness_grading_for_all_implementations(dataset, text, answer):
    from tah.evaluate.jobs import _score_one
    p = dict(dataset=dataset, answer=answer)
    expected = _score_one(text, answer, dataset, is_code=False)
    grades = [runner.grade(p, text) for _ in ("engine", "reference", "oracle")]
    assert grades[0] == grades[1] == grades[2]
    assert tuple(grades[0].values()) == expected


def test_fixed_criteria_and_prefixes():
    assert analysis.prefix([1, 2, 3], [1, 2, 4]) == 2
    assert analysis.prefix([1, 2], [1, 2, 3]) == 2
    assert analysis.prefix([], [1]) == 0
    assert analysis.c1([10] * 100, [10] * 100)["status"] == "PASS"
    assert analysis.c1([8] * 100, [10] * 100)["status"] == "not shown"
    at_margin = analysis.c1([9] * 100, [10] * 100)
    assert at_margin["status"] == "PASS" and at_margin["difference_lower95"] == -1
    assert analysis.c2([True] * 398, [True] * 398)["status"] == "PASS"
    assert analysis.c2([True] * 398, [False] * 398)["status"] == "not shown"
    assert analysis.c2([True], [True])["status"] == "not shown"
    d = analysis.c2([1] * 198 + [0] * 200, [0] * 198 + [1] * 200)
    assert d["status"] == "not shown"  # small estimate, wide interval
    assert analysis.c1([7, 8, 9], [8, 8, 8]) == analysis.c1([7, 8, 9], [8, 8, 8])


def test_cdf_margin_known_boundary():
    p = torch.tensor([[.5, .3, .2]], dtype=torch.float64)
    v = runner.cdf_details(p, torch.tensor([.51]), torch.tensor([1]))[0]
    assert v[:3].tolist() == [.5, .8, 1.]
    assert abs(v[4] - .01) < 1e-7


def test_cli_pipeline_resume_and_determinism(tiny, tmp_path):
    manifest = fixture_manifest()
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    runs = {}
    for impl in ("engine", "reference", "oracle"):
        runs[impl] = tmp_path / impl
        argv = ["run", impl, "--model", str(tiny), "--manifest", str(path), "--out", str(runs[impl]),
                "--device", "cpu", "--max-new-tokens", "4", "--block", "8"]
        if impl == "oracle":
            argv.extend(["--oracle-batch", "2"])
        runner.main(argv)
        info = json.loads((runs[impl] / "run.json").read_text())
        assert info["config"]["batch"] == (2 if impl == "oracle" else 8)
        files = {p: p.read_bytes() for p in runs[impl].rglob("*") if p.is_file()}
        runner.main(argv)
        assert all(p.read_bytes() == value for p, value in files.items())
        # A crash leaves incomplete block samples. They must all be replaced,
        # with the original batch, rather than dropping completed rows.
        info = json.loads((runs[impl] / "run.json").read_text())
        info.update(completed_blocks=[], status="running")
        write_json(runs[impl] / "run.json", info)
        runner.sample_path(runs[impl], "p0").write_text("corrupt partial sample")
        runner.main(argv)
        assert json.loads(runner.sample_path(runs[impl], "p0").read_text())["id"] == "p0"
        with pytest.raises(ValueError, match="different configuration"):
            runner.main([*argv, "--max-new-tokens", "5"])
    outputs = []
    for i in range(2):
        out = tmp_path / f"analysis{i}"
        analysis.main(["--manifest", str(path), "--engine", str(runs["engine"]),
                       "--reference", str(runs["reference"]), "--oracle", str(runs["oracle"]),
                       "--output", str(out)])
        outputs.append([(out / f"r1o2.{suffix}").read_bytes() for suffix in ("json", "md")])
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0][0])
    assert report["comparisons"]["engine_reference"]["identical_fraction"] == 1
    assert report["C1"]["status"] == report["C2"]["status"] == "PASS"
    assert report["conclusion"] == "not shown" and not report["full_workload"]
    # Inject one divergence and require its oracle CDF margin in both artifacts.
    r = json.loads(runner.sample_path(runs["engine"], "p0").read_text())
    r["token_ids"][1] = 3
    r["sampled_token_ids"][1] = 3
    write_json(runner.sample_path(runs["engine"], "p0"), r)
    report = analysis.analyze(manifest, runs["engine"], runs["reference"], runs["oracle"])
    divergence = report["oracle_divergences"][0]
    assert divergence["id"] == "p0" and divergence["sampled_position"] == 1
    assert 0 <= divergence["normalized_margin"] <= .5
    assert "p0" in analysis.markdown(report)
    # Truncated trajectories are censoring, not an unobserved near-tie.
    r["token_ids"] = r["token_ids"][:1]
    r["sampled_token_ids"] = r["sampled_token_ids"][:1]
    r["pass_counts"] = r["pass_counts"][:1]
    write_json(runner.sample_path(runs["engine"], "p0"), r)
    report = analysis.analyze(manifest, runs["engine"], runs["reference"], runs["oracle"])
    assert report["oracle_divergences"][0]["kind"] == "length_censored"
    # Analysis refuses incomplete runs.
    info = json.loads((runs["engine"] / "run.json").read_text())
    info["status"] = "running"
    write_json(runs["engine"] / "run.json", info)
    with pytest.raises(ValueError, match="incomplete"):
        analysis.analyze(manifest, runs["engine"], runs["reference"], runs["oracle"])


def test_cpu_guard_never_loads_real_checkpoint(tmp_path):
    model = tmp_path / "large"
    model.mkdir()
    write_json(model / "config.json", dict(hidden_size=2048, vocab_size=152000))
    path = tmp_path / "manifest.json"
    write_json(path, fixture_manifest())
    with pytest.raises(ValueError, match="tiny fixture"):
        runner.main(["run", "oracle", "--model", str(model), "--manifest", str(path),
                     "--out", str(tmp_path / "out"), "--device", "cpu"])


def test_prepare_harness_workload_and_fixed_seeds(tmp_path, monkeypatch):
    import transformers
    import tah.evaluate.datasets as datasets

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return "chat:" + messages[0]["content"]

        def encode(self, text):
            return [3, len(text) % 97]

    rows = [dict(id=f"{ds}_{j}", _source_dataset=ds, _original_index=j, question=f"{ds} question {j}", answer="1")
            for ds, n in (("math500", 500), ("gpqa", 198)) for j in range(n)]
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a: Tokenizer())
    monkeypatch.setattr(datasets, "load_combined_dataset", lambda names: (rows, {}))
    output = tmp_path / "manifest.json"
    runner.prepare("unused", output)
    manifest = json.loads(output.read_text())
    assert len(manifest["problems"]) == 398
    assert [p["seed"] for p in manifest["problems"]] == list(range(42, 440))
    assert manifest["problems"][200]["id"] == "gpqa_0"
    assert len(manifest["oracle_ids"]) == 100
    assert manifest["problems"][0]["prompt_text"] == "chat:math500 question 0"
    before = output.read_bytes()
    runner.prepare("unused", output)
    assert output.read_bytes() == before
    manifest["problems"][0]["seed"] = 99
    with pytest.raises(ValueError, match="seed mapping"):
        runner.validate_manifest(manifest)
    with pytest.raises(ValueError, match="temperature"):
        runner.read_recipe(None, False)
    cfg = runner.read_recipe(runner.RECIPE, False)
    assert cfg["max_new_tokens"] == 4096 and cfg["top_k"] == 20


def test_engine_runner_close_releases_host_arena(tmp_path):
    # The runner's engine patches form reference cycles, so del alone would keep
    # the (pinned on CUDA) host swap arena until the next gc; close() must not.
    import gc
    import weakref
    from script.tah.validation.engine_run import EngineRunner
    reference(tmp_path, max_iter=3, label=False)
    gc.collect()
    gc.disable()
    try:
        for _ in range(3):
            engine_runner = EngineRunner(tmp_path, device="cpu", kv_capacity=50, max_batch_size=3,
                                  host_swap_bytes=20000, eos_token_id=None)
            arena = weakref.ref(engine_runner.engine.host_arena)
            engine_runner.close()
            assert arena().closed and arena().buffer is None
            del engine_runner
    finally:
        gc.enable()
