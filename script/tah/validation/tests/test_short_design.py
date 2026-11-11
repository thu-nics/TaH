"""CPU coverage/append guards, capped CRN execution, tail inference and calibration."""
import copy
import json

import numpy as np
import pytest
import torch

from script.tah.validation import analyze, capture, coupled_analyze, coupled_run, near_tie, short_c1
from script.tah.validation.common import write_json
from script.tah.validation.tests.test_coupled import tiny as tiny  # noqa: F811


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test initialized CUDA"))
    torch.set_num_threads(2)


def tail_rows(n=8):
    return [dict(id=f"p{i}", **{k: dict(margins=np.linspace(.1, 1., 30)*(1+i/10),
                                       events=np.arange(30)) for k in near_tie.KINDS}) for i in range(n)]


def test_tail_paired_identical_shift_and_order_guard():
    rows = tail_rows()
    result = near_tie.compare(rows, rows, resamples=500)
    assert result["conclusion"] == "PASS"
    worse = copy.deepcopy(rows)
    for row in worse:
        for kind in near_tie.KINDS:
            row[kind]["margins"] *= 1.5
    assert near_tie.compare(worse, rows, resamples=500)["conclusion"] == "not shown"
    with pytest.raises(ValueError, match="ordered sequence"):
        near_tie.compare(rows, rows[::-1])
    cor = near_tie.correlations(rows, rows)
    assert all(np.isclose(x["per_sequence_maxima_pearson"], 1.) and x["position_jaccard"] == 1 for x in cor.values())


def test_weighted_quantile_matches_explicit_cluster_replay():
    arrays = [np.array([1., 2.]), np.array([7., 8., 9.]), np.array([])]
    idx = np.array([[0, 0, 1], [1, 2, 2], [2, 2, 2], [0, 1, 2]])
    actual = near_tie.resampled_quantiles(arrays, idx)
    expected = []
    for sample in idx:
        data = np.concatenate([arrays[i] for i in sample])
        expected.append(np.quantile(data, .9, method="inverted_cdf") if len(data) else 0.)
    assert np.array_equal(actual, expected)
    empty = tail_rows()
    for row in empty:
        for kind in near_tie.KINDS:
            row[kind]["margins"] = np.array([])
    assert near_tie.compare(empty, empty, resamples=50)["conclusion"] == "not shown"


def test_self_split_distinguishes_power_and_false_alarms():
    a = near_tie.self_split(tail_rows(), splits=80, bootstrap=100)
    b = near_tie.self_split(tail_rows(), splits=80, bootstrap=100)
    assert a == b and a["diagnostic_only"] and a["exact_permutations"] == 70
    for row in a["rates"].values():
        assert "noninferiority_not_shown" in row and "boundary_false_noninferiority" in row
        assert row["equality_false_alarm"] <= .15
        assert .2 < row["max_not_shown"] < .8


def test_conclusion_requires_sizes_and_validity():
    r = dict(oracle=dict(dtype="fp32"), requirements=dict(sequences_at_least_200=True,
             positions_at_least_500k=True, free_mode_present=True),
             validity=dict(status="PASS: float32 oracle valid", size_requirement_met=True))
    assert analyze.qualified_conclusion(r, True, ["PASS"]) == "PASS"
    for key in r["requirements"]:
        bad = copy.deepcopy(r)
        bad["requirements"][key] = False
        assert analyze.qualified_conclusion(bad, True, ["PASS"]) != "PASS"
    for validity in ({}, dict(status="FAIL", size_requirement_met=True),
                     dict(status="PASS", size_requirement_met=False)):
        bad = dict(r, validity=validity)
        assert analyze.qualified_conclusion(bad, True, ["PASS"]) != "PASS"
    assert analyze.qualified_conclusion(r, False, ["missing"]) == "incomplete"


def test_c1_margin_pairing_and_censor_cap():
    ref = np.tile([20., 40., 100., 512.], 20)
    assert short_c1.bound(ref, ref)["status"] == "PASS"
    assert short_c1.bound(.8*ref, ref)["status"] == "not shown"
    assert short_c1.bound(ref, ref)["alpha"] == .025
    with pytest.raises(ValueError, match=r"\[0,512\]"):
        short_c1.bound([513., 10.], [512., 10.])
    assert coupled_analyze.prefix([1]*512, [1]*512) == 512
    assert coupled_analyze.prefix([1, 2], [1, 3]) == 1


def short_fixture():
    ps = [dict(id=f"p{i}", dataset=("math500" if i < 2 else "gpqa" if i < 4 else "gsm8k"),
               index=i, seed=42+i, prompt=[3, 7], answer="1") for i in range(6)]
    return dict(version=1, protocol=short_c1.PROTOCOL, fixture=True, max_new_tokens=512, prompt_limit=675,
                problems=ps, oracle_ids=[p["id"] for p in ps], eos_token_id=0, pad_token_id=0,
                subsets=dict(stage1=[p["id"] for p in ps[:4]], stage2=[p["id"] for p in ps[4:]]))


def test_short_c1_stages_streams_resume_and_fixture_cannot_pass(tiny, tmp_path, monkeypatch):  # noqa: F811
    cfg_path = tiny/"config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["max_position_embeddings"] = 1024
    write_json(cfg_path, cfg)
    manifest = short_fixture()
    path = tmp_path/"manifest.json"
    write_json(path, manifest)
    monkeypatch.setattr(coupled_run, "grade", lambda *args: dict(predicted_answer=None, has_answer=False, is_correct=False))
    configs = {}
    for stage in ("stage1", "stage2"):
        for impl in ("engine", "reference", "oracle"):
            out = tmp_path/stage/impl
            argv = ["run", impl, "--model", str(tiny), "--manifest", str(path), "--out", str(out),
                    "--device", "cpu", "--max-new-tokens", "512", "--block", "4", "--subset", stage]
            if impl == "oracle":
                argv += ["--oracle-batch", "2"]
            coupled_run.main(argv)
            before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
            coupled_run.main(argv)
            assert before == {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
            cfg, samples = coupled_analyze.load_run(out, manifest, impl)
            assert list(samples) == manifest["subsets"][stage]
            configs[stage, impl] = samples
        assert [s["sampled_token_ids"] for s in configs[stage, "engine"].values()] == [
            s["sampled_token_ids"] for s in configs[stage, "reference"].values()]
    (tmp_path/"stage1/oracle.path").write_text(str(tmp_path/"stage1/oracle"))
    with pytest.raises(ValueError, match="fixture"):
        short_c1.analyze(path, tmp_path, 1, tmp_path/"unused.json")
    with pytest.raises(ValueError, match="cap512"):
        coupled_run.main(["run", "engine", "--model", str(tiny), "--manifest", str(path),
                          "--out", str(tmp_path/"bad"), "--device", "cpu", "--subset", "stage1"])
    bad = copy.deepcopy(manifest)
    bad["problems"][0]["prompt"] = [1]*676
    with pytest.raises(ValueError, match="memory envelope"):
        short_c1.validate(bad)


def test_prepare_uses_prompt_only_exclusions_and_keeps_seeds(tmp_path, monkeypatch):
    import transformers
    import tah.evaluate.jobs as jobs

    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0

        def encode(self, text):
            return [1]*int(text)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args: Tokenizer())
    monkeypatch.setattr(jobs, "_make_prompt", lambda text, tok, is_code: text)
    for ds, count in (("math500", 4), ("gpqa", 4), ("gsm8k", 520)):
        file = tmp_path/f"{ds}_tah_engine/run/samples.jsonl"
        file.parent.mkdir(parents=True)
        file.write_text("".join(json.dumps(dict(id=f"{ds}_{i}", problem="676" if i == 0 else "10",
                                               correct_answer="1", is_correct=False))+"\n" for i in range(count)))
    output = tmp_path/"manifest.json"
    short_c1.prepare("unused", output, tmp_path, fixture=True)
    manifest = json.loads(output.read_text())
    assert len(manifest["subsets"]["stage1"]) == 6 and len(manifest["subsets"]["stage2"]) == 512
    assert len(manifest["excluded"]) == 3
    assert next(p for p in manifest["problems"] if p["id"] == "math500_1")["seed"] == 43
    before = output.read_bytes()
    short_c1.prepare("unused", output, tmp_path, fixture=True)
    assert output.read_bytes() == before
    manifest["problems"][0]["seed"] += 1
    write_json(output, manifest)
    with pytest.raises(ValueError, match="replace"):
        short_c1.prepare("unused", output, tmp_path, fixture=True)


def test_reverse_order_is_explicit_provenance():
    args = capture.parse(["engine", "--manifest", "m.json", "--model", "m", "--mode", "free",
                          "--out", "out", "--sequence-order", "reverse"])
    assert args.sequence_order == "reverse"


def test_reverse_capture_records_and_resume_guard(tiny, tmp_path):  # noqa: F811
    from script.tah.validation.common import load_record
    manifest = dict(version=1, pad_token_id=0, eos_token_id=None,
                    sequences=[dict(id="a", prompt=[3, 7], forced=[1, 1, 1]),
                               dict(id="b", prompt=[3, 7, 11], forced=[1, 1])])
    path = tmp_path/"teacher.json"
    write_json(path, manifest)
    argv = ["engine", "--manifest", str(path), "--model", str(tiny), "--device", "cpu", "--mode", "free",
            "--kv-capacity", "1000", "--block", "4", "--max-batch-size", "2", "--prefill-chunk-size", "4",
            "--out", str(tmp_path/"free"), "--sequence-order", "reverse"]
    capture.main(argv)
    info = json.loads((tmp_path/"free/run.json").read_text())
    assert info["sequences"] == ["b", "a"] and info["config"]["sequence_order"] == "reverse"
    assert int(load_record(tmp_path/"free", "a")["first"]) == 1
    assert int(load_record(tmp_path/"free", "b")["first"]) == 2
    with pytest.raises(SystemExit, match="different configuration"):
        capture.main(argv[:-2])


def test_paired_look_error_ratio_and_bias():
    from script.tah.validation.paired_look import bounds
    rows = [dict(n=100, m2_n=100, m1_sum=10+i, m2_sum=2+i/10,
                 m3_sum=30+i, bias_sum=.01) for i in range(32)]
    result = bounds(rows, rows)
    assert all(v["status"] == "PASS" for v in result.values())
    worse = [dict(r, m1_sum=1.2*r["m1_sum"]) for r in rows]
    assert bounds(worse, rows)["M1"]["status"] == "not shown"


def test_short_confirmatory_qualification_and_hard_futility(tmp_path, monkeypatch):
    ps = [dict(id=f"p{i}", dataset="math500" if i < 498 else "gpqa" if i < 692 else "gsm8k",
               index=i, seed=42+i, prompt=[3], answer="1") for i in range(1204)]
    manifest = dict(version=1, protocol=short_c1.PROTOCOL, max_new_tokens=512, prompt_limit=675,
                    selection_rule="all eligible Math500/GPQA; SHA256 order first 512 eligible GSM8K",
                    problems=ps, oracle_ids=[p["id"] for p in ps],
                    subsets=dict(stage1=[p["id"] for p in ps[:692]], stage2=[p["id"] for p in ps[692:]]))
    path = tmp_path/"manifest.json"
    write_json(path, manifest)
    stage = tmp_path/"stage1"
    stage.mkdir()
    (stage/"oracle.path").write_text(str(stage/"oracle"))
    oracle_run = tmp_path/"valid-oracle"
    write_json(oracle_run/"run.json", dict(model_sha256="model", config=dict(eval_config_sha256="eval",
        oracle_constants="bf16-rounded buffers including inv_freq (R1-O amendment 2026-10-13)")))
    evidence = tmp_path/"validity.json"
    write_json(evidence, dict(validity=dict(status="PASS", size_requirement_met=True),
                              oracle=dict(run=str(oracle_run), threshold=.8984375)))
    scenario = {"bad_engine": False}

    def load(directory, m, implementation):
        cfg = dict(pilot=0, fixture=False, max_new_tokens=512, subset="stage1", dtype="bf16",
                   problems=m["subsets"]["stage1"], batch=8, block_size=128, model_sha256="model",
                   source_sha256="source", eval_config_sha256="eval", environment={}, oracle_constants="constants",
                   sampling={}, engine_options=dict(cuda_graphs=True, overlap=True, compile=False, max_batch_size=64))
        if implementation == "oracle":
            cfg.update(dtype="fp32", batch=4)
        tokens = [] if scenario["bad_engine"] and implementation == "engine" else [1]*512
        return cfg, {p: dict(token_ids=tokens, sampled_token_ids=tokens, dataset="math500") for p in cfg["problems"]}

    monkeypatch.setattr(short_c1, "load_run", load)
    good = short_c1.analyze(path, tmp_path, 1, evidence)
    assert good["conclusion"] == "PASS" and good["censoring"]["engine_cap_censored"] == 1
    with pytest.raises(ValueError, match="interim rule"):
        short_c1.analyze(path, tmp_path, 2, evidence)
    scenario["bad_engine"] = True
    bad = short_c1.analyze(path, tmp_path, 1, evidence)
    assert bad["next_action"] == "stop_for_futility" and bad["best_possible_final_margin_sum"] < 0


def test_position_correlation_uses_joint_eligible_universe():
    a, b = tail_rows(2), tail_rows(2)
    for x, y in zip(a, b):
        for kind in near_tie.KINDS:
            x[kind].update(events=np.arange(50), eligible_events=np.arange(100))
            y[kind].update(events=np.arange(0,100,2), eligible_events=np.arange(100))
    report = near_tie.correlations(a, b)
    assert all(x["disagreement_position_phi"] == 0 and x["jointly_eligible_positions"] == 200
               for x in report.values())


def test_tail_alpha_025_uses_975_bound_and_t_floor():
    from script.tah.evaluation.compare_runs import t_critical
    rows = tail_rows(64)
    loose = near_tie.compare(rows, rows, resamples=500, alpha=.05)
    strict = near_tie.compare(rows, rows, resamples=500, alpha=.025)
    assert strict['alpha'] == .025 and strict['confidence'] == .975
    assert strict['original_max_binding'] is False
    for kind in near_tie.KINDS:
        result = strict['criteria'][kind]
        assert result['critical'] >= t_critical(.975, 63)
        assert result['upper975'] == result['upper_bound']
        assert result['upper_bound'] >= loose['criteria'][kind]['upper95']
        assert 'upper95' not in result
    with pytest.raises(ValueError, match='alpha'):
        near_tie.compare(rows, rows, alpha=0)


@pytest.fixture
def o1_looks(tmp_path, monkeypatch):
    import hashlib
    from script.tah.validation import paired_look
    ids = [f'p{i}' for i in range(200)]
    manifest = tmp_path/'manifest.json'
    write_json(manifest, dict(sequences=[dict(id=s, dataset='math500') for s in ids],
                             subsets=dict(stage1=ids[:64], stage2=ids[64:])))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    model = tmp_path/'model'
    write_json(model/'config.json', dict(vocab_size=100))
    for stage in (1, 2):
        directory = tmp_path/'o1'/f'stage{stage}'
        directory.mkdir(parents=True)
        (directory/'oracle-fp32.path').write_text(str(directory/'oracle'))
    evidence = tmp_path/'o1/stage1/analysis/r1o1.json'
    write_json(evidence, dict(validity=dict(status='PASS', size_requirement_met=True),
                              no_large_errors=dict(decision=dict(status='not shown'))))
    state = dict(forced_pass=True, tail_pass=True, calls=[], compared=[])

    def info(path):
        path = type(tmp_path)(path)
        stage = int(path.parent.name[-1])
        state['calls'].append((stage, path.name))
        return dict(complete=True, sequences=ids[:64] if stage == 1 else ids[64:],
                    manifest_sha256=digest, model_sha256='model', threshold=.9,
                    config=dict(model=str(model), dtype='bf16', batch=8, layout='generate',
                                max_batch_size=32, mode='free' if path.name.endswith('free') else 'forced'))

    def collect(run, oracle, expected, *args, **kwargs):
        return [dict(n=3000) for _ in expected]

    def extract(run, oracle):
        stage = int(run.parent.name[-1])
        return [dict(id=s, top1=dict(eligible_events=np.arange(3000)))
                for s in (ids[:64] if stage == 1 else ids[64:])]

    def bound(*args):
        return {k: dict(status='PASS' if state['forced_pass'] else 'not shown')
                for k in ('M1', 'M2', 'M3', 'bias')}

    def tail(e, r, *, alpha):
        assert alpha == .025 and [v['id'] for v in e] == [v['id'] for v in r]
        state['compared'].append(len(e))
        return dict(criteria={k: dict(status='PASS' if state['tail_pass'] else 'not shown',
            original_max=dict(engine=2., reference=1., status='not shown')) for k in near_tie.KINDS})

    monkeypatch.setattr(paired_look.a, 'run_info', info)
    monkeypatch.setattr(paired_look.a, 'collect', collect)
    monkeypatch.setattr(paired_look, 'bounds', bound)
    monkeypatch.setattr(paired_look.near_tie, 'extract', extract)
    monkeypatch.setattr(paired_look.near_tie, 'compare', tail)
    return paired_look, tmp_path, manifest, evidence, state


def test_original_max_failure_does_not_gate_o1_or_trigger_futility(o1_looks):
    tool, root, manifest, _, state = o1_looks
    result = tool.analyze(root, manifest, 1)
    assert result['conclusion'] == 'PASS'
    assert result['original_maximum']['binding'] is False
    assert all(v['status'] == 'not shown' for v in result['original_maximum']['criteria'].values())
    assert result['forced_next_action'] == result['free_next_action'] == 'stop_for_success'
    with pytest.raises(ValueError, match='interim rule'):
        tool.analyze(root, manifest, 2)
    state.update(forced_pass=False, tail_pass=False)
    pending = tool.analyze(root, manifest, 1)
    assert pending['forced_next_action'] == pending['free_next_action'] == 'continue_fixed_136'


@pytest.mark.parametrize('extend_arm', ['forced', 'free', 'both'])
def test_o1_independent_increments_and_permanent_success(o1_looks, extend_arm):
    tool, root, manifest, _, state = o1_looks
    state.update(forced_pass=extend_arm == 'free', tail_pass=extend_arm == 'forced')
    tool.analyze(root, manifest, 1)
    before = (root/'o1/look1.json').read_bytes()
    # Success at look 1 stays final even if a recomputed descriptive bound does not pass.
    state.update(forced_pass=extend_arm != 'free', tail_pass=extend_arm != 'forced', calls=[])
    result = tool.analyze(root, manifest, 2)
    assert result['conclusion'] == 'PASS'
    stage2 = {name for stage, name in state['calls'] if stage == 2}
    assert ('ref-forced' in stage2) == (extend_arm != 'free')
    assert ('engine-forced' in stage2) == (extend_arm != 'free')
    assert ('ref-free' in stage2) == (extend_arm != 'forced')
    assert ('engine-free' in stage2) == (extend_arm != 'forced')
    assert result['forced_sequences'] == (64 if extend_arm == 'free' else 200)
    assert result['free_sequences'] == (64 if extend_arm == 'forced' else 200)
    assert (root/'o1/look1.json').read_bytes() == before
    stopped = result['criteria'] if extend_arm == 'free' else result['tail']['criteria']
    if extend_arm != 'both':
        assert all(v['stopped_for_success_at_look'] == 1 for v in stopped.values())


def test_o1_preserves_success_of_each_tail_kind(o1_looks):
    tool, root, manifest, _, state = o1_looks
    state['tail_pass'] = False
    initial = tool.analyze(root, manifest, 1)
    initial['tail']['criteria']['decision']['status'] = 'PASS'
    write_json(root/'o1/look1.json', initial)
    final = tool.analyze(root, manifest, 2)
    assert final['tail']['criteria']['decision']['status'] == 'PASS'
    assert final['tail']['criteria']['top1']['status'] == 'not shown'
    assert final['free_next_action'] == 'stop_at_maximum_not_shown'
    assert final['conclusion'] == 'not shown'


def test_o1_validity_stops_both_arms(o1_looks):
    tool, root, manifest, evidence, state = o1_looks
    state.update(forced_pass=False, tail_pass=False)
    write_json(evidence, dict(validity=dict(status='FAIL', size_requirement_met=True)))
    result = tool.analyze(root, manifest, 1)
    assert result['forced_next_action'] == result['free_next_action'] == 'stop_for_futility'
    with pytest.raises(ValueError, match='interim rule'):
        tool.analyze(root, manifest, 2)


@pytest.mark.parametrize('forced,free,valid', [(False, False, True), (True, False, True),
                                             (False, True, True), (True, True, True), (True, True, False)])
def test_night_chain_stage2_branches_without_gpu(tmp_path, forced, free, valid):
    import subprocess
    import sys
    from pathlib import Path
    script_path = Path('notes/tools/WP23/confirmatory.sh')
    if not script_path.is_file():
        pytest.skip('Local night-chain script is not distributed with the validation package')
    script = script_path.read_text()
    body = script.split('elif [[ $phase == stage2 ]]; then\n', 1)[1].rsplit("else\n  echo 'Use stage1 or stage2'", 1)[0]
    # Only the final analysis invocation is stubbed; execute the actual branching and validity guard.
    body = body.replace('CUDA_VISIBLE_DEVICES=\'\' "$PY" -m script.tah.validation.paired_look --root "$O" --manifest "$F" --look 2',
                        'record look2')
    write_json(tmp_path/'o1/look1.json', dict(validity_met=valid,
        forced_next_action='continue_fixed_136' if forced else 'stop_for_success',
        free_next_action='continue_fixed_136' if free else 'stop_for_success'))
    header = '''set -euo pipefail
PY=$1; O=$2; F=manifest; cap=(capture); common=()
record() { printf '%s\\n' "$1" >> "$O/actions"; }
run() { record "$1"; }
oracle_tf() { record oracle; ORACLE_OUT=oracle; }
c1_stage() { record c1; }
continue_stage() {
  [[ $1 != */c1/* ]] || return 0
  "$PY" - "$1" "$2" "${3:-next_action}" <<'CHECK'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))[sys.argv[3]]==sys.argv[2] else 1)
CHECK
}
'''
    result = subprocess.run(['bash', '-c', header+body, 'test', sys.executable, str(tmp_path)],
                            capture_output=True, text=True)
    calls = (tmp_path/'actions').read_text().splitlines() if (tmp_path/'actions').exists() else []
    assert (result.returncode == 0) == valid, result.stderr
    assert ('oracle' in calls) == (valid and (forced or free))
    assert ('ref-forced-stage2' in calls) == (valid and forced)
    assert ('eng-forced-stage2' in calls) == (valid and forced)
    assert ('ref-free-stage2' in calls) == (valid and free)
    assert ('eng-free-stage2' in calls) == (valid and free)
    assert ('look2' in calls) == (valid and (forced or free))
    assert ('c1' in calls) == valid


def test_tail_reports_extreme_max_failure_without_changing_tail_bound():
    reference = tail_rows(64)
    engine = copy.deepcopy(reference)
    for kind in near_tie.KINDS:
        engine[0][kind]['margins'][-1] = 100.
    report = near_tie.compare(engine, reference, alpha=.025, resamples=500)
    assert report['conclusion'] == 'PASS' and report['original_max_binding'] is False
    for kind in near_tie.KINDS:
        result = report['criteria'][kind]['original_max']
        assert result['status'] == 'not shown' and result['engine'] == 100.
        assert result['engine_events'] == result['reference_events'] == 64*30
        assert result['limit'] == 1.1*result['reference']
    for row in reference:
        for kind in near_tie.KINDS:
            row[kind]['margins'] = np.array([])
    no_reference = near_tie.compare(engine, reference, resamples=50)
    assert no_reference['conclusion'] == 'not shown'
    assert all(v['original_max']['reference'] is None and v['original_max']['status'] == 'not shown'
               for v in no_reference['criteria'].values())
    neither = near_tie.compare(reference, reference, resamples=50)
    assert neither['conclusion'] == 'not shown'
    assert all(v['original_max']['status'] == 'PASS' for v in neither['criteria'].values())
