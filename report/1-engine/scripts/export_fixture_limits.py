"""Read-only integrity/noise export for the completed reference fixture suite."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from tah.engine.fixtures import FixtureSuite

torch.set_num_threads(4)

import os
root = Path(os.environ.get('TAH_GTF_FIXTURES', 'data/engine/fixtures/reference'))
suite = FixtureSuite(root)
m = suite.manifest
assert m['production_gates_eligible'] is True
assert m['profile'] == 'full'
assert set(m['checkpoints']) == {'real', 'mlp3', 'mlp1', 'label3'}
assert m['self_noise']['mlp3']['full_fixture_depth_coverage_pass'] is True
assert len(m['records']) == 276
for key in ('reference_source_sha256', 'tool_source_sha256'):
    for path, expected in m[key].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert hashlib.sha256(Path(m['eval_config']['path']).read_bytes()).hexdigest() == m['eval_config']['sha256']
checkpoint_files = 0
for name, checkpoint in m['checkpoints'].items():
    for relative, expected in checkpoint['files'].items():
        path = Path(checkpoint['path']) / relative
        assert path.stat().st_size == expected['bytes'], path
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected['sha256'], path
        checkpoint_files += 1
    assert json.loads((root / name / 'self_noise.json').read_text()) == m['self_noise'][name]

counts = Counter()
baseline_counts = {}
generated_counts = Counter()
for record in suite.records():
    path = root / record['file']
    assert path.stat().st_size == record['bytes'], path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
    t = suite.load(record)
    name = record['checkpoint']
    counts[f"{name}/{record['kind']}"] += 1
    if record['kind'] == 'teacher_forced':
        assert t['visited'].dtype == torch.bool
        assert ((t['pass_count'] >= 1) & (t['pass_count'] <= len(t['visited']))).all()
        assert not t['decision'][-1].any()
        assert (t['continue_probability'][-1][t['visited'][-1]] == 0).all()
        assert (t['decision'][:-1] == t['visited'][1:]).all()
        assert torch.equal(t['final_logits_sample'].argmax(-1).to(torch.int32), t['final_top1'][t['logit_positions']])
        expected_values = t['final_logits_sample'].gather(-1, t['final_topk_indices'][t['logit_positions']].long())
        assert torch.equal(expected_values, t['final_topk_values'][t['logit_positions']])
        if name == 'label3':
            assert torch.equal(t['pass_count'], t['iter_count_labels'].clamp(min=1, max=3))
        if record['variant'] == 'single':
            totals = baseline_counts.setdefault(name, [[0, 0] for _ in range(len(t['visited']))])
            for depth, total in enumerate(totals):
                total[0] += int(t['visited'][depth].sum())
                total[1] += int(t['decision'][depth].sum())
    else:
        assert len(t['generated_ids']) <= record['max_new_tokens']
        assert not (t['generated_ids'] == record['eos_token_id']).any()
        generated_counts[name] += len(t['generated_ids'])

limits = {}
for name, noise in m['self_noise'].items():
    assert counts[f'{name}/teacher_forced'] == 45
    assert counts[f'{name}/greedy'] == 24
    assert baseline_counts[name] == [[row['visited'], row['continued']] for row in noise['depth_totals']]
    tf = noise['teacher_forced']['summary']
    assert len(noise['teacher_forced']['comparisons']) == 36
    assert len(noise['greedy']['comparisons']) == 18
    gtf = {'per_depth_min_decision_agreement': .995, 'max_mismatch_threshold_margin': .01}
    for key in ('logit_mae', 'logit_rmse', 'logit_max_abs'):
        assert tf[f'g_tf_{key}_limit'] == 1.5 * tf[key]
        gtf[f'{key}_limit'] = tf[f'g_tf_{key}_limit']
    gtf['top1_disagreement_limit'] = tf['g_tf_top1_disagreement_limit']
    for coordinates in ('retained_topk_error', 'retained_dense_error'):
        gtf[coordinates] = {key: value for key, value in tf[coordinates].items() if key.startswith('g_tf_')}
    gtf['reference_fixed_decision_diagnostic'] = {
        'status': 'PASS' if all(d['decision_agreement'] >= .995 and d['decision_mismatches'] == d['mismatches_within_0.01'] for d in tf['per_depth']) else 'FAIL',
        'per_depth': tf['per_depth'],
        'note': 'Reference layout diagnostic only; engine G-TF has not been run. Visitation cascades are counted conservatively as mismatches.'}
    positions = noise['greedy']['divergence_positions']
    finite = sorted(p for p in positions if p is not None)
    points = [{'position': p, 'max_divergence_fraction': sum(q <= p for q in finite) / len(positions)}
              for p in sorted(set(finite))]
    limits[name] = {
        'G-TF': gtf,
        'G-GEN': {'rule': 'candidate first-divergence CDF <= reference CDF at every token position; identical pairs are +infinity',
                  'comparison_pairs': len(positions), 'reference_divergence_positions': positions,
                  'reference_divergence_cdf_breakpoints': points,
                  'max_divergent_fraction': len(finite) / len(positions)},
        'reference_self_noise': tf,
    }
actual_bytes = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
assert actual_bytes == m['total_fixture_bytes']
assert sum(record['bytes'] for record in m['records']) == m['fixture_tensor_bytes']
report = {'status': 'PASS', 'records_verified': len(m['records']), 'checkpoint_files_verified': checkpoint_files,
          'counts': dict(counts), 'baseline_depth_totals': baseline_counts,
          'generated_tokens_all_layouts': dict(generated_counts),
          'fixture_tensor_bytes': m['fixture_tensor_bytes'], 'total_fixture_bytes': actual_bytes,
          'manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()}
out = Path(os.environ['ENGINE_REPORT_OUTPUT']) / 'fixture-limits'
out.mkdir(parents=True, exist_ok=True)
(out / 'integrity.json').write_text(json.dumps(report, indent=2) + '\n')
(out / 'tolerances.json').write_text(json.dumps({'production_gates_eligible': True, 'fixture_root': str(root),
    'manifest_sha256': report['manifest_sha256'], 'checkpoints': limits}, indent=2) + '\n')
assert not torch.cuda.is_initialized()
print(json.dumps(report, indent=2))
