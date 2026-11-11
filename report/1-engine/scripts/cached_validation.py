"""Production cached/reference validation plus preserved cross-layout diagnostic."""
import json
import os
from pathlib import Path

import pytest
import torch

from tah.engine.fixtures import FixtureSuite
from tah.engine.kv_cache import SlotPool
from tah.engine.model import EagerTaHModel, TeacherForcedOutput
from tah.engine.parity import (assert_gtf, compare_output, derive_decision_limits,
                               gtf_summary, production_noise, require_production)
from tah.engine.tests import test_gtf as gtf_tests

_gtf_test_candidates = [value for name, value in vars(gtf_tests).items()
                        if name.startswith('test_gtf_every_') and name.endswith('_fixture_set')]
if len(_gtf_test_candidates) != 1:
    raise RuntimeError('Expected one full-fixture G-TF test entry point')
run_oracle_gtf = _gtf_test_candidates[0]

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES') == '', reason='CUDA hidden; production cached G-TF deferred')]


@pytest.mark.parametrize('name', ['real', 'mlp3', 'mlp1', 'label3'])
@pytest.mark.parametrize('chunk_size', [1, 256])
def test_cached_production_gtf(name, chunk_size, monkeypatch):
    suite = FixtureSuite(os.environ.get('TAH_GTF_FIXTURES', 'data/engine/fixtures/reference'))
    require_production(suite)
    tolerances_path = Path(os.environ['TAH_GTF_TOLERANCES'])
    noise = production_noise(suite, name, json.loads(tolerances_path.read_text()))
    decision_limits = derive_decision_limits(noise)
    # Identical to the predeclared self-noise limits; never infer these from
    # candidate cached outputs or reuse the obsolete fixed-decision gate.
    limits_path = Path(os.environ['ENGINE_REPORT_OUTPUT']) / 'full' / f'{name}-limits.json'
    if limits_path.exists():
        assert decision_limits == json.loads(limits_path.read_text())['decision_limits']
    original_forward = EagerTaHModel.forward
    paired = []
    templates = [suite.load(r) for r in suite.records(kind='teacher_forced', checkpoint=name) if r['variant'] == 'single']

    @torch.inference_mode()
    def cached_forward(model, input_ids, *, iter_count_labels=None, position_ids=None, hidden_positions=None):
        assert position_ids is None
        ids = torch.as_tensor(input_ids, device=model.device)
        labels = None if iter_count_labels is None else torch.as_tensor(iter_count_labels, device=model.device)
        selected = torch.as_tensor(hidden_positions, device=model.device)
        pool = SlotPool(len(ids) * model.max_iter, model.config['num_hidden_layers'],
            model.config['num_key_value_heads'], model.config['head_dim'], model.max_iter,
            device=model.device, dtype=model.dtype)
        request = pool.create_request('gtf', context_limit=len(ids))
        chunks = []
        for start in range(0, len(ids), chunk_size):
            end = min(start + chunk_size, len(ids))
            local_selected = selected[(selected >= start) & (selected < end)] - start
            chunk = model.forward_chunk(ids[start:end], pool, request, start_position=start,
                iter_count_labels=None if labels is None else labels[start:end],
                hidden_positions=local_selected, decode=chunk_size == 1)
            # Keep GPU vocab storage bounded by chunk even in the parity tool.
            for key, value in vars(chunk).items():
                setattr(chunk, key, value.cpu())
            chunks.append(chunk)
            if chunk_size == 1 and (end % 256 == 0 or end == len(ids)):
                print(f"Cached G-TF {name}: {end}/{len(ids)} tokens (decode)", flush=True)
        output = TeacherForcedOutput(
            torch.cat([c.logits for c in chunks]), torch.cat([c.iter_count for c in chunks]),
            *[torch.cat([getattr(c, key) for c in chunks], 1) for key in
              ['visited', 'decision', 'decision_logits', 'continue_probability']],
            selected.cpu(), chunks[0].hidden_layer_indices,
            torch.cat([c.decider_hidden for c in chunks], 1))
        pool.release(request)
        assert pool.audit()['free'] == pool.capacity
        del pool
        full = original_forward(model, ids, iter_count_labels=labels, hidden_positions=selected)
        template = next(t for t in templates if torch.equal(t['input_ids'], ids.cpu()))
        # Full-sequence traces and retained coordinates become the paired oracle,
        # with the same production self-noise limits (no tolerance floor).
        paired_template = dict(template)
        for key in ['visited', 'decision', 'decision_logits', 'continue_probability', 'decider_hidden']:
            paired_template[key] = getattr(full, key).cpu()
        paired_template['pass_count'] = full.iter_count.cpu()
        paired_template['final_topk_values'] = full.logits.cpu().gather(-1, template['final_topk_indices'].long())
        paired_template['final_logits_sample'] = full.logits.cpu()[template['logit_positions']]
        paired.append(compare_output(paired_template, output, full.logits.cpu(), suite.manifest['checkpoints'][name]['threshold']))
        return output

    monkeypatch.setattr(EagerTaHModel, 'forward', cached_forward)
    destination = Path(os.environ['ENGINE_REPORT_OUTPUT']) / f'cached-{chunk_size}'
    monkeypatch.setenv('TAH_GTF_RESULTS', str(destination))
    # Existing gate verifies hashes, every layout, full-vocabulary reference
    # replay, margins, and unchanged production noise limits.
    run_oracle_gtf(name)
    stats = gtf_summary(paired)
    report = dict(
        production_gates_eligible=True, gate='G-TF cached vs full reference',
        chunk_size=chunk_size, summary=stats, comparisons=paired,
        decision_limits=decision_limits, noise_limits=noise)
    try:
        assert_gtf(paired, noise)
    except AssertionError:
        report['status'] = 'FAIL'
        raise
    else:
        report['status'] = 'PASS'
    finally:
        (destination / f'{name}-vs-full.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', choices=['real','mlp3','mlp1','label3'], required=True)
    parser.add_argument('--chunk', type=int, choices=[1,256], required=True)
    args=parser.parse_args()
    with pytest.MonkeyPatch.context() as patch:
        test_cached_production_gtf(args.checkpoint,args.chunk,patch)
