"""Prefill tail regression: independent CPU oracle and optional CUDA memcheck."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from tah.engine.attention import AttentionRow, PrefillAttention
from tah.engine.kv_cache import SlotPool


CUDA_HIDDEN = os.environ.get('CUDA_VISIBLE_DEVICES') == ''
DEVICES = ['cpu', pytest.param('cuda', marks=[pytest.mark.gpu,
    pytest.mark.skipif(CUDA_HIDDEN or not torch.cuda.is_available(), reason='CUDA required')])]
LENGTHS = [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 80, 81, 127, 128, 129]
HEADS = [(2, 2), (4, 2), (8, 2), (16, 8)]


def layout(device, q_len, heads, kv_heads, *, mixed=False):
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(8037 + q_len)
    limit = 128 + q_len
    pool = SlotPool(limit * 8 + 32, 1, kv_heads, 128, 3, device=device)
    requests = [pool.create_request(i, context_limit=limit) for i in range(2)]
    # Interleave allocations for different requests so physical page indices
    # cannot be mistaken for logical token positions or mask column indices.
    for depth in range(3):
        for request in requests:
            positions = (list(range(limit)) if depth == 0 else
                         list(range(0, 128, 3 + depth)) +
                         list(range(128, limit, depth)))
            pool.allocate(request, depth, positions)
    if mixed:
        # Repeated request/depth groups, non-byte-aligned mask segments, and
        # non-monotonic query coordinates within a group.
        rows = [AttentionRow(requests[0], p, 2) for p in range(128, limit)]
        rows += [AttentionRow(requests[1], p, 1) for p in [7, 1, 5, 2, 9, 0, 3]]
        rows += [AttentionRow(requests[0], p, 3) for p in range(limit - 1, 127, -1)]
        rows += [AttentionRow(requests[1], 11, 2)]
    else:
        rows = [AttentionRow(requests[0], p, 2) for p in range(128, limit)]
    values = torch.randn(pool.values.shape, generator=gen).bfloat16()
    q = torch.randn(len(rows), heads, 128, generator=gen).bfloat16()
    expected = torch.empty(q.shape, dtype=torch.float32)
    # Independent oracle groups by request only, not by the implementation's
    # consecutive request/depth groups. It includes every allocated stream and
    # reconstructs visibility from host coordinates.
    for request in requests:
        selected = [i for i, row in enumerate(rows) if row.request is request]
        if not selected:
            continue
        entries = [(slot, p, depth) for depth, table in enumerate(request.streams, 1)
                   for slot, p in zip(table.host_slots, table.host_positions)]
        ids = torch.tensor([slot for slot, _, _ in entries])
        mask = torch.tensor([[p <= rows[i].position and depth <= rows[i].depth
                              for _, p, depth in entries] for i in selected])
        k, v = [values[0, j, ids, 0].float().transpose(0, 1)
                .repeat_interleave(heads // kv_heads, 0)[None] for j in (0, 1)]
        expected[selected] = F.scaled_dot_product_attention(
            q[selected].float().transpose(0, 1)[None], k, v,
            attn_mask=mask[None, None])[0].transpose(0, 1)
    pool.values.copy_(values.to(device))
    return pool, rows, q.to(device), expected


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('heads,kv_heads', HEADS)
@pytest.mark.parametrize('q_len', LENGTHS)
def test_prefill_tail_boundaries(device, heads, kv_heads, q_len):
    pool, rows, q, expected = layout(device, q_len, heads, kv_heads)
    attention = PrefillAttention(pool, rows, heads)
    actual = attention.run(q, 0)
    if device == 'cuda':
        torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu().float(), expected, atol=.015, rtol=.02)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('heads,kv_heads', HEADS)
def test_prefill_tail_mixed_groups(device, heads, kv_heads):
    pool, rows, q, expected = layout(device, 81, heads, kv_heads, mixed=True)
    attention = PrefillAttention(pool, rows, heads)
    actual = attention.run(q, 0)
    if device == 'cuda':
        torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu().float(), expected, atol=.015, rtol=.02)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(CUDA_HIDDEN or not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('case', ['aligned', 'tail81', 'tail80', 'collector80'])
def test_prefill_tail_memcheck(case, tmp_path):
    sanitizer = shutil.which('compute-sanitizer')
    if sanitizer is None:
        pytest.skip('CUDA Compute Sanitizer required')
    log = tmp_path / f'{case}.memcheck.log'
    env = dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING='1')
    result = subprocess.run([
        sanitizer, '--target-processes', 'application-only', '--tool', 'memcheck',
        '--error-exitcode', '99', '--print-limit', '4', '--log-file', str(log),
        sys.executable, '-m', 'tah.engine.tests.prefill_tail_probe',
        '--case', case, '--path', 'engine'],
        cwd=Path(__file__).resolve().parents[3], env=env,
        capture_output=True, text=True, timeout=180)
    report = log.read_text() if log.exists() else 'No memcheck report'
    assert result.returncode == 0, result.stdout + result.stderr + report
    assert 'ERROR SUMMARY: 0 errors' in report, report
    assert 'NUMERICAL_PASS' in result.stdout, result.stdout


@pytest.mark.gpu
@pytest.mark.skipif(CUDA_HIDDEN or not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.skipif(os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING') == '1',
                    reason='CUDA graphs require the caching allocator; run separately with caching enabled')
def test_chunked_prefill_generation_graph_matches_eager(tmp_path):
    import json
    from tah.engine import TaHEngine, Request, SamplingParams
    from tah.engine.tests.test_model import reference

    reference(tmp_path, max_iter=2, label=False, head_dim=128)
    config_file = tmp_path / 'config.json'
    config = json.loads(config_file.read_text())
    config['max_position_embeddings'] = 512
    config_file.write_text(json.dumps(config))
    prompt = [3 + i % 89 for i in range(209)]
    with TaHEngine(tmp_path, max_batch_size=1, kv_capacity=1024,
                   prefill_chunk_size=128, graph_batch_sizes=[1],
                   host_swap_bytes=0, overlap=False, eos_token_id=None) as engine:
        requests = [Request(prompt, 8, 37)]
        sampling = SamplingParams(temperature=0)
        engine.cuda_graphs = False
        eager = engine.generate(requests, sampling)[0]
        engine.cuda_graphs = True
        graphed = engine.generate(requests, sampling)[0]
        assert graphed.token_ids == eager.token_ids
        assert graphed.prompt_pass_counts == eager.prompt_pass_counts == [2] * 209
        assert graphed.generated_pass_counts == eager.generated_pass_counts == [2] * 8
        assert engine.last_run['graph_steps'] > 0
        assert engine.pool.audit()['owned'] == 0


@pytest.mark.gpu
@pytest.mark.skipif(CUDA_HIDDEN or not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('threshold,depth', [(-1., 2), (2., 1)])
def test_collector_209_token_prompt(tmp_path, threshold, depth):
    import json
    from research.bandit_router.engine import EnginePairs
    from tah.engine.model import EagerTaHModel
    from tah.engine.tests.test_model import reference

    reference(tmp_path, max_iter=2, label=False, head_dim=128)
    config_file = tmp_path / 'config.json'
    config = json.loads(config_file.read_text())
    config['max_position_embeddings'] = 512
    config_file.write_text(json.dumps(config))
    model = EagerTaHModel.from_pretrained(tmp_path, device='cuda').eval()
    (model.eval_iter_decider or model.iter_decider).threshold.data.fill_(threshold)
    pairs = EnginePairs(model, 1024)
    prompt = [3 + i % 89 for i in range(209)]
    try:
        with torch.inference_mode():
            history = pairs.start(prompt, 512)
            assert history == [depth] * 208
            features, losses, selected = pairs.step(prompt[-1], 9, 208)
            torch.cuda.synchronize()
            assert selected == depth
            assert torch.isfinite(features).all()
            assert torch.isfinite(losses).all() and losses.shape == (2,)
    finally:
        pairs.release()
    assert pairs.pool.audit()['owned'] == 0
