"""Overlap identity: run the pipelined scheduler, then replay its row sets without it.

The lag makes a finished request run one discarded ghost row and delays
admission by one step. Non-overlap mode with the test-only ghost-lag flag
executes exactly those per-step row sets, so tokens, decisions and pass counts
must be identical (GEMM shapes and every live row are the same).
"""


def overlap_vs_replay(eng, requests, params, hook=None):
    """Run overlap, then non-overlap with ghost-lag replay; return both traces."""
    traces = {}
    for mode in ('overlap', 'replay'):
        rows, decisions = [], []
        eng.overlap, eng._ghost_lag = mode == 'overlap', mode == 'replay'
        if mode == 'overlap':
            def resolved(record, resolved_rows):
                rows.append([s.index for s, _ in record['entries']])
                attempts = {s.index: a for s, a in record['entries']}
                decisions.extend((index, attempts[index], pos, depth, keep)
                                 for _, index, pos, depth, keep in resolved_rows)
            eng._resolved_hook, decode = resolved, None
        else:
            decode = eng._decode
            def observed(running):
                status = decode(running)
                rows.append([s.index for s in running])
                decisions.extend((s.index, s.restarts, s.position, s.depth, bool(r[0]))
                                 for s, r in zip(running, status) if s.index not in eng._lagged)
                return status
            eng._decode = observed
        eng._step_hook = hook
        try:
            results = eng.generate(requests, params)
        finally:
            eng._resolved_hook, eng._step_hook = None, None
            if decode is not None:
                eng._decode = decode
        final = {r.request_id: r.timing['restarts'] for r in results}
        traces[mode] = dict(results=results, rows=rows, run=eng.last_run,
                            decisions=[d for d in decisions if d[1] == final[d[0]]])
        assert eng.pool.audit()['owned'] == 0 and eng.pool.free_slots == eng.pool.capacity
        assert eng.swap.used_bytes == 0
    eng.overlap, eng._ghost_lag = True, False
    return traces['overlap'], traces['replay']


def assert_same_execution(a, b):
    assert a['rows'] == b['rows'], 'executed per-step row sets differ'
    assert a['decisions'] == b['decisions']
    assert [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts, r.finish_reason)
            for r in a['results']] == [
           (r.token_ids, r.prompt_pass_counts, r.generated_pass_counts, r.finish_reason)
            for r in b['results']]
