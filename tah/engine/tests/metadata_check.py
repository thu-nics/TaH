"""CPU-only Triton interpretation/offline compilation; never loads a GPU module."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['interpret', 'compile'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU tooling requires CUDA_VISIBLE_DEVICES=""')
    if args.mode == 'interpret':
        os.environ['TRITON_INTERPRET'] = '1'
    import torch
    import triton
    from tah.engine.attention_kernels import count_visible, pack_visible, stage_sources
    report = dict(production_gates_eligible=False, mode=args.mode, cases=[])
    if args.mode == 'compile':
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        for arch in [80, 86, 89]:
            for fn, names in [(count_visible, ['pos', 'depths', 'starts', 'lengths', 'query_pos', 'query_depth', 'counts']),
                (pack_visible, ['slots', 'pos', 'depths', 'starts', 'lengths', 'query_pos', 'query_depth', 'indptr', 'indices']),
                (stage_sources, ['descriptors', 'slots', 'positions', 'depths', 'starts', 'lengths', 'query_pos', 'query_depth'])]:
                signature = {name: '*i64' if name in ['descriptors', 'pos', 'positions', 'query_pos'] else '*i32' for name in names}
                constants = {'BLOCK': 256}
                if fn is stage_sources:
                    constants['STREAMS'] = 3
                compiled = triton.compile(ASTSource(fn, signature, constants=constants), target=GPUTarget('cuda', arch, 32))
                report['cases'].append(dict(arch=arch, kernel=fn.__name__, ptx_bytes=len(compiled.asm['ptx']),
                                            cubin_bytes=len(compiled.asm['cubin'])))
    else:
        for seed in range(20):
            gen = torch.Generator().manual_seed(seed)
            lengths = torch.tensor([1, 7, 255, 256, 257, 513, 2049], dtype=torch.int32)
            starts = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()[:-1]])
            total = int(lengths.sum())
            pos = torch.randint(2000, (total,), generator=gen)
            depths = torch.randint(1, 4, (total,), generator=gen, dtype=torch.int32)
            slots = torch.randperm(total, generator=gen).int()
            qp = torch.randint(2000, (len(lengths),), generator=gen)
            qd = torch.randint(1, 4, (len(lengths),), generator=gen, dtype=torch.int32)
            counts = torch.empty_like(lengths)
            count_visible[(len(lengths),)](pos, depths, starts, lengths, qp, qd, counts, BLOCK=256)
            ptr = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()])
            ids = torch.full((total,), -1, dtype=torch.int32)
            pack_visible[(len(lengths),)](slots, pos, depths, starts, lengths, qp, qd, ptr, ids, BLOCK=256)
            for row in range(len(lengths)):
                sl = slice(int(starts[row]), int(starts[row] + lengths[row]))
                expected = slots[sl][(pos[sl] <= qp[row]) & (depths[sl] <= qd[row])]
                assert int(counts[row]) == len(expected)
                assert torch.equal(ids[int(ptr[row]):int(ptr[row + 1])], expected)
            assert bool((ids[int(ptr[-1]):] == -1).all()), 'compaction wrote outside packed result'
            descriptors, references, expected_slots, expected_pos, expected_depth = [], [], [], [], []
            for row in range(len(lengths)):
                sl = slice(int(starts[row]), int(starts[row] + lengths[row]))
                slot_tables = [slots[sl][depths[sl] == d].contiguous() for d in [1, 2, 3]]
                pos_tables = [pos[sl][depths[sl] == d].contiguous() for d in [1, 2, 3]]
                references.extend(slot_tables + pos_tables)
                descriptors.append([*[t.data_ptr() for t in slot_tables], *[t.data_ptr() for t in pos_tables],
                                    *[len(t) for t in slot_tables], int(starts[row]), int(lengths[row]), int(qp[row]), int(qd[row])])
                expected_slots.extend(slot_tables)
                expected_pos.extend(pos_tables)
                expected_depth.extend([torch.full((len(t),), d, dtype=torch.int32) for d, t in enumerate(slot_tables, 1)])
            staged_slots, staged_pos, staged_depth = torch.empty_like(slots), torch.empty_like(pos), torch.empty_like(depths)
            staged_starts, staged_lengths, staged_qp, staged_qd = [torch.empty_like(t) for t in [starts, lengths, qp, qd]]
            stage_sources[(len(lengths),)](torch.tensor(descriptors, dtype=torch.long), staged_slots, staged_pos,
                staged_depth, staged_starts, staged_lengths, staged_qp, staged_qd, STREAMS=3, BLOCK=256)
            for actual, expected in [(staged_slots, torch.cat(expected_slots)), (staged_pos, torch.cat(expected_pos)),
                (staged_depth, torch.cat(expected_depth)), (staged_starts, starts), (staged_lengths, lengths),
                (staged_qp, qp), (staged_qd, qd)]:
                assert torch.equal(actual, expected), 'pointer-based staging mismatch'
            report['cases'].append(dict(seed=seed, rows=len(lengths), sources=total, visible=int(ptr[-1])))
    assert not torch.cuda.is_initialized()
    report['cuda_initialized'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f'{args.mode}: {len(report["cases"])} cases PASS; CUDA uninitialized')


if __name__ == '__main__':
    main()
