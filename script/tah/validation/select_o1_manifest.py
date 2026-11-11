"""Rebuild the frozen O1 sequence manifest from saved evaluation samples.

Run from the repository root with:
    python -m script.tah.validation.select_o1_manifest --output PATH
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from script.tah.validation.select_sequences import DEFAULT_SOURCES, main as select_sequences


SEEDS = (42, 1234, 2024)
DATASETS = ('math500', 'gpqa', 'humaneval', 'mbpp')


def default_sources(data_root: Path) -> list[str]:
    sources = list(DEFAULT_SOURCES[:2])
    for seed in SEEDS:
        canonical = data_root / f'paper_protocol_seed{seed}' / 'canonical'
        for dataset in DATASETS:
            matches = sorted(canonical.glob(f'{dataset}_tah_engine/**/samples.jsonl'))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f'Expected one saved {dataset} sample file for seed {seed} under {canonical}; found {len(matches)}'
                )
            sources.append(f'engine:{dataset}={matches[0]}')
    return sources


def _select(output: Path, sources: list[str], tokenizer: Path, count: int, positions: int) -> dict:
    args = [
        '--min-sequences', str(count), '--min-positions', str(positions),
        '--validity-batch', '4', '--tokenizer', str(tokenizer), '--output', str(output),
    ]
    for source in sources:
        args.extend(['--source', source])
    select_sequences(args)
    manifest = json.loads(output.read_text())
    sequences = sorted(manifest['sequences'], key=lambda item: (len(item['forced']), item['id']))
    manifest['sequences'] = sequences
    manifest['selection']['ordering'] = 'ascending generated positions, then id; before confirmatory runs'
    output.write_text(json.dumps(manifest, sort_keys=True) + '\n')
    return manifest


def rebuild(output: Path, data_root: Path, tokenizer: Path, source_overrides: list[str] | None = None) -> None:
    """Write the final nested stage1/stage2/validity/b1 manifest to output."""
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = source_overrides if source_overrides is not None else default_sources(data_root)
    with tempfile.TemporaryDirectory(prefix='o1-manifest-', dir=output.parent) as temporary:
        work = Path(temporary)
        first_path = work / 'stage1.json'
        maximum_path = work / 'maximum.json'
        first = _select(first_path, sources, tokenizer, 64, 160_000)
        first_sequences = first['sequences']
        ordered = first_sequences
        chosen = ordered[2:20] + ordered[-2:]
        if len(chosen) != 20 or sum(len(item['forced']) for item in chosen) < 20_000:
            raise ValueError('The selected validity subset does not meet its frozen size requirements')
        chosen_ids = {item['id'] for item in chosen}
        first['subsets']['validity'] = [item['id'] for item in ordered if item['id'] in chosen_ids]
        first['selection']['validity_rule'] = '20 shortest, replacing the two shortest by the two longest'
        first_path.write_text(json.dumps(first, sort_keys=True) + '\n')

        maximum = _select(maximum_path, sources, tokenizer, 200, 500_000)
        first_ids = {item['id'] for item in first_sequences}
        if not first_ids <= {item['id'] for item in maximum['sequences']}:
            raise ValueError('The 200-sequence selection does not contain the 64-sequence stage-1 set')
        master = maximum
        master['sequences'] = first_sequences + [item for item in maximum['sequences'] if item['id'] not in first_ids]
        master['subsets'] = dict(
            stage1=[item['id'] for item in first_sequences],
            stage2=[item['id'] for item in master['sequences'][64:]],
            validity=first['subsets']['validity'],
            b1=[next(item['id'] for item in first_sequences if item['dataset'] == dataset)
                for dataset in sorted(first['totals']['by_dataset'])],
        )
        master['selection'].update(
            stage1_sequences=64,
            maximum_sequences=200,
            stage1_min_positions=160_000,
            ordering='stage1 sorted by length/id, then stage2 sorted by length/id',
            validity_rule=first['selection']['validity_rule'],
            b1_rule='shortest selected sequence per dataset',
        )
        output.write_text(json.dumps(master, sort_keys=True) + '\n')
    print(f'Wrote nested O1 manifest: {output} ({len(master["sequences"])} sequences)')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='destination manifest JSON')
    parser.add_argument('--data-root', type=Path, default=Path('data/engine'),
                        help='root containing paper_protocol_seed{42,1234,2024}/canonical')
    parser.add_argument('--tokenizer', type=Path, default=Path('models/TaH-plus-1.7B'))
    parser.add_argument('--source', action='append', default=None,
                        help='override all input sources (backend:dataset=path; repeatable)')
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    rebuild(args.output, args.data_root, args.tokenizer, args.source)


if __name__ == '__main__':
    main()
