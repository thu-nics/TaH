#!/usr/bin/env python3
"""Named steps that regenerate the chapter's data. No arguments list the steps.

    reproduce.sh r2 SEED [--output-root DIR]         GPU: R2 protocol run + merge (protocol_paper.yaml)
    reproduce.sh code_tah2 SEED [--output-root DIR]  GPU: TaH2-prompt code run + merge (protocol_paper_code_tah2.yaml)
    reproduce.sh curate [--output-root DIR]          CPU: curate_r2.py from DIR, then analyze.py

STEP --dry-run prints the exact commands without running them. Runs are written to
DIR/paper_protocol_seedSEED/ and DIR/paper_code_tah2_seedSEED/ (default DIR:
data/engine/r2-reproduction, outside the curated data/). The registered runs live in
data/engine/; `curate --output-root data/engine` re-curates them.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PY = REPO / '.venv-tah/bin/python'
RECIPES = 'script/tah/recipes/qwen3_1.7'
SEEDS = (42, 1234, 2024)
STEPS = {'r2': ('protocol_paper.yaml', 'paper_protocol_seed'),
         'code_tah2': ('protocol_paper_code_tah2.yaml', 'paper_code_tah2_seed')}


def chapter():
    spec = importlib.util.spec_from_file_location('baseline_analysis', ROOT / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def listing():
    analysis = chapter()
    t = analysis.runtimes()
    hours, minutes = analysis.hours, analysis.minutes
    print(f"r2 SEED: one R2 protocol run (9 benchmarks), {hours(t['r2']['min'])}–{hours(t['r2']['max'])} per seed "
          f"(measured); one 24 GB GPU; writes OUTPUT_ROOT/paper_protocol_seedSEED/")
    print(f"code_tah2 SEED: HumanEval+ and MBPP+ with the TaH2 code prompt, {minutes(t['code-tah2']['min'])}–"
          f"{minutes(t['code-tah2']['max'])} per seed (measured); GPU; writes OUTPUT_ROOT/paper_code_tah2_seedSEED/")
    print('curate: curate_r2.py from OUTPUT_ROOT, then analyze.py; seconds, CPU; writes one sample table and one run-metadata JSON per study, plus shared protocol and dataset data.')
    print(f'Registered seeds: {", ".join(map(str, SEEDS))}. Use STEP [SEED] --dry-run for exact commands. No work executed.')


def plan(step, seed, out):
    if step == 'curate':
        return [([str(PY), 'report/2-baseline/scripts/curate_r2.py', '--runs-root', str(out)], False),
                ([str(PY), 'report/2-baseline/analyze.py'], False)]
    config, prefix = STEPS[step]
    root = out / f'{prefix}{seed}'
    return [([str(PY), 'script/tah/evaluation/run_protocol.py', 'run', '--config', f'{RECIPES}/{config}',
              '--model_path', 'models/TaH-plus-1.7B', '--backend', 'tah_engine', '--seed', str(seed),
              '--output_root', str(root), '--logger_level', 'WARNING'], True),
            ([str(PY), 'script/tah/evaluation/run_protocol.py', 'merge', '--output_root', str(root)], False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', nargs='?', choices=[*STEPS, 'curate'])
    parser.add_argument('seed', nargs='?', type=int)
    parser.add_argument('--output-root', type=Path, default=REPO / 'data/engine/r2-reproduction')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.step:
        listing()
        return
    if args.step in STEPS and args.seed not in SEEDS:
        parser.error(f'{args.step} needs a registered seed: {", ".join(map(str, SEEDS))}')
    out = args.output_root.resolve()
    if out == ROOT or out.is_relative_to(ROOT):
        parser.error('Keep full run outputs outside the chapter directory')
    commands = plan(args.step, args.seed, out)
    for command, _ in commands:
        print(shlex.join(command), flush=True)
    print('Then run `reproduce.sh curate` (CPU) to refresh data/ and the tables.' if args.step in STEPS else
          'Writes data/, README.md, APPENDIX.md, tables/ and figures/.', flush=True)
    if args.dry_run:
        return
    for command, gpu in commands:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=args.gpu if gpu else '')
        subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == '__main__':
    main()
