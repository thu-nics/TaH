"""CPU integration check for rebuilding the saved nested O1 manifest."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from script.tah.validation.select_o1_manifest import default_sources
from script.tah.validation.select_sequences import parse_source


REPO = Path(__file__).resolve().parents[4]


def test_cli_rebuilds_frozen_o1_manifest_byte_identically(tmp_path):
    frozen = REPO / 'data/engine/oracle/wp23c/inputs/manifest.json'
    tokenizer = REPO / 'models/TaH-plus-1.7B'
    if not frozen.is_file() or not tokenizer.exists():
        pytest.skip('saved study inputs or tokenizer are not present in this checkout')
    missing = [str(path) for _, _, path in map(parse_source, default_sources(Path('data/engine')))
               if not (REPO / path).is_file()]
    if missing:
        pytest.skip('saved evaluation samples are missing: ' + ', '.join(missing))

    output = tmp_path / 'regenerated-manifest.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(
        [sys.executable, '-m', 'script.tah.validation.select_o1_manifest', '--output', str(output)],
        cwd=REPO, env=env, check=True,
    )
    assert output.read_bytes() == frozen.read_bytes()
