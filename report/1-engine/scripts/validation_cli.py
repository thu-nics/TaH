#!/usr/bin/env python3
"""Path adapter for existing GPU validation; no numerical or gate changes.

The repository CLI historically hard-codes ignored fixture paths. This adapter
points it at the reproduction directory and archived workload prompts without
editing the engine or writing into the historical evidence directories.
"""
import importlib
import json
import lzma
import os
from pathlib import Path
import sys


def fixture_setup(name):
    import yaml
    from tah.engine.fixtures import FixtureSuite
    from tah.engine.parity import require_production, production_noise
    from tah.engine.tests.engine_check import digest
    suite=FixtureSuite(os.environ['TAH_GTF_FIXTURES'])
    require_production(suite)
    meta=suite.manifest['checkpoints'][name];root=Path(meta['path'])
    for file,info in meta['files'].items():
        assert digest(root/file)==info['sha256'], file
    for file,value in suite.manifest['reference_source_sha256'].items():
        assert digest(file)==value, file
    eval_config=Path(suite.manifest['eval_config']['path'])
    assert digest(eval_config)==suite.manifest['eval_config']['sha256']
    override=yaml.safe_load(eval_config.read_text()) if name=='real' else None
    noise=production_noise(suite,name,json.loads(Path(os.environ['TAH_GTF_TOLERANCES']).read_text()))
    records=[r for r in suite.records(kind='greedy',checkpoint=name) if r['variant']=='single']
    prompts=[suite.load(r)['prompt_ids'].tolist() for r in records]
    return suite,root,override,noise,records,prompts,eval_config


def workload_prompts(model_path='models/TaH-plus-1.7B',repeat=2):
    from transformers import AutoTokenizer
    chapter=Path(__file__).resolve().parents[1]
    path=chapter.parents[1]/json.loads((chapter/'scripts/sources.json').read_text())['workload']
    with (lzma.open(path,'rt') if path.suffix=='.xz' else path.open()) as f:rows=[json.loads(line) for line in f]
    questions={row['id']:row['problem'] for row in rows}
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    return [tokenizer.apply_chat_template([dict(role='user',content=questions[f'math500_{i}'])],
        tokenize=True,add_generation_prompt=True) for i in range(len(questions)) for _ in range(repeat)]


def main():
    name=sys.argv.pop(1)
    if name not in ('engine_check','generated_gtf'):raise SystemExit('Unknown validator')
    check=importlib.import_module('tah.engine.tests.engine_check')
    old_digest=check.digest
    def digest(path):
        if Path(path).name=='tolerances.json' and 'TAH_GTF_TOLERANCES' in os.environ:
            path=os.environ['TAH_GTF_TOLERANCES']
        return old_digest(path)
    check.digest=digest
    check.fixture_setup=fixture_setup
    check.workload_prompts=workload_prompts
    module=check if name=='engine_check' else importlib.import_module('tah.engine.tests.generated_gtf')
    module.main()

if __name__=='__main__':main()
