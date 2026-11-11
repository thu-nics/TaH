#!/usr/bin/env python3
"""Reference W-perf generation plus sampled device-memory measurement (GPU)."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
from curate import DATA, REPO, ROOT, dump, Collector

out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
config=ROOT/'scripts/config/reference-perf.yaml'
command=[str(REPO/'.venv-tah/bin/python'),'script/tah/evaluation/eval.py','--eval_config',str(config),'--model_path','models/TaH-plus-1.7B','--dataset_name','math500','--data_range','0','64','--backend','tah','--random_seed','42','--job_nums','1','--tp_size_per_job','1','--grading-workers','0','--output_dir',str(out/'run')]
print(' '.join(command),flush=True)
monitor=out/'device-memory.csv'
with monitor.open('w') as memory:
    process=subprocess.Popen(['nvidia-smi','--id='+os.environ.get('CUDA_VISIBLE_DEVICES','0'),'--query-gpu=memory.used','--format=csv,noheader,nounits','--loop=5'],stdout=memory)
    try:subprocess.run(command,cwd=REPO,check=True)
    finally:process.terminate();process.wait()
files=list((out/'run').rglob('detailed_results.csv'))
assert len(files)==1,files
rows=list(csv.DictReader(files[0].open()))
assert len(rows)==128 and len({(r['problem_id'],r['sample_idx']) for r in rows})==128
seconds=sum(float(r['processing_time']) for r in rows)
tokens=sum(int(r['output_tokens']) for r in rows)
reference=dict(generated_tokens=tokens,inference_wall_seconds=seconds,published_tokens_per_second=tokens/seconds,peak_sampled_memory_mib=max(float(x.strip()) for x in monitor.read_text().splitlines()),memory_sampling_interval_seconds=5,timing_scope='CSV batch inference contributions')
dump(out/'reference.json',reference)
collector=Collector({'reference':str(out/'reference.json')});collector.load('reference')
current=json.loads((DATA/'speed.json').read_text())
current['reference']=reference;current['reference_provenance']=dict(csv_path=str(files[0]),monitor_path=str(monitor),config_path=str(config))
# Keep provenance for the untouched engine observations as well.
collector.used+=json.loads((DATA/'MANIFEST.json').read_text())['files']['speed.json'].get('sources',[])
collector.save('speed.json',current,'Reference batch inference timing and sampled device memory; engine throughput and allocator peaks.')
