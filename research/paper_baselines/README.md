# Original TaH benchmark baselines

This folder evaluates the released **Standard-0.6B** and **TaH-plus-0.6B**
checkpoints on the benchmarks in Table 2 of
[Think-at-Hard v4](https://arxiv.org/html/2511.08577v4#S6.T2).
`published_0.6b.json` contains paper-reported reference scores, not scores from our GPU.

The remote checkout is `/root/TaH` on the supplied CUDA machine. Source files
are also kept in the local CS475 checkout. Existing remote experiment files
were preserved. No backbone or router training is performed.

## Evaluation plan

| Benchmark | Problems | Samples/problem | Output token limit |
| --- | ---: | ---: | ---: |
| GSM8K | 1,319 | 1 | 4,096 |
| MATH500 | 500 | 1 | 8,192 |
| AMC23 | 40 | 8 | 8,192 |
| AIME25 | 30 | 8 | 8,192 |
| OlympiadBench | 675 | 1 | 8,192 |
| MMLU-STEM | 3,153 | 1 | 4,096 |
| HumanEval++ | 164 | 2 | 8,192 |
| MBPP++ | 378 | 2 | 8,192 |
| GPQA-Diamond | 198 | 2 | 8,192 |

Temperature is 0.6. The remaining sampling settings follow the released
TaH recipe: top-p 0.95, top-k 20, min-p 0. TaH's continuation threshold is
0.9 and maximum depth is 2. Both checkpoints are frozen and evaluated with
BF16 and SDPA, eight samples per GPU batch, on an L40S-24Q with 24 GiB VRAM.
The complete suite produces **15,374 completions** across both models.

The paper describes the general protocol; the authors' published per-benchmark
recipes additionally specify two samples per GPQA/HumanEval/MBPP problem.
We follow those observed recipes. `prepare.py` recovers exact problem text,
gold answers, sample counts and length budgets from the public evaluation
artifacts in `nics-efc/TaH-plus-0.6B`. It discards the authors' generated
answers. No model is given a reference answer or an earlier generated solution.

For AMC23/AIME25, pass@1 is the mean correctness of the eight independent
samples, not the fraction of problems where any sample succeeds.

## Reproduction limits

- Only Standard and TaH+ have been downloaded for this run. The paper's
  SoftThink, Ouro, AlwaysThink, and pruned TaH rows require their respective
  models and implementations. Forcing two passes on TaH is a routing
  ablation and does not reproduce the separately trained AlwaysThink model.
- All nine datasets use the exact problem strings and gold answers in the
  authors' public evaluation logs, with repeated inputs deduplicated by ID.
  This preserves their MMLU/GPQA options and prompt wording. The direct
  GPQA Hub dataset required authentication; the public evaluation archive
  already supplies the full 198-question evaluation inputs.
- Code prompts use the released repository's code-completion helper,
  including its assistant prefix. Code scores require EvalPlus base **and**
  extra tests. Generation alone does not produce a code accuracy score.
- The primary math/QA score only grades text after a completed `<think>`
  block, preventing intermediate reasoning from being treated as an answer.
  The multiple-choice grader recognizes boxed final letters, which the
  repository's string grader misses. Original repository grading is saved
  alongside our primary score for comparison. Our scores therefore are not
  claimed as an exact recreation of the paper's reported numbers.
- Seeds are deterministic per batch. Changing batch size changes random
  sampling streams. Run settings, model/dataset revisions and runner hash
  are saved; incompatible settings cannot append to an existing run.

## Run and inspect

From the remote repository root:

```bash
.venv-tah/bin/python research/paper_baselines/prepare.py
.venv-tah/bin/python research/paper_baselines/verify.py
.venv-tah/bin/python -u research/paper_baselines/run_suite.py
```

`verify.py` checks greedy output parity against the released generation
helper, left-padding behavior, routing accounting, uneven EOS handling,
and final-answer grading. A short 64-token engineering pilot is stored
separately from the full benchmark results.

The queue runs one dataset at a time, each for Standard then TaH+, and
continues to other datasets if a job fails. `status.json` records running,
queued, complete, failed and blocked jobs. `full_<dataset>.log` contains
progress and diagnostics. A file lock prevents two copies of this queue
from competing for the GPU. The process is launched detached so closing
SSH does not terminate the benchmark run.

Each `full/<dataset>/` directory contains:

- `config.json`: exact run settings and provenance.
- `<model>_<dataset>.jsonl`: per-sample generations, scores, truncation,
  EOS status, token counts, depth histogram and compute measurements.
- `<model>_<dataset>.summary.json`: aggregated measured metrics.
- Code datasets additionally contain solutions and EvalPlus results.
- `COMPLETE`: the dataset's requested model runs have finished.

Runtime includes prefill and decoding. Reported per-sample seconds are
batch runtime divided by batch size, not individual response latency.
Average prediction depth weights emitted non-EOS tokens. Total valid-token
passes also includes prompt processing and EOS-predicting computation,
but excludes masked finished rows; padding and router overhead can still
consume wall-clock time. Peak memory is PyTorch allocated memory, not total
GPU memory reported by `nvidia-smi`.

Interrupted runs resume from recorded `(problem_id, sample_idx)` pairs.
The `COMPLETE` marker applies only to accessible selected inputs; blocked
  inputs, if any, remain visible in the top-level status.
