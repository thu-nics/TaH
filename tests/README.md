## What's here

This directory holds the TaH test suite: 11 pytest files (`test_*.py`) covering individual
components (iter label, loss, input/output updaters, iter decider, causal cache),
wrapper-level forward/save-load/sft smoke checks, the released-checkpoint sanity test,
and the eval jobs runner; a shared `_harness.py` that captures and diffs per-component
baselines against upstream public TaH; a `conftest.py` with device + seed fixtures; and
two standalone benchmark scripts (`bench.py`, `bench_compile.py`) sitting alongside the
tests rather than under pytest.

## Run the tests

```bash
pytest tests/ -q                  # all tests
pytest tests/test_<name>.py -v    # single file
TAH_TEST_DEVICE=cpu pytest tests/ # force CPU (auto-detects CUDA otherwise)
```

`TAH_TEST_DEVICE` is read by `conftest.py` and `_harness.py` to pick the device for
both fixtures and baseline subprocesses. The simple component tests
(`test_iter_label.py`, `test_loss.py`, `test_input_updater.py`,
`test_output_updater.py`) run cleanly on CPU; tests that touch the released checkpoint
or full wrapper (e.g. `test_released_checkpoint.py`) require a GPU and will download
`nics-efc/TaH-plus-1.7B` from Hugging Face on first run.

## Baseline-snapshot harness

On first run, `_harness.py` spawns a subprocess scoped to `/tmp/TaH-pub` (a checkout
of the public upstream TaH at [thu-nics/TaH](https://github.com/thu-nics/TaH)) to
capture per-component `.pt` snapshots into `tests/baselines/`. Subsequent runs in
this repo diff cleaned tah-release outputs against the recorded snapshots, giving the
suite drift detection between this cleaned fork and upstream. The `tests/baselines/`
directory is gitignored — snapshots regenerate on first run.

## Benchmarks

```bash
python tests/bench.py components   # microbenchmarks for the wrapper's hot helpers
python tests/bench.py e2e          # forward + 32-token generate on TaH-plus-1.7B
python tests/bench_compile.py      # one-off torch.compile vs eager experiment
```

## Component baselines

Single B200, torch 2.11+cu128, bf16:

| helper | ms |
|---|---|
| topk_softmax_input_update | 0.48 |
| additive_logits_update | 0.03 |
| gather_active | 0.19 |
| scatter_back | 0.12 |
| MLPIterDecider.forward | 0.86 |
| NextTokenPredLoss.final | 0.23 |
| IterDeciderLoss.intra | 0.55 |
| **TaHForCausalLM.forward** (TaH-plus-1.7B, T=15) | **18.0** |
| **TaHForCasualLM_generate(32)** | **691** (~21.6 ms / token) |
