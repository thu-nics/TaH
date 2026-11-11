# Report chapters — format

`report/` holds the evidence for the research report (LaTeX on Overleaf). One numbered directory per major change or experiment: `1-engine/`, `2-baseline/`, `3-…`. `report/README.md` indexes the chapters. Everything is in English and committed.

A chapter is a **curated summary for the report**, not an archive. Include only what supports the chapter's claims; complete raw outputs stay in the experiment output directories and are referenced, not copied.

Each chapter contains:
- `README.md` — about 1–2 pages, structured as:
  1. **Summary** (4–6 sentences): what was done, the headline results, what is still open.
  2. **What was done** — short; the few design or method points a reader needs, each with one sentence on why.
  3. **Results** — one subsection per claim: the claim, how it was tested (2–3 sentences), one compact table or figure, one or two sentences of interpretation, and a one-line **Reproduce:** under each table/figure — the `reproduce.sh` step that regenerates its data, its runtime and GPU need, and the data file(s) it writes. Prefer one row per item that matters (e.g. per checkpoint) over exhaustive breakdowns. Explain every column that is not self-evident.
  4. **Limitations and negative results** — short bullets.
  5. **Reproduce and data** — a table mapping each table/figure → `reproduce.sh` step → command it runs → runtime → data files; then a short data guide listing only the files in `data/` with what each contains.
- `README.template.md` — the human-editable prose with placeholders for tables and numbers; `analyze.py` fills them and writes `README.md`. Edit the template, never the generated README.
- `data/` — only the data the README's tables and figures are computed from: metric JSON, per-sample CSV, large text compressed as `.jsonl.xz`. Target a few MB; never multi-GB tensors. **Compact layout:** one long-format per-sample table per study (columns identifying run, seed, dataset, problem, sample; only the columns the analysis reads) and one run-metadata JSON per study. Never mirror per-run or per-benchmark directory trees of the source outputs; provenance goes into `MANIFEST.json`. Only final (confirmatory) runs; exploratory runs stay outside the chapter. `data/MANIFEST.json` records checkpoint SHA256, environment versions, git commit, source path and SHA256 of every data file, seeds and dates.
- `analyze.py` — CPU only, seconds: reads only `data/`, regenerates every number, table and figure, writes `README.md`, `tables/*.tex` (booktabs, `\input`-able in Overleaf) and `figures/*.pdf`. Deterministic: a second run is byte-identical. No number is typed by hand into the template or the LaTeX source.
- `reproduce.sh` (or `.py`) — GPU, may take hours: regenerates `data/` from the checkpoint/fixtures with code in this repository. Organized as named steps that match the README tables (e.g. `./reproduce.sh speed`, `./reproduce.sh equivalence`); `./reproduce.sh` without arguments lists the steps with what each produces, runtime and GPU need. Each step prints the exact commands it runs and writes into `data/`, after which `analyze.py` refreshes the tables.
- Optional `APPENDIX.md` (generated) for full-detail tables a reviewer might ask for; keep it short too.

Describe results and methods, not the development process (no task/agent bookkeeping).

**Write for non-experts.** A reader outside the field must be able to follow every chapter. When a concept, metric or statistical method is introduced (e.g. total-variation distance, teacher forcing, non-inferiority/TOST, bootstrap confidence interval, coupled sampling), explain in plain language what it measures and why it answers the question, the first time it appears. Paraphrase specialist terms in ordinary words, within reason. Nothing is left unexplained.
