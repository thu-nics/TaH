# Agent notes for this repository

## Documentation
How to install, run evaluations (harness CLI, YAML keys, backends, datasets), read results and use the TaH engine is documented in the user guide: start at [`guide/README.md`](guide/README.md). Read the relevant topic file before using or changing those interfaces; when you change a user-facing interface (CLI flag, config key, output format, public API), update the matching guide file in the same change. Keep the guide concise and user-facing: measured results belong in `report/`, developer and validation documentation in `dev-guide/`.

## Report evidence
Every major result that goes into the research report (LaTeX, written on Overleaf) is recorded as a numbered chapter directory under `report/` (e.g. `report/1-engine/`), committed to git and written in English: a short curated summary for the report reader (not an archive), only the data its tables need, and scripts that regenerate the tables and the data. Format and rules: [`report/AGENTS.md`](report/AGENTS.md).

## Code changes
Prefer additive changes: keep original code paths intact; new functionality goes into new modules or scripts unless changing existing code is unavoidable. Before changing an area, read its topic in [`dev-guide/`](dev-guide/README.md) (how to validate a change) and keep it current, following the conventions there.

## Evaluation for the report
Report evaluations follow [`report/PROTOCOL.md`](report/PROTOCOL.md) (benchmarks, seeds, analysis). Run them with the protocol configs it names, not ad-hoc settings.
