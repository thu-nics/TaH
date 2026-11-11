# Developer guide

For people changing the code: how the internals fit together and how to check that a change is still correct. Using the repository (install, evaluate, read results, call the engine) is covered by the user guide in [`../guide/`](../guide/README.md).

| Topic | Read before changing |
|---|---|
| [engine.md](engine.md) | anything in `tah/engine/` — modules, test suite, parity/validation tools, cache/attention APIs, overlap scheduler, sampling internals |
| [validation.md](validation.md) | anything in `script/tah/validation/` — oracle measurement (R1-O1), coupled-sampling primitives, how to run the oracle evaluation |
| [evaluation.md](evaluation.md) | `tah/evaluate/` and `script/tah/evaluation/` — dataset mapping, prompts, grading, protocol runner and analysis |

## Conventions (keep this guide small)

- **One topic file per area** (`engine.md`, later e.g. `evaluation.md`, `training.md`); add a row to the table above. No documentation files inside code directories.
- **Fixed sections per file, in this order:** *Scope* (what the area is, 2–3 sentences) → *Module map* (one line per module) → *Validating a change* (which tests/tools to run for which kind of change, exact commands) → *Internals reference* (only contracts that are not obvious from the code: invariants, ordering rules, numerical traps).
- **Size:** aim for ≤ ~400 lines per file; above that, split by sub-topic rather than growing.
- **Not here:** measured results or benchmarks (→ `report/`), user-facing usage (→ `guide/`), history of how something was developed, anything a docstring or `--help` already says (link to the code instead).
