#!/usr/bin/env bash
# No arguments list named steps; STEP --dry-run prints commands without running them.
set -euo pipefail
chapter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$chapter_dir/scripts/reproduce.py" "$@"
