#!/usr/bin/env bash
# Multi-target wrapper around run_full_pipeline_common.sh.
#
# Iteration coverage: see run_full_pipeline_common.sh.  This script just
# delegates: it loops over a list of targets and runs the common
# pipeline + iter1 + merge flow for each.
#
# Default target list is the historical "non-brick GC fields" set.  Pass
# targets on the command line to override.
#
# Until 2026-06-07 this file accidentally contained TWO concatenated
# scripts (a wrapper and the original standalone end-to-end), so every
# invocation submitted each target's full pipeline twice.  Cleaned up to
# a single delegating wrapper.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
common_script="${script_dir}/run_full_pipeline_common.sh"

run_targets=("$@")
if [[ ${#run_targets[@]} -eq 0 ]]; then
    run_targets=(cloudef sgrc sgrb2 arches quintuplet sgra)
fi

for target in "${run_targets[@]}"; do
    "${common_script}" "${target}"
done
