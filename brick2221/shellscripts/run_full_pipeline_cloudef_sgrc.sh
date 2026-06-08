#!/usr/bin/env bash
# Multi-target wrapper around run_full_pipeline_common.sh.
#
# Despite the historical "cloudef_sgrc" name, this script is a generic
# multi-target launcher: it loops over whatever target list you pass on
# the command line and runs run_full_pipeline_common.sh for each.
#
# Iteration coverage: see run_full_pipeline_common.sh -- pipeline +
# refcat + iter1 + merge.  iter2/iter3/iter4 NOT included.
#
# Default target list (when called with no args):
#     cloudef sgrc sgrb2 arches quintuplet sgra
#
# Targets NOT in the default list:
#   - brick (proposal 2221 narrowband + 1182 broadband):
#       reduced separately; use submit_full_chain.sh and
#       run_iter3_cataloging.sh with --target brick / brick-1182.
#   - gc2211 (asteroid survey, 5 GC obs): use run_full_pipeline_gc2211.sh
#       (it loops obs 023 028 046 049 050 inside one run).
#
# To run literally every field this codebase knows about:
#     bash run_full_pipeline_cloudef_sgrc.sh cloudef sgrc sgrb2 arches quintuplet sgra
#     bash run_full_pipeline_gc2211.sh
#     # (brick is reduced via submit_full_chain.sh as above)
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
