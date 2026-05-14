#!/usr/bin/env bash
# Pipeline for JWST proposal 2211 (asteroid survey) -- the 5 GC pointings
# only.  Each observation gets its own pass through run_full_pipeline_common.sh
# with FIELD/FILTERS overridden via env vars, because the 5 obs IDs occupy
# different fields (023/028/046/049/050) and use different filter sets:
#
#   obs 023: F200W, F277W  ( 40 frames)
#   obs 028: F150W, F277W  (240 frames)
#   obs 046: F200W, F277W  (240 frames)
#   obs 049: F200W, F277W  (160 frames)
#   obs 050: F200W, F277W  ( 60 frames)
#
# All 5 obs share the same /orange/adamginsburg/jwst/gc2211/ basepath; the
# per-obs CRF outputs differ only by the destreak_o${field}_crf suffix.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A FILTERS_BY_OBS
FILTERS_BY_OBS[023]=F200W,F277W
FILTERS_BY_OBS[028]=F150W,F277W
FILTERS_BY_OBS[046]=F200W,F277W
FILTERS_BY_OBS[049]=F200W,F277W
FILTERS_BY_OBS[050]=F200W,F277W

# Default: run all 5 obs.  Override with GC2211_OBS=028,046 to subset.
OBS_LIST=${GC2211_OBS:-023,028,046,049,050}

IFS=',' read -ra obs_arr <<< "${OBS_LIST}"
for obs in "${obs_arr[@]}"; do
    if [[ -z "${FILTERS_BY_OBS[$obs]:-}" ]]; then
        echo "Unknown gc2211 obs id: ${obs}" >&2
        exit 2
    fi
    echo "=== gc2211 obs=${obs} filters=${FILTERS_BY_OBS[$obs]} ==="
    FIELD=${obs} FILTERS=${FILTERS_BY_OBS[$obs]} REF_FILTER=${FILTERS_BY_OBS[$obs]%%,*} \
        "${script_dir}/run_full_pipeline_common.sh" gc2211
done
