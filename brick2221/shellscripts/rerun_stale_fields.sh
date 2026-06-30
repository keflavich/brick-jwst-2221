#!/bin/bash
# rerun_stale_fields.sh -- rerun the GC-pipeline fields that were stale and NOT
# already in the SLURM queue, on CURRENT jwst-gc-pipeline code (so the products
# get m8 forced-fill + dedup + a GCPIPEV provenance stamp).
#
# Context (2026-06-29): a GCPIPEV/staleness audit found most products predated the
# Jun-27 provenance hook (GCPIPEV=NONE) and lacked m8.  The PM campaign (gc2211,
# sgra, arches, quintuplet, w51, sickle, sgrb2) was already rerunning from the
# jwst-gc-pipeline-wt-pm worktree; this script covers the REST.
#
# Mechanism: the jwst-gc-pipeline low-resource chain (submit_cataloging_chain.sh
# = per-filter m12..m6 array + m7 cross-band finalize; the m7 finalize auto-runs
# m8 + dedup).  Reductions are only re-run where the on-disk reduction predates
# the Jun-24/28 reduction fixes.  PIPE_ROOT pins the running code (-> GCPIPEV).
#
# Depth per field (see the stale-field audit):
#   cloudc            m7->m8 (m6 already current)
#   cloudef, wd2      m6->m8 (reuse existing reduction)
#   sgrc, wd1         full re-reduction -> cataloging
#   brick 1182/2221   re-catalog from the Jun-20 module-locked crf (keeps the
#                     locked VIRAC2 astrometry, adds m8 on current code).  Brick
#                     is TWO single-proposal runs sharing one data tree, unioned
#                     downstream by merge_catalogs _ok2221or1182 -- NOT one job.
#                     (Canonical re-reduce+m7 runner: run_full_pipeline_brick.sh.)
#
# Suffix note: cloudc/wd2 carry both align_* and destreak_* crf; the documented
# convention (jwst_gc_pipeline _resolve_each_suffix) is SW=destreak, LW=align,
# applied here via --each-suffix-overrides.  VERIFY against a canonical product.
#
# Usage:
#   rerun_stale_fields.sh             # submit everything
#   rerun_stale_fields.sh --dry-run   # print sbatch commands only
#   PIPE_ROOT=/path/to/checkout rerun_stale_fields.sh
set -uo pipefail

PIPE_ROOT=${PIPE_ROOT:-/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline}
export PIPE_ROOT
R="$PIPE_ROOT/scripts/reduction"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
SB() { if (( DRY )); then echo "sbatch $*" >&2; echo "DRYJOB"; else sbatch "$@"; fi; }
COMMIT=$(git -C "$PIPE_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "### PIPE_ROOT=$PIPE_ROOT @ $COMMIT  dry=$DRY"

# m7+m8 finalize only (m6 on disk).  $1 target $2 prop $3 field $4 filters
# $5 each_suffix $6 overrides(csv, may be empty)
m7_only() {
    local T=$1 P=$2 F=$3 FILT=$4 SFX=$5 OV=$6
    [ -n "$OV" ] && export EXTRA_ARGS="--each-suffix-overrides=$OV" || export EXTRA_ARGS=""
    local EXP="ALL,PROPOSAL=$P,FIELD=$F,TARGET=$T,MODULES=merged,EACH_SUFFIX=$SFX"
    EXP="$EXP,FILTERS=$FILT,PARALLEL_WORKERS=4,PIPE_ROOT=$PIPE_ROOT"
    local J; J=$(SB --parsable --job-name="${T}-catalog-m7" \
        --cpus-per-task=4 --mem=64gb --time=24:00:00 \
        --export="$EXP" "$R/submit_cataloging_m7.sbatch")
    echo "  $T m7+m8 -> $J"; unset EXTRA_ARGS
}

# Full cataloging chain (m12..m6 array + m7/m8).  Extra trailing args -> chain env.
chain() {
    local T=$1 P=$2 F=$3 FILT=$4 SFX=$5 OV=$6; shift 6
    [ -n "$OV" ] && export EXTRA_ARGS="--each-suffix-overrides=$OV" || export EXTRA_ARGS=""
    echo "  $T cataloging chain"
    TARGET="$T" PROPOSAL="$P" FIELD="$F" MODULES=merged EACH_SUFFIX="$SFX" \
        FILTERS="$FILT" PIPE_ROOT="$PIPE_ROOT" "$@" \
        bash "$R/submit_cataloging_chain.sh" | sed 's/^/    /'
    unset EXTRA_ARGS
}

# Full re-reduction array; echoes the array job id (for an afterok chain dep).
reduce() {
    local P=$1 F=$2 FILT=$3 T=$4
    read -r -a _A <<< "$FILT"
    # MODULES omitted: nrca,nrcb,merged is the sbatch default and a comma value
    # inside --export is split by SLURM (comma trap).
    SB --parsable --array=0-$(( ${#_A[@]} - 1 )) \
        --export="ALL,PROPOSAL=$P,FIELD=$F,FILTERS=$FILT,PIPE_ROOT=$PIPE_ROOT" \
        --job-name="${T}-reduce" "$R/submit_reduction.sbatch"
}

echo "== cloudc  (m7->m8) =="
m7_only cloudc 2221 002 "F182M F187N F212N F405N F410M F466N" destreak_o002_crf \
    "F405N:align_o002_crf,F410M:align_o002_crf,F466N:align_o002_crf"

echo "== cloudef  (m6->m8) =="
chain cloudef 2092 002 "F162M F210M F360M F480M" destreak_o002_crf ""

echo "== wd2  (m6->m8; SW=destreak/LW=align) =="
chain wd2 3523 005 \
 "F115W F150W F162M F164N F182M F187N F200W F212N F250M F277W F300M F323N F335M F405N F410M F444W F466N" \
 destreak_o005_crf \
 "F250M:align_o005_crf,F277W:align_o005_crf,F300M:align_o005_crf,F323N:align_o005_crf,F335M:align_o005_crf,F405N:align_o005_crf,F410M:align_o005_crf,F444W:align_o005_crf,F466N:align_o005_crf"

echo "== sgrc  (re-reduce -> catalog) =="
SGRC=$(reduce 4147 012 "F115W F162M F182M F212N F360M F405N F470N F480M" sgrc)
echo "  sgrc reduction -> $SGRC"
chain sgrc 4147 012 "F115W F162M F182M F212N F360M F405N F470N F480M" destreak_o012_crf "" DEP="afterok:$SGRC"

echo "== wd1  (re-reduce -> catalog) =="
WD1=$(reduce 1905 001 "F115W F150W F164N F187N F200W F212N F277W F323N F405N F444W F466N" wd1)
echo "  wd1 reduction -> $WD1"
chain wd1 1905 001 "F115W F150W F164N F187N F200W F212N F277W F323N F405N F444W F466N" destreak_o001_crf "" DEP="afterok:$WD1"

# Brick: re-catalog from the existing module-locked crf (no re-reduction).  Two
# single-proposal chains; dense broadband -> 8cpu/48h.
echo "== brick 1182/004  (re-catalog locked crf) =="
chain brick 1182 004 "F115W F200W F356W F444W" destreak_o004_crf "" \
    PERFILTER_CPUS=8 PERFILTER_MEM=96gb PERFILTER_TIME=48:00:00

echo "== brick 2221/001  (re-catalog locked crf) =="
chain brick 2221 001 "F182M F187N F212N F405N F410M F466N" destreak_o001_crf "" \
    PERFILTER_CPUS=8 PERFILTER_MEM=96gb PERFILTER_TIME=48:00:00

echo "### DONE.  Already-in-queue (do NOT duplicate): gc2211 sgra arches quintuplet w51 sickle sgrb2."
