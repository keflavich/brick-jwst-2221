#!/usr/bin/env bash
# submit_manual_pipeline.sh -- new default PSF-photometry path
# (jwst_gc_pipeline.photometry.cataloging.run_manual_pipeline).
#
# This is THE default photometry pipeline as of 2026-06-09; see
#   /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline/PHOTOMETRY_PIPELINE.md
# and README.md in that repo for full docs.
#
# The manual-iteration pipeline runs every phase (m12 -> m3 -> m4 -> m5
# -> m6 (-> m7 if multi-filter)) IN-PROCESS as a single non-array job
# (phases are sequential; cannot be split across a SLURM array).  This
# script submits one sbatch per (target, filter[_csv], module, field).
# Multi-field targets (e.g. cloudef obs 002 + 005) get one submission
# per field; pass multiple filters comma-separated to engage the
# cross-filter m7 seed.
#
# Usage:
#   submit_manual_pipeline.sh <target> <filter[,filter,...]> <module> [tag] [extra_dep]
#
# Examples:
#   # Single-filter test
#   submit_manual_pipeline.sh sgrb2 F212N nrcb V9
#   # Multi-filter (engages m7 cross-band seed):
#   submit_manual_pipeline.sh sgrb2 F210M,F212N nrcb V9
#
# The legacy iter1->iter2->merge->iter3 chain is in submit_full_chain.sh
# (kept for in-flight brick rerun).  Eventually that script will also
# default to this path; this companion runner is the migration target.

set -e

target="${1:?usage: submit_manual_pipeline.sh <target> <filter[,filter,...]> <module> [tag] [extra_dep]}"
filter="${2:?missing filter}"
module="${3:?missing module}"
tag="${4:-V9}"
extra_dep="${5:-}"

PYTHON=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
SCRIPT=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py

python_target=""

case "$target" in
  sickle)
    basepath=/orange/adamginsburg/jwst/sickle ; proposal_id=3958
    fields=(007)
    logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/ ;;
  brick)
    basepath=/blue/adamginsburg/adamginsburg/jwst/brick ; proposal_id=2221
    fields=(001)
    logdir=/blue/adamginsburg/adamginsburg/brick_logs ;;
  brick-1182)
    basepath=/blue/adamginsburg/adamginsburg/jwst/brick ; proposal_id=1182
    fields=(004) ; python_target=brick
    logdir=/blue/adamginsburg/adamginsburg/brick_logs ;;
  cloudc)
    basepath=/blue/adamginsburg/adamginsburg/jwst/cloudc ; proposal_id=2221
    fields=(002)
    logdir=/blue/adamginsburg/adamginsburg/cloudc_logs ;;
  sgrb2)
    basepath=/orange/adamginsburg/jwst/sgrb2 ; proposal_id=5365
    fields=(001)
    logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/ ;;
  sgra)
    basepath=/orange/adamginsburg/jwst/sgra ; proposal_id=1939
    fields=(001)
    logdir=/blue/adamginsburg/adamginsburg/logs/sgra_jwst/ ;;
  cloudef)
    basepath=/orange/adamginsburg/jwst/cloudef ; proposal_id=2092
    fields=(002 005)
    logdir=/blue/adamginsburg/adamginsburg/logs/cloudef_jwst/ ;;
  arches)
    basepath=/orange/adamginsburg/jwst/arches ; proposal_id=2045
    fields=(001)
    logdir=/blue/adamginsburg/adamginsburg/logs/arches_jwst/ ;;
  quintuplet)
    basepath=/orange/adamginsburg/jwst/quintuplet ; proposal_id=2045
    fields=(003)
    logdir=/blue/adamginsburg/adamginsburg/logs/quintuplet_jwst/ ;;
  sgrc)
    basepath=/orange/adamginsburg/jwst/sgrc ; proposal_id=4147
    fields=(012)
    logdir=/blue/adamginsburg/adamginsburg/logs/sgrc_jwst/ ;;
  gc2211-023|gc2211-028|gc2211-046|gc2211-049|gc2211-050)
    basepath=/orange/adamginsburg/jwst/gc2211 ; proposal_id=2211
    fld="${target#gc2211-}" ; fields=("${fld}") ; python_target=gc2211
    logdir=/blue/adamginsburg/adamginsburg/logs/gc2211_jwst/ ;;
  *) echo "Unknown target: $target" >&2 ; exit 2 ;;
esac

python_target="${python_target:-${target}}"
mkdir -p "$logdir"

DEP=""
if [[ -n "$extra_dep" ]]; then DEP="--dependency=afterok:$extra_dep"; fi

# sgrb2 uses align_o<NNN>_crf for LW filters, destreak_o<NNN>_crf for SW.
get_each_suffix() {
  local fld="$1" filt="$2"
  if [[ "$target" == "sgrb2" ]]; then
    case "$filt" in
      F150W|F182M|F187N|F210M|F212N) echo "destreak_o${fld}_crf" ;;
      *)                              echo "align_o${fld}_crf" ;;
    esac
  else
    echo "destreak_o${fld}_crf"
  fi
}

# Per-target resource defaults; override via env vars.
# Phases are sequential, but per-frame fits within a phase parallelize
# via --parallel-workers. Default to a fat single-node config.
MEM=${MANUAL_MEM:-256gb}
CPUS=${MANUAL_CPUS:-32}
WALLTIME=${MANUAL_TIME:-96:00:00}

submitted=()
for fld in "${fields[@]}"; do
  # For multi-filter input, suffix is taken from the first filter for the
  # CLI flag; the script's per-frame loop reads --each-suffix verbatim per
  # filter from the same value, so multi-filter calls must use a single
  # suffix family.  sgrb2 LW/SW mixes will require --each-suffix per
  # filter (TODO); document and reject for now.
  first_filt="${filter%%,*}"
  each_suffix=$(get_each_suffix "$fld" "$first_filt")
  if [[ "${filter}" == *,* ]] && [[ "$target" == "sgrb2" ]]; then
    # detect any LW/SW mix across the CSV (sgrb2 only)
    for f in ${filter//,/ }; do
      sfx=$(get_each_suffix "$fld" "$f")
      if [[ "$sfx" != "$each_suffix" ]]; then
        echo "ERROR: sgrb2 multi-filter mixes LW/SW (different each_suffix). Split into separate calls per family." >&2
        exit 2
      fi
    done
  fi

  JOB=$(sbatch --parsable $DEP \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=1 --cpus-per-task=${CPUS} --nodes=1 \
    --mem=${MEM} --time=${WALLTIME} \
    --job-name=webb-manual-${target}-o${fld}-${filter//,/+}-${module}-${tag} \
    --output=${logdir}/webb-manual-${target}-o${fld}-${filter//,/+}-${module}-${tag}_%j.log \
    --wrap "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \
      ${PYTHON} ${SCRIPT} \
        --filternames=${filter} --modules=${module} \
        --proposal_id=${proposal_id} --field=${fld} \
        --target=${python_target} --each-suffix=${each_suffix} \
        --each-exposure --daophot --skip-crowdsource --skip-if-done \
        --parallel-workers=${CPUS}")
  submitted+=("${JOB}")
  echo "submitted manual-iter ${target}/o${fld}/${filter}/${module} = ${JOB}"
done

csv=$(IFS=,; echo "${submitted[*]}")
echo "${target}/${filter}/${module}: jobs=[${csv}]"
