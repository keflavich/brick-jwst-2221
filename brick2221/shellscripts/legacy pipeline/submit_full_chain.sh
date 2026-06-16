#!/usr/bin/env bash
# ============================================================
# DEPRECATED -- legacy iter1-4 cataloging pipeline.
# Superseded by the manual-iteration pipeline (default):
#   submit_manual_pipeline.sh  -> jwst_gc_pipeline.photometry.cataloging
# Retired to 'legacy pipeline/'. Kept for reference only.
# ============================================================
echo "DEPRECATED: $(basename "$0") belongs to the legacy iter1-4 cataloging" >&2
echo "pipeline, superseded by submit_manual_pipeline.sh (manual-iteration path)." >&2
echo "This script has been retired and no longer runs. Recover from git if needed." >&2
exit 1
# submit_full_chain.sh -- iter1 -> iter2 -> merge -> iter3-launch chain
# for one (target, filter, module) combo.
#
# Iteration coverage of this script (vs README "Iter1 / Iter2 / Iter3 /
# Iter4 cataloging cycle" section):
#   iter1: yes (per-obs SLURM array, --each-exposure --daophot)
#   iter2: yes (per-obs SLURM array, --iteration-label=iter2
#              --postprocess-residuals)
#   merge: yes (one merge per target, gated on all iter2 arrays;
#              --indiv-merge-methods=daoiterative; --merge-singlefields)
#   iter3: yes (launched via run_iter3_cataloging.sh, gated on merge)
#   iter4: NO  (run run_iter4resbgrefit.sh after iter3 completes)
#   pipeline (Detector1/Image2/Image3): NO -- assumes
#          *_destreak_o<NNN>_crf.fits already exist on disk for every obs.
#          Use run_full_pipeline_<target>.sh to produce them.
#
# Multi-obs targets (cloudef = obs 002 + 005, gc2211 = 5 separate
# pointings, brick = proposal 2221 obs 001 + proposal 1182 obs 004)
# require launching submit_full_chain.sh once per (target, filter,
# module) and letting the per-target case block below populate the
# ``fields`` array.  Within this script, iter1 and iter2 submit one
# SLURM array per (filter, module, obs); merge runs once per target and
# is gated on the union of all those arrays.
#
# Usage:
#   submit_full_chain.sh <target> <filter> <module> [tag] [extra_dep]
#
# Examples:
#   submit_full_chain.sh sickle F470N nrcb V8
#   submit_full_chain.sh cloudef F480M merged V8
#   submit_full_chain.sh brick-1182 F115W merged V8
#   submit_full_chain.sh gc2211-028 F277W merged V8
#
# Per-target params mirror the case blocks in run_iter3_cataloging.sh
# (proposal_id, field/fields, each_suffix, ref_filter).

set -e

target="${1:?usage: submit_full_chain.sh <target> <filter> <module> [tag] [extra_dep]}"
filter="${2:?missing filter}"
module="${3:?missing module}"
tag="${4:-V8}"
extra_dep="${5:-}"

PYTHON=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
SCRIPT=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
MERGE_SCRIPT=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/merge_catalogs.py
ITER3_LAUNCHER=/blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/run_iter3_cataloging.sh

# python_target is what gets passed as --target= to the Python script;
# launcher targets like brick-1182 / gc2211-028 reuse the parent target's
# obs_filters / seed catalog by setting python_target=brick / gc2211.
python_target=""

case "$target" in
  sickle)
    basepath=/orange/adamginsburg/jwst/sickle ; proposal_id=3958 ; field=001
    fields=(007) ; ref_filter=f470n
    logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
    array_range="0-23" ;;
  brick)
    # proposal 2221, obs 001, narrowband filters.  Broadband filters live
    # under proposal 1182 obs 004 -- launch as ``brick-1182`` separately.
    basepath=/blue/adamginsburg/adamginsburg/jwst/brick ; proposal_id=2221
    field=001 ; fields=(001) ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/brick_logs
    array_range="0-23" ;;
  brick-1182)
    # proposal 1182, obs 004 (F115W/F200W/F356W/F444W broadband).
    # python_target=brick so obs_filters and seed catalog lookups still
    # work; outputs land in the same /jwst/brick basepath.
    basepath=/blue/adamginsburg/adamginsburg/jwst/brick ; proposal_id=1182
    field=004 ; fields=(004) ; ref_filter=f405n
    python_target=brick
    logdir=/blue/adamginsburg/adamginsburg/brick_logs
    array_range="0-23" ;;
  cloudc)
    basepath=/blue/adamginsburg/adamginsburg/jwst/cloudc ; proposal_id=2221
    field=002 ; fields=(002) ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/cloudc_logs
    array_range="0-23" ;;
  sgrb2)
    basepath=/orange/adamginsburg/jwst/sgrb2 ; proposal_id=5365
    field=001 ; fields=(001) ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
    case "$filter" in F187N) array_range="0-47" ;; *) array_range="0-23" ;; esac ;;
  sgra)
    basepath=/orange/adamginsburg/jwst/sgra ; proposal_id=1939
    field=001 ; fields=(001) ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/sgra_jwst/
    array_range="0-23" ;;
  cloudef)
    # Cloud E (obs 002) + Cloud F (obs 005) reduced together; merge
    # operates on the union.  Each obs has its own destreak_o<NNN>_crf
    # suffix so iter1/iter2 must run once per obs.
    basepath=/orange/adamginsburg/jwst/cloudef ; proposal_id=2092
    field=005 ; fields=(002 005) ; ref_filter=f210m
    logdir=/blue/adamginsburg/adamginsburg/logs/cloudef_jwst/
    array_range="0-23" ;;
  arches)
    basepath=/orange/adamginsburg/jwst/arches ; proposal_id=2045
    field=001 ; fields=(001) ; ref_filter=f212n
    logdir=/blue/adamginsburg/adamginsburg/logs/arches_jwst/
    array_range="0-23" ;;
  quintuplet)
    basepath=/orange/adamginsburg/jwst/quintuplet ; proposal_id=2045
    field=003 ; fields=(003) ; ref_filter=f212n
    logdir=/blue/adamginsburg/adamginsburg/logs/quintuplet_jwst/
    array_range="0-23" ;;
  sgrc)
    basepath=/orange/adamginsburg/jwst/sgrc ; proposal_id=4147
    field=012 ; fields=(012) ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/sgrc_jwst/
    array_range="0-23" ;;
  gc2211-023|gc2211-028|gc2211-046|gc2211-049|gc2211-050)
    # Proposal 2211 has 5 GC pointings; each obs is its own launcher
    # target.  python_target stays gc2211 so obs_filters / seed catalog
    # lookups still work.  Common filter across all 5: F277W.
    basepath=/orange/adamginsburg/jwst/gc2211 ; proposal_id=2211
    field="${target#gc2211-}" ; fields=("${field}")
    ref_filter=f277w
    python_target=gc2211
    logdir=/blue/adamginsburg/adamginsburg/logs/gc2211_jwst/
    array_range="0-23" ;;
  w51)
    # W51 (Yoo prop 6151, obs 001).  SW filters have 64 frames; LW have 16.
    # array_range="0-63" covers SW; extra tasks for LW skip gracefully.
    basepath=/orange/adamginsburg/jwst/w51 ; proposal_id=6151
    field=001 ; fields=(001) ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/w51_jwst/
    array_range="0-63" ;;
  wd1)
    # Westerlund 1 (Guarcello prop 1905, obs 001).  SW filters have 96 frames; LW have 24.
    # array_range="0-95" covers SW; extra tasks for LW skip gracefully.
    basepath=/orange/adamginsburg/jwst/wd1 ; proposal_id=1905
    field=001 ; fields=(001) ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/wd1_jwst/
    array_range="0-95" ;;
  *) echo "Unknown target: $target" >&2
     echo "Known targets: sickle brick brick-1182 cloudc sgrb2 sgra cloudef arches quintuplet sgrc gc2211-{023,028,046,049,050} w51 wd1" >&2
     exit 2 ;;
esac

python_target="${python_target:-${target}}"

mkdir -p "$logdir"
DEP=""
if [[ -n "$extra_dep" ]]; then DEP="--dependency=afterok:$extra_dep"; fi

# sgrb2 uses align_o001_crf for LW filters and destreak_o001_crf for SW.
# Compute each_suffix per obs (with the sgrb2 LW override applied).
get_each_suffix() {
  local fld="$1"
  if [[ "$target" == "sgrb2" ]]; then
    case "$filter" in
      F182M|F187N|F210M|F212N) echo "destreak_o${fld}_crf" ;;
      *)                        echo "align_o${fld}_crf" ;;
    esac
  elif [[ "$target" == "wd1" ]]; then
    case "$filter" in
      F150W) echo "o001_crf" ;;
      *)     echo "destreak_o${fld}_crf" ;;
    esac
  else
    echo "destreak_o${fld}_crf"
  fi
}

iter1_ids=()
iter2_ids=()

for fld in "${fields[@]}"; do
  each_suffix=$(get_each_suffix "$fld")
  COMMON="--filternames=${filter} --modules=${module} --proposal_id=${proposal_id} --field=${fld} --target=${python_target} --each-suffix=${each_suffix} --each-exposure --daophot --skip-crowdsource --skip-if-done"

  # No --array: manual-iteration pipeline is a single in-process job and
  # must NOT run under a SLURM array (SLURM_ARRAY_TASK_ID must be unset).
  ITER1=$(sbatch --parsable $DEP \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=1 --cpus-per-task=4 --mem=64gb --time=96:00:00 \
    --job-name=webb-cat-${target}-o${fld}-${filter}-${module}-iter1-${tag} \
    --output=${logdir}/webb-cat-${target}-o${fld}-${filter}-${module}-iter1-${tag}_%j.log \
    --wrap "export OMP_NUM_THREADS=1; ${PYTHON} ${SCRIPT} ${COMMON}")
  iter1_ids+=("${ITER1}")

  ITER2=$(sbatch --parsable --dependency=afterok:${ITER1} \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=1 --cpus-per-task=4 --mem=64gb --time=96:00:00 \
    --job-name=webb-cat-${target}-o${fld}-${filter}-${module}-iter2-${tag} \
    --output=${logdir}/webb-cat-${target}-o${fld}-${filter}-${module}-iter2-${tag}_%j.log \
    --wrap "export OMP_NUM_THREADS=1; ${PYTHON} ${SCRIPT} ${COMMON} --iteration-label=iter2 --postprocess-residuals")
  iter2_ids+=("${ITER2}")
done

merge_dep=$(IFS=:; echo "${iter2_ids[*]}")
MERGE=$(sbatch --parsable --dependency=afterok:${merge_dep} \
  --account=astronomy-dept --qos=astronomy-dept-b --mem=64gb --time=24:00:00 \
  --job-name=webb-merge-${target}-iter2-${filter}-${tag} \
  --output=${logdir}/webb-merge-${target}-iter2-${filter}-${tag}_%j.log \
  --wrap "${PYTHON} ${MERGE_SCRIPT} --merge-singlefields --modules=merged --indiv-merge-methods=daoiterative --skip-crowdsource --target=${python_target} --iteration-label=iter2 --ref-filter=${ref_filter}")

ITER3_LAUNCH=$(sbatch --parsable --dependency=afterok:${MERGE} \
  --account=astronomy-dept --qos=astronomy-dept-b --mem=4gb --time=00:30:00 \
  --job-name=webb-iter3-launch-${target}-${filter}-${tag} \
  --output=${logdir}/webb-iter3-launch-${target}-${filter}-${tag}_%j.log \
  --wrap "bash ${ITER3_LAUNCHER} --target ${target} --iter2-merge-dep ${MERGE} ${filter}")

iter1_csv=$(IFS=,; echo "${iter1_ids[*]}")
iter2_csv=$(IFS=,; echo "${iter2_ids[*]}")
echo "${target}/${filter}/${module}: iter1=[${iter1_csv}] iter2=[${iter2_csv}] merge=${MERGE} iter3-launch=${ITER3_LAUNCH}"
