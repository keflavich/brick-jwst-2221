#!/usr/bin/env bash
# Submit a full iter1 -> iter2 -> merge -> iter3 chain for one
# (target, filter, module) combo using the skycoord-bug-fixed code
# (2026-06-02).  Builds on the V7 sickle pattern.
#
# Usage:
#   submit_full_chain.sh <target> <filter> <module> [tag]
#
# Examples:
#   submit_full_chain.sh sickle F470N nrcb V8
#   submit_full_chain.sh cloudef F480M merged V8
#
# Per-target params from run_iter3_cataloging.sh case blocks.

set -e

target="${1:?usage: submit_full_chain.sh <target> <filter> <module> [tag]}"
filter="${2:?missing filter}"
module="${3:?missing module}"
tag="${4:-V8}"

PYTHON=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
SCRIPT=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
MERGE_SCRIPT=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/merge_catalogs.py
ITER3_LAUNCHER=/blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/run_iter3_cataloging.sh

case "$target" in
  sickle)
    basepath=/orange/adamginsburg/jwst/sickle ; proposal_id=3958 ; field=007
    each_suffix=destreak_o007_crf ; ref_filter=f470n
    logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
    array_range="0-23" ;;
  brick)
    basepath=/blue/adamginsburg/adamginsburg/jwst/brick ; proposal_id=2221 ; field=001
    each_suffix=destreak_o001_crf ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/brick_logs
    array_range="0-23" ;;
  cloudc)
    basepath=/blue/adamginsburg/adamginsburg/jwst/cloudc ; proposal_id=2221 ; field=002
    each_suffix=destreak_o002_crf ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/cloudc_logs
    array_range="0-23" ;;
  sgrb2)
    basepath=/orange/adamginsburg/jwst/sgrb2 ; proposal_id=5365 ; field=001
    case "$filter" in
      F182M|F187N|F210M|F212N) each_suffix=destreak_o001_crf ;;
      *) each_suffix=align_o001_crf ;;
    esac
    ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
    case "$filter" in F187N) array_range="0-47" ;; *) array_range="0-23" ;; esac ;;
  sgra)
    basepath=/orange/adamginsburg/jwst/sgra ; proposal_id=1939 ; field=001
    each_suffix=destreak_o001_crf ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/sgra_jwst/
    array_range="0-23" ;;
  cloudef)
    basepath=/orange/adamginsburg/jwst/cloudef ; proposal_id=2092 ; field=005
    each_suffix=destreak_o005_crf ; ref_filter=f210m
    logdir=/blue/adamginsburg/adamginsburg/logs/cloudef_jwst/
    array_range="0-23" ;;
  arches)
    basepath=/orange/adamginsburg/jwst/arches ; proposal_id=2045 ; field=001
    each_suffix=destreak_o001_crf ; ref_filter=f212n
    logdir=/blue/adamginsburg/adamginsburg/logs/arches_jwst/
    array_range="0-23" ;;
  quintuplet)
    basepath=/orange/adamginsburg/jwst/quintuplet ; proposal_id=2045 ; field=003
    each_suffix=destreak_o003_crf ; ref_filter=f212n
    logdir=/blue/adamginsburg/adamginsburg/logs/quintuplet_jwst/
    array_range="0-23" ;;
  sgrc)
    basepath=/orange/adamginsburg/jwst/sgrc ; proposal_id=4147 ; field=012
    each_suffix=destreak_o012_crf ; ref_filter=f405n
    logdir=/blue/adamginsburg/adamginsburg/logs/sgrc_jwst/
    array_range="0-23" ;;
  *) echo "Unknown target: $target" >&2 ; exit 2 ;;
esac

mkdir -p "$logdir"
extra_dep="${5:-}"  # optional dep on prior target's job
DEP=""
if [[ -n "$extra_dep" ]]; then DEP="--dependency=afterok:$extra_dep"; fi

COMMON="--filternames=${filter} --modules=${module} --proposal_id=${proposal_id} --field=${field} --target=${target} --each-suffix=${each_suffix} --each-exposure --daophot --skip-crowdsource --bundle-size=1 --skip-if-done"

ITER1=$(sbatch --parsable $DEP --array=${array_range} \
  --account=astronomy-dept --qos=astronomy-dept-b \
  --ntasks=1 --cpus-per-task=4 --mem=64gb --time=24:00:00 \
  --job-name=webb-cat-${target}-${filter}-${module}-iter1-${tag} \
  --output=${logdir}/webb-cat-${target}-${filter}-${module}-iter1-${tag}_%j-%A_%a.log \
  --wrap "export OMP_NUM_THREADS=1; ${PYTHON} ${SCRIPT} ${COMMON}")

ITER2=$(sbatch --parsable --dependency=afterok:${ITER1} --array=${array_range} \
  --account=astronomy-dept --qos=astronomy-dept-b \
  --ntasks=1 --cpus-per-task=4 --mem=64gb --time=24:00:00 \
  --job-name=webb-cat-${target}-${filter}-${module}-iter2-${tag} \
  --output=${logdir}/webb-cat-${target}-${filter}-${module}-iter2-${tag}_%j-%A_%a.log \
  --wrap "export OMP_NUM_THREADS=1; ${PYTHON} ${SCRIPT} ${COMMON} --iteration-label=iter2 --postprocess-residuals")

MERGE=$(sbatch --parsable --dependency=afterok:${ITER2} \
  --account=astronomy-dept --qos=astronomy-dept-b --mem=64gb --time=24:00:00 \
  --job-name=webb-merge-${target}-iter2-${filter}-${tag} \
  --output=${logdir}/webb-merge-${target}-iter2-${filter}-${tag}_%j.log \
  --wrap "${PYTHON} ${MERGE_SCRIPT} --merge-singlefields --modules=merged --indiv-merge-methods=daoiterative --skip-crowdsource --target=${target} --iteration-label=iter2 --ref-filter=${ref_filter}")

ITER3_LAUNCH=$(sbatch --parsable --dependency=afterok:${MERGE} \
  --account=astronomy-dept --qos=astronomy-dept-b --mem=4gb --time=00:30:00 \
  --job-name=webb-iter3-launch-${target}-${filter}-${tag} \
  --output=${logdir}/webb-iter3-launch-${target}-${filter}-${tag}_%j.log \
  --wrap "bash ${ITER3_LAUNCHER} --target ${target} --iter2-merge-dep ${MERGE} ${filter}")

echo "${target}/${filter}/${module}: iter1=${ITER1} iter2=${ITER2} merge=${MERGE} iter3-launch=${ITER3_LAUNCH}"
