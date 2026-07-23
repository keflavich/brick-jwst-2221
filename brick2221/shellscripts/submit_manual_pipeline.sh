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
# Module token rules (see PHOTOMETRY_PIPELINE.md):
#   - A single token "nrcb" auto-expands to "nrcb1..nrcb4" for SW filters
#     and "nrcblong" for LW filters; same for "nrca".
#   - For mixed SW/LW multi-filter runs, this script forbids the call only
#     for sgrb2 (LW uses align_o<NNN>_crf, SW uses destreak_o<NNN>_crf).
#     Other targets use the same suffix family so SW/LW in one call works.
#
# Resource env-var overrides (defaults: ~1 CPU/frame up to 128 / 6 GB/CPU / 96h):
#   MANUAL_CPUS=64 MANUAL_MEM=512gb MANUAL_TIME=72:00:00 \
#     submit_manual_pipeline.sh <target> <filter> <module>
#   MANUAL_CPU_CAP (default 128), MANUAL_FALLBACK_CPUS (default 32, used only
#   when no crf frames are on disk at submit time), MANUAL_MEM_PER_CPU_GB (6).
#
# SPLIT_FILTERS=1 (opt-in, multi-filter only): submit one single-filter job per
# filter (m12..m6, concurrent, each self-sized) + one afterok m7 cross-band
# finalize, instead of one monolithic node running all filters serially.  Cuts
# wall clock ~Nx on everything but m7.  EXPERIMENTAL -- validate on one field.
#   SPLIT_FILTERS=1 submit_manual_pipeline.sh brick-1182 F115W,F200W,F356W,F444W merged
#
# ----- Per-target examples -----
#
# IMPORTANT: the <filter> argument is passed verbatim to --filternames.
# Pass a SINGLE filter for a single-filter run (no m7 cross-band seed)
# or a COMMA-SEPARATED list to engage the m7 cross-band seed across all
# given filters.  The canonical full-coverage call for each target is
# shown FIRST below; minimal single-filter calls are also shown for
# quick smoke-tests.
#
# brick (2221 narrowband, 6 filters):
#   submit_manual_pipeline.sh brick F182M,F187N,F212N,F405N,F410M,F466N merged
#   # single-filter smoke test:
#   submit_manual_pipeline.sh brick F405N merged
#
# brick-1182 (1182 broadband, 4 filters; outputs under /jwst/brick/):
#   submit_manual_pipeline.sh brick-1182 F115W,F200W,F356W,F444W merged
#   # single-filter:
#   submit_manual_pipeline.sh brick-1182 F115W merged
#
# cloudc (2221 obs 002, 6 filters):
#   submit_manual_pipeline.sh cloudc F182M,F187N,F212N,F405N,F410M,F466N merged
#   submit_manual_pipeline.sh cloudc F410M merged
#
# sickle (3958 obs 007, 5 filters):
#   submit_manual_pipeline.sh sickle F187N,F210M,F335M,F470N,F480M nrcb
#   submit_manual_pipeline.sh sickle F470N nrcb
#
# sgrb2 (SPECIAL: LW uses align_o001_crf, SW uses destreak_o001_crf).
#   Multi-filter calls that mix SW+LW families are REJECTED; split into
#   two calls.  Per-family canonical full coverage:
#     SW: submit_manual_pipeline.sh sgrb2 F150W,F182M,F187N,F210M,F212N nrcb
#     LW: submit_manual_pipeline.sh sgrb2 F300M,F360M,F405N,F410M,F466N,F480M nrcb
#   # single-filter:
#     submit_manual_pipeline.sh sgrb2 F212N nrcb         # SW
#     submit_manual_pipeline.sh sgrb2 F360M nrcb         # LW
#
# sgra (1939, 3 filters):
#   submit_manual_pipeline.sh sgra F115W,F212N,F405N merged
#   submit_manual_pipeline.sh sgra F212N merged
#
# arches (2045 obs 001, 2 filters):
#   submit_manual_pipeline.sh arches F212N,F323N merged
#   # single-filter (NO m7 seed):
#   submit_manual_pipeline.sh arches F212N merged
#
# quintuplet (2045 obs 003, 2 filters):
#   submit_manual_pipeline.sh quintuplet F212N,F323N merged
#   submit_manual_pipeline.sh quintuplet F212N merged
#
# sgrc (4147 obs 012, 8 filters):
#   submit_manual_pipeline.sh sgrc F115W,F162M,F182M,F212N,F360M,F405N,F470N,F480M merged
#   submit_manual_pipeline.sh sgrc F212N merged
#
# cloudef (SPECIAL: TWO obs 002 + 005 reduced together).
#   This script auto-loops fields=(002 005) and submits ONE sbatch PER
#   obs.  Each obs gets an independent run_manual_pipeline; the per-obs
#   outputs live under /orange/adamginsburg/jwst/cloudef/.  Final
#   cross-obs catalog requires a separate cross-obs merge step (TODO);
#   for now, run downstream union via merge_catalogs.py manually.
#   submit_manual_pipeline.sh cloudef F210M nrcb
#   submit_manual_pipeline.sh cloudef F162M,F210M,F360M,F480M nrcb
#
# gc2211 (SPECIAL: 5 separate GC pointings = 5 launcher targets).
#   Each obs id is its own launcher target.  Filter sets differ per obs:
#     obs 028: F150W,F277W;  obs 023/046/049/050: F200W,F277W.
#   Submit one call per obs:
#     submit_manual_pipeline.sh gc2211-023 F200W,F277W merged
#     submit_manual_pipeline.sh gc2211-028 F150W,F277W merged
#     submit_manual_pipeline.sh gc2211-046 F200W,F277W merged
#     submit_manual_pipeline.sh gc2211-049 F200W,F277W merged
#     submit_manual_pipeline.sh gc2211-050 F200W,F277W merged
#   Or loop over all 5 (per-obs filter set varies; do them individually
#   for clarity).
#
# ----- Legacy iter1->iter2->merge->iter3 chain -----
# submit_full_chain.sh is preserved for the in-flight brick rerun.  Once
# that completes, submit_full_chain.sh will default to this manual path
# too; this companion runner is the migration target.

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
    # 3958 obs 007 = sickle NIRCam.  3958 MIRI obs 001/002 are also sickle, but
    # obs 003 is the BRICK MIRI field (not the sickle): catalog it with
    # target=brick (proposal_id 3958, field 003) so it lands in brick/ and does
    # not clash with sickle/ products.  The basepath split is handled by
    # field_to_reg_mapping in crowdsource_catalogs_long.py (3958/003 -> brick).
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
  wd1)
    basepath=/orange/adamginsburg/jwst/wd1 ; proposal_id=1905
    fields=(001)
    logdir=/blue/adamginsburg/adamginsburg/logs/wd1_jwst/ ;;
  wd2)
    basepath=/orange/adamginsburg/jwst/wd2 ; proposal_id=3523
    fields=(005)
    logdir=/blue/adamginsburg/adamginsburg/logs/wd2_jwst/ ;;
  w51)
    basepath=/orange/adamginsburg/jwst/w51 ; proposal_id=6151
    fields=(001)
    logdir=/blue/adamginsburg/adamginsburg/logs/w51_jwst/ ;;
  *) echo "Unknown target: $target" >&2 ; exit 2 ;;
esac

python_target="${python_target:-${target}}"
mkdir -p "$logdir"

DEP=""
if [[ -n "$extra_dep" ]]; then DEP="--dependency=afterok:$extra_dep"; fi

# sgrb2 uses align_o<NNN>_crf for LW filters, destreak_o<NNN>_crf for SW.
# w51/wd2/sickle are extended-emission fields: PipelineRerunNIRCAM-LONG.py
# forces do_destreak=False for these, copying _cal.fits -> _align.fits and
# running TweakReg+Image3 on that, producing align_o<NNN>_crf.fits instead
# of destreak_o<NNN>_crf.fits (see EXTENDED_EMISSION_FIELDS in that script).
get_each_suffix() {
  local fld="$1" filt="$2"
  if [[ "$target" == "sgrb2" ]]; then
    case "$filt" in
      F150W|F182M|F187N|F210M|F212N) echo "destreak_o${fld}_crf" ;;
      *)                              echo "align_o${fld}_crf" ;;
    esac
  elif [[ "$target" == "w51" || "$target" == "wd2" || "$target" == "sickle" ]]; then
    echo "align_o${fld}_crf"
  else
    echo "destreak_o${fld}_crf"
  fi
}

# crf frame count on disk for one (field, filter).
_count_frames() {
  local fld="$1" filt="$2" sfx
  sfx=$(get_each_suffix "$fld" "$filt")
  (shopt -s nullglob; a=( "${basepath}/${filt^^}/pipeline/"*"${sfx}.fits" ); echo ${#a[@]})
}

# CPUS for a frame count: MANUAL_CPUS override > ~1-CPU-per-frame (capped) >
# fallback.  ``--parallel-workers=CPUS`` fits this many frames at once, so a
# too-low value serializes the per-frame fits into multiple waves.
_cpus_for_frames() {
  local nframes="$1"
  if [[ -n "${MANUAL_CPUS:-}" ]]; then echo "$MANUAL_CPUS"; return; fi
  if (( nframes > 0 )); then
    (( nframes < CPU_CAP )) && echo "$nframes" || echo "$CPU_CAP"
  else
    echo "${MANUAL_FALLBACK_CPUS:-32}"
  fi
}

# Per-target resource defaults; override via env vars.
# Phases are sequential, but per-frame fits within a phase parallelize via
# --parallel-workers (one worker per detector-exposure).  To get the full
# benefit we allocate ~1 CPU per frame: count the crf inputs for the HEAVIEST
# filter in this run and size CPUS to that, capped at a single node
# (MANUAL_CPU_CAP, default 128 = largest hpg-default node, since the in-process
# ProcessPoolExecutor cannot span nodes).  Runs with <=cap frames then fit every
# frame in one parallel batch.
CPU_CAP=${MANUAL_CPU_CAP:-128}
_max_frames=0
for _f in ${filter//,/ }; do
  _n=$(_count_frames "${fields[0]}" "$_f")
  (( _n > _max_frames )) && _max_frames=$_n
done
CPUS=$(_cpus_for_frames "$_max_frames")
(( CPUS < 1 )) && CPUS=1
# Frame-based autosizing silently fell back (no crf on disk at submit time --
# e.g. cataloging queued before/independently of the reduction).  A 32-wide
# job runs the 192-frame SW filters in ~6 serial waves per phase; warn loudly
# so it is not mistaken for a full-width run.
if [[ -z "${MANUAL_CPUS:-}" ]] && (( _max_frames == 0 )); then
  echo "WARNING: no crf frames matched ${basepath}/<FILTER>/pipeline/*<suffix>.fits" >&2
  echo "         at submit time -> frame-based CPU autosizing FAILED, using CPUS=${CPUS}." >&2
  echo "         SW (192-frame) filters will then run ${CPUS}-wide waves, not one batch." >&2
  echo "         Fix: set MANUAL_CPUS=128, or resubmit once the crf files exist." >&2
fi
# ~6 GB/worker (a SW 75k-source fit holds the 2048^2 frame + fit state); capped
# by node RAM via Slurm.  Override with MANUAL_MEM.
MEM=${MANUAL_MEM:-$(( CPUS * ${MANUAL_MEM_PER_CPU_GB:-6} ))gb}
WALLTIME=${MANUAL_TIME:-96:00:00}
PARTITION=${MANUAL_PARTITION:-hpg-default}
echo "resources: ${_max_frames} frames (heaviest filter) -> CPUS=${CPUS} (cap ${CPU_CAP}) MEM=${MEM} partition=${PARTITION}" >&2

# Submit one manual-pipeline sbatch.  Used by the SPLIT_FILTERS path (below);
# the default single-job path keeps its own inline sbatch call unchanged.
#   args: fld filters_csv cpus mem depflag extra_flags jobsuffix
_submit_manual() {
  local fld="$1" filters="$2" cpus="$3" mem="$4" depflag="$5" extra="$6" jsuf="$7"
  local es; es=$(get_each_suffix "$fld" "${filters%%,*}")
  sbatch --parsable $depflag \
    --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
    --partition=${PARTITION} \
    --ntasks=1 --cpus-per-task=${cpus} --nodes=1 \
    --mem=${mem} --time=${WALLTIME} \
    --job-name=webb-manual-${target}-o${fld}-${filters//,/+}-${module}-${tag}${jsuf} \
    --output=${logdir}/webb-manual-${target}-o${fld}-${filters//,/+}-${module}-${tag}${jsuf}_%j.log \
    --wrap "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \
      ${PYTHON} ${SCRIPT} \
        --filternames=${filters} --modules=${module} \
        --proposal_id=${proposal_id} --field=${fld} \
        --target=${python_target} --each-suffix=${es} \
        --each-exposure --daophot --skip-crowdsource --skip-if-done \
        --max-group-size=${MANUAL_MAX_GROUP_SIZE:-10} \
        --parallel-workers=${cpus} ${extra}"
}

submitted=()
for fld in "${fields[@]}"; do
  # SPLIT_FILTERS=1 (opt-in, multi-filter only): instead of one monolithic job
  # that runs all filters x all phases serially on ONE node, submit one
  # single-filter job per filter (each runs m12..m6 -- a single-filter run has
  # no cross-filter m7 phase, so it naturally stops at m6 -- sized to ITS OWN
  # frame count) CONCURRENTLY, then ONE multifilter --manual-start-phase=m7
  # finalize that depends afterok on all of them and reuses their m12..m6
  # products.  The per-filter jobs' drizzles/renders overlap across nodes
  # instead of running one-after-another, cutting wall clock ~Nx on everything
  # but the (inherently cross-filter) m7 step.  EXPERIMENTAL: validate the m7
  # finalize reuses the per-filter products on one field before trusting it.
  if [[ "${SPLIT_FILTERS:-0}" == "1" && "${filter}" == *,* ]]; then
    echo "SPLIT_FILTERS=1: per-filter jobs + afterok m7 finalize for o${fld}" >&2
    pf_jobs=()
    for _f in ${filter//,/ }; do
      _nf=$(_count_frames "$fld" "$_f")
      _pc=$(_cpus_for_frames "$_nf"); (( _pc < 1 )) && _pc=1
      _pm=${MANUAL_MEM:-$(( _pc * ${MANUAL_MEM_PER_CPU_GB:-6} ))gb}
      _j=$(_submit_manual "$fld" "$_f" "$_pc" "$_pm" "$DEP" "--manual-stop-after-phase=m6" "-pf")
      pf_jobs+=("$_j")
      echo "  per-filter ${_f}: ${_nf} frames -> CPUS=${_pc} MEM=${_pm} job=${_j}" >&2
    done
    _dep="--dependency=afterok:$(IFS=:; echo "${pf_jobs[*]}")"
    _j7=$(_submit_manual "$fld" "$filter" "$CPUS" "$MEM" "$_dep" "--manual-start-phase=m7" "-m7")
    submitted+=("${pf_jobs[@]}" "$_j7")
    echo "submitted split manual ${target}/o${fld}/${filter}/${module}: per-filter=[${pf_jobs[*]}] m7-finalize=${_j7}"
    continue
  fi

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
    --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
    --partition=${PARTITION} \
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
        --max-group-size=${MANUAL_MAX_GROUP_SIZE:-10} \
        --parallel-workers=${CPUS}")
  submitted+=("${JOB}")
  echo "submitted manual-iter ${target}/o${fld}/${filter}/${module} = ${JOB}"
done

csv=$(IFS=,; echo "${submitted[*]}")
echo "${target}/${filter}/${module}: jobs=[${csv}]"
