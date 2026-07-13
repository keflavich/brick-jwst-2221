#!/usr/bin/env bash
# run_full_pipeline_brick.sh -- full BRICK reduction + cataloging chain on the
# NEW module-locked VIRAC2 astrometric reference (Offsets_JWST_Brick<prop>_VIRAC2locked.csv).
#
# Brick is two proposals reduced into the same tree (/orange/adamginsburg/jwst/brick):
#   1182 obs 004 : broadband  F115W F200W F356W F444W
#   2221 obs 001 : narrowband F182M F187N F212N F405N F410M F466N
#
# Per filter:
#   1) REDUCTION (Image3) with --skip_step1and2 : reuse up-to-date *_cal.fits
#      (Detector1/Image2 are alignment-independent), regenerate *_destreak.fits
#      fresh -> fix_alignment applies the per-VISIT module-locked offset
#      (tweakreg is skip:True so the lock is preserved) -> fresh *_destreak_o<F>_crf
#      + *_i2d on the new reference.
#   2) CATALOGING through m7 via submit_manual_pipeline.sh (manual-iteration
#      m12->m3->m4->m5->m6->m7), one multi-filter job per proposal (engages the
#      m7 cross-band seed), dependent on that proposal's reduction jobs.
#
# fix_alignment is idempotent on the RAOFFSET header, but the *_destreak.fits it
# writes to is regenerated fresh each run, so the new offsets ARE applied (the old
# RAOFFSET lived only on the previous destreak/crf, which Image3 overwrites).
#
# Usage:
#   run_full_pipeline_brick.sh            # submit everything
#   run_full_pipeline_brick.sh --dry-run  # print the sbatch commands only
#   TAG=V11 run_full_pipeline_brick.sh    # override cataloging tag (default V10)
#
set -euo pipefail

DRYRUN=0
[[ "${1:-}" == "--dry-run" ]] && DRYRUN=1

PYTHON=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
PIPELINE=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/PipelineRerunNIRCAM-LONG.py
CATLAUNCH=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/submit_manual_pipeline.sh
CRDS_PATH=${CRDS_PATH:-/orange/adamginsburg/jwst/crds}
LOGDIR=/blue/adamginsburg/adamginsburg/brick_logs
TAG=${TAG:-V10}
mkdir -p "${LOGDIR}"

# Cataloging resource caps.  The auto-sizer in submit_manual_pipeline.sh scales
# CPUS to the frame count (->128) and MEM=CPUS*6 (->768gb), which never schedules
# on astronomy-dept-b.  The 2026-04 streaming merge refactor dropped the merge
# peak to ~30 GB, so CPU was the only real driver: cap to a schedulable size.
# Override via env (CAT_CPUS / CAT_MEM).
export MANUAL_CPUS=${CAT_CPUS:-32}
export MANUAL_MEM=${CAT_MEM:-192gb}

# proposal : field : filters(csv) : cataloging-launcher-target
P1182="1182:004:F115W,F200W,F356W,F444W:brick-1182"
P2221="2221:001:F182M,F187N,F212N,F405N,F410M,F466N:brick"

submit_reduction() {
  # $1 proposal_id  $2 field  $3 filter
  local prop="$1" field="$2" filt="$3"
  local wrap="CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${PYTHON} ${PIPELINE} --proposal_id=${prop} --field=${field} --filternames=${filt} --modules=merged --skip_step1and2"
  if (( DRYRUN )); then
    echo "sbatch [reduction ${prop}/${field}/${filt}]: ${wrap}" >&2
    echo "REDU_${prop}_${filt}_DRYJOB"
    return
  fi
  sbatch --parsable \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 \
    --job-name="brick${prop}-o${field}-reduce-${filt}" \
    --output="${LOGDIR}/webb-pipe-brick-${prop}-${filt}_%j.log" \
    --wrap "${wrap}"
}

run_side() {
  # $1 = "prop:field:filters:cattarget"
  IFS=':' read -r prop field filters cattarget <<< "$1"
  echo "=== BRICK ${cattarget} (prop ${prop}, field ${field}): ${filters} ===" >&2
  local -a redu_ids=()
  local f
  for f in ${filters//,/ }; do
    jid=$(submit_reduction "${prop}" "${field}" "${f}")
    redu_ids+=("${jid}")
    echo "  reduction ${f}: ${jid}" >&2
  done
  local dep; dep=$(IFS=:; echo "${redu_ids[*]}")
  echo "  cataloging (m12..m7) depends on: ${dep}" >&2
  if (( DRYRUN )); then
    echo "  ${CATLAUNCH} ${cattarget} ${filters} merged ${TAG} ${dep}"
  else
    "${CATLAUNCH}" "${cattarget}" "${filters}" merged "${TAG}" "${dep}"
  fi
}

run_side "${P1182}"
run_side "${P2221}"
echo "ALL BRICK PIPELINE JOBS SUBMITTED (tag ${TAG})." >&2
