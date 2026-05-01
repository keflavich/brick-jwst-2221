#!/usr/bin/env bash
# Submit remaining sgrb2 filters not covered by the first run, plus iter1/iter2 merges
# gated on ALL plain iter1/iter2 job IDs (old + new).
#
# Remaining work:
#   F187N: nrca4 iter2 plain + nrca4 bgsub pair + nrcb1-4 all pairs + mosaics
#   F210M, F212N (SW, 24 files/mod): all 8 modules plain+bgsub pairs + mosaics
#   F300M F360M F405N F410M F466N F480M (LW, 24 files/mod): all 2 modules pairs + mosaics
#   iter1 merge + iter2 merge gated on ALL plain IDs (old first-run + new)
#
# Old first-run plain job IDs (2026-04-23):
#   F150W iter1_plain: 30796358 30796362 30796366 30796370 30796374 30796378 30796382 30796386
#   F150W iter2_plain: 30796359 30796363 30796367 30796371 30796375 30796379 30796383 30796387
#   F182M iter1_plain: 30796394 30796398 30796402 30796406 30796410 30796414 30796418 30796422
#   F182M iter2_plain: 30796395 30796399 30796403 30796407 30796411 30796415 30796419 30796423
#   F187N nrca1-4 iter1_plain: 30796430 30796434 30796438 30796442
#   F187N nrca1-3 iter2_plain: 30796431 30796435 30796439
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
mkdir -p "${logdir}"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
proposal_id=5365; target=sgrb2; field=001
basepath=/orange/adamginsburg/jwst/sgrb2
short_mem=48gb; long_mem=32gb

sw_mods=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
lw_mods=(nrcalong nrcblong)

# Accumulate all plain iter1/iter2 IDs across ALL runs so far.
# Run 1 (2026-04-23 ~15:26): F150W, F182M, F187N nrca1-4 iter1 plain, F187N nrca1-3 iter2 plain
# Run 2 (2026-04-23 ~15:55): F187N nrca4 iter2 plain (30798631),
#                             F187N nrcb1 iter1 plain (30798644), iter2 plain (30798645)
# Run 3 (2026-04-23 ~16:30): F187N nrcb2-4 iter1/iter2 plain
# Run 4 (2026-04-23 ~16:40): F210M nrca1-4, nrcb1-3 iter1/iter2 plain (nrcb4 missing, nrcb3 bgsub iter2 missing)
all_iter1_plain=(
    30796358 30796362 30796366 30796370 30796374 30796378 30796382 30796386
    30796394 30796398 30796402 30796406 30796410 30796414 30796418 30796422
    30796430 30796434 30796438 30796442
    30798644  # F187N nrcb1 iter1 plain
    30807736 30807740 30807744  # F187N nrcb2-4 iter1 plain
    30808445 30808449 30808453 30808457 30808461 30808465 30808469  # F210M nrca1-4,nrcb1-3 iter1 plain
)
all_iter2_plain=(
    30796359 30796363 30796367 30796371 30796375 30796379 30796383 30796387
    30796395 30796399 30796403 30796407 30796411 30796415 30796419 30796423
    30796431 30796435 30796439
    30798631  # F187N nrca4 iter2 plain
    30798645  # F187N nrcb1 iter2 plain
    30807737 30807741 30807745  # F187N nrcb2-4 iter2 plain
    30808446 30808450 30808454 30808458 30808462 30808466 30808470  # F210M nrca1-4,nrcb1-3 iter2 plain
)

# ---------- helpers ----------

# Submit iter1+iter2 pair (plain or bgsub); adds plain IDs to global arrays.
# Args: filter module mem each_suffix array_range bgsub_opt
submit_pair() {
    local filter="$1" module="$2" mem="$3" each_suffix="$4" array_range="$5" bgsub_opt="${6:-}"
    local bgsub_tag="${bgsub_opt:+-bgsub}"
    local dao_args="--daophot --skip-crowdsource${bgsub_opt:+ ${bgsub_opt}}"

    local i1
    i1=$(sbatch --parsable \
        --array="${array_range}" \
        --job-name="webb-cat-${target}-${filter}-${module}${bgsub_tag}-eachexp" \
        --output="${logdir}/webb-cat-${target}-${filter}-${module}${bgsub_tag}-eachexp_%j-%A_%a.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=2 --nodes=1 --mem="${mem}" --time=96:00:00 \
        --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} \
--each-exposure --proposal_id=${proposal_id} --target=${target} \
--each-suffix=${each_suffix} ${dao_args}")
    echo "  Submitted iter1 ${i1}  ${target} ${filter} ${module} ${bgsub_opt:-plain}" >&2

    local i2
    i2=$(sbatch --parsable \
        --dependency="afterok:${i1}" \
        --array="${array_range}" \
        --job-name="webb-cat-${target}-iter2-${filter}-${module}${bgsub_tag}-eachexp" \
        --output="${logdir}/webb-cat-${target}-iter2-${filter}-${module}${bgsub_tag}-eachexp_%j-%A_%a.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=2 --nodes=1 --mem="${mem}" --time=96:00:00 \
        --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} \
--each-exposure --proposal_id=${proposal_id} --target=${target} \
--each-suffix=${each_suffix} ${dao_args} \
--iteration-label=iter2 --postprocess-residuals")
    echo "  Submitted iter2 ${i2}  ${target} ${filter} ${module} ${bgsub_opt:-plain}" >&2

    if [[ -z "${bgsub_opt}" ]]; then
        all_iter1_plain+=("${i1}")
        all_iter2_plain+=("${i2}")
    fi
}

# Submit mosaics for iter1 and iter2 residuals.
# Args: filter agg_mod iter1_dep_colon iter2_dep_colon
submit_mosaics() {
    local filter="$1" agg_mod="$2" i1dep="$3" i2dep="$4"
    for iter_label in "" "iter2"; do
        local iter_tag="${iter_label:+-${iter_label}}"
        local dep="${i1dep}"; [[ -n "${iter_label}" ]] && dep="${i2dep}"
        local iter_py="None"; [[ -n "${iter_label}" ]] && iter_py="'${iter_label}'"
        sbatch --dependency="afterok:${dep}" \
            --job-name="webb-mosaic-${target}-${filter}-${agg_mod}${iter_tag}" \
            --output="${logdir}/webb-mosaic-${target}-${filter}-${agg_mod}${iter_tag}_%j.log" \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 \
            --wrap "${python_exec} -c \"import sys; sys.path.insert(0,'${analysis_dir}'); \
import crowdsource_catalogs_long as c; \
[c.mosaic_each_exposure_residuals(basepath='${basepath}', \
  filtername='${filter}', proposal_id='${proposal_id}', field='${field}', \
  module='${agg_mod}', residual_kind=kind, desat=False, bgsub=False, \
  epsf=False, blur=False, group=False, pupil='clear', \
  iteration_label=${iter_py}) \
 for kind in ('basic','iterative')]\"" > /dev/null
        echo "  Submitted mosaic ${filter} ${agg_mod}${iter_tag}" >&2
    done
}

# ---------- F187N: FULLY DONE (submitted 2026-04-23, mosaics submitted separately) ----------
# nrca1-4 + nrcb1-4 iter1/iter2 plain and bgsub all submitted.
# Mosaics submitted with IDs 30808203-30808206.

# ---------- F210M: nrcb4 only (nrca1-4 + nrcb1-3 already done in run 4) ----------
# Also need nrcb3 iter2 bgsub (gated on 30808471, but bgsub not critical for merge)
echo "=== F210M nrcb4 ===" >&2
filter=F210M; each_suffix=destreak_o001_crf; array_range="0-23"; mem=${short_mem}
# Collect already-submitted iter1/iter2 plain IDs for F210M
f210m_i1_ids=(30808445 30808449 30808453 30808457 30808461 30808465 30808469)
f210m_i2_ids=(30808446 30808450 30808454 30808458 30808462 30808466 30808470)
submit_pair "${filter}" nrcb4 "${mem}" "${each_suffix}" "${array_range}" ""
f210m_i1_ids+=("${all_iter1_plain[-1]}")
f210m_i2_ids+=("${all_iter2_plain[-1]}")
submit_pair "${filter}" nrcb4 "${mem}" "${each_suffix}" "${array_range}" "--bgsub"
# F210M mosaics (gated on all 8 modules' plain iter1/iter2)
f210m_i1dep=$(IFS=:; echo "${f210m_i1_ids[*]}")
f210m_i2dep=$(IFS=:; echo "${f210m_i2_ids[*]}")
submit_mosaics "${filter}" nrca "${f210m_i1dep}" "${f210m_i2dep}"
submit_mosaics "${filter}" nrcb "${f210m_i1dep}" "${f210m_i2dep}"

# ---------- F212N ----------
echo "=== F212N ===" >&2
filter=F212N; each_suffix=destreak_o001_crf; array_range="0-23"; mem=${short_mem}
sw_i1_ids=(); sw_i2_ids=()
for module in "${sw_mods[@]}"; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" ""
    sw_i1_ids+=("${all_iter1_plain[-1]}")
    sw_i2_ids+=("${all_iter2_plain[-1]}")
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" "--bgsub"
done
sw_i1dep=$(IFS=:; echo "${sw_i1_ids[*]}")
sw_i2dep=$(IFS=:; echo "${sw_i2_ids[*]}")
submit_mosaics "${filter}" nrca "${sw_i1dep}" "${sw_i2dep}"
submit_mosaics "${filter}" nrcb "${sw_i1dep}" "${sw_i2dep}"

# ---------- LW filters ----------
for filter in F300M F360M F405N F410M F466N F480M; do
    echo "=== ${filter} ===" >&2
    each_suffix=align_o001_crf; array_range="0-23"; mem=${long_mem}
    lw_i1_ids=(); lw_i2_ids=()
    for module in "${lw_mods[@]}"; do
        submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" ""
        lw_i1_ids+=("${all_iter1_plain[-1]}")
        lw_i2_ids+=("${all_iter2_plain[-1]}")
        submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" "--bgsub"
    done
    lw_i1dep=$(IFS=:; echo "${lw_i1_ids[*]}")
    lw_i2dep=$(IFS=:; echo "${lw_i2_ids[*]}")
    for agg_mod in "${lw_mods[@]}"; do
        submit_mosaics "${filter}" "${agg_mod}" "${lw_i1dep}" "${lw_i2dep}"
    done
done

# ---------- Merges gated on ALL plain iter1/iter2 across both runs ----------
iter1_merge_dep=$(IFS=:; echo "${all_iter1_plain[*]}")
iter2_merge_dep=$(IFS=:; echo "${all_iter2_plain[*]}")

iter1_merge_id=$(sbatch --parsable \
    --dependency="afterok:${iter1_merge_dep}" \
    --job-name="webb-cat-merge-${target}-iter1" \
    --output="${logdir}/webb-cat-merge-${target}-iter1_%j.log" \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
    --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py \
--merge-singlefields --modules=merged --indiv-merge-methods=dao \
--skip-crowdsource --target=${target}")
echo "Submitted iter1 merge ${iter1_merge_id}"

iter2_merge_id=$(sbatch --parsable \
    --dependency="afterok:${iter2_merge_dep}" \
    --job-name="webb-cat-merge-${target}-iter2" \
    --output="${logdir}/webb-cat-merge-${target}-iter2_%j.log" \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
    --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py \
--merge-singlefields --modules=merged --indiv-merge-methods=daoiterative \
--skip-crowdsource --target=${target} --iteration-label=iter2")
echo "Submitted iter2 merge ${iter2_merge_id}"

echo "DONE submitting continuation for ${target}."
echo "Run iter3 after merge completes:"
echo "  bash run_iter3_cataloging.sh --target=sgrb2 --iter2-merge-dep ${iter2_merge_id}"
