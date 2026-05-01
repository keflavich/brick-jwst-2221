#!/usr/bin/env bash
# Continuation 2 for sgrb2: F150W bgsub recovery + F212N + LW filters + merges.
# Run this after the recovery script jobs have started clearing (~45 min after recovery).
#
# Hardcoded accumulated plain iter2 IDs (for merge dependency):
#   F150W: nrca2=30811720, nrca3=30811309, nrca4=30811311, nrcb1-4=30811313-30811319
#   F182M: 30811293,30811295,30811297,30811299,30811301,30811303,30811305,30811307
#   F187N nrca1-4: 30811321,30811323,30811325,30811327
#   F187N nrcb1: 30798645, nrcb2: 30807737, nrcb3: 30811329, nrcb4: 30807745
#   F210M: 30808446,30808450,30808454,30808458,30808462,30808466,30808470,30811350
#
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
mkdir -p "${logdir}"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
proposal_id=5365; target=sgrb2; field=001
basepath=/orange/adamginsburg/jwst/sgrb2
sw_mem=192gb
long_mem=32gb

sw_mods=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
lw_mods=(nrcalong nrcblong)

# All accumulated plain iter1/iter2 IDs (old completed + recovery new).
# Old F150W nrca1 iter1/iter2 (30796358/30796359): likely purged, omit from dependency.
# F210M: all modules resubmitted at 96gb (OOM at 48gb across all 8 modules).
# F187N nrcb2,nrcb4: resubmitted at 96gb.
# F182M nrca1-4,nrcb1-2,nrcb4: resubmitted at 192gb (OOM at 96gb, 24000+ sources).
# F150W nrca3-4,nrcb2-3: resubmitted at 192gb (OOM at 96gb).
all_iter1_plain=(
    30819085  # F150W nrca2 iter1 (192gb)
    30816301 30816303 30818296 30816305 30816307 30821873  # F150W nrca3-4,nrcb1-4 (all 192gb)
    30816287 30816289 30816291 30816293 30816295 30816297 30816519 30816299  # F182M nrca1-4,nrcb1-4 (all 192gb)
    30818286 30818288 30818290 30818292 30818294  # F187N nrca1-4,nrcb3 (192gb)
    30798644  # F187N nrcb1 (completed)
    30819087  # F187N nrcb2 iter1 (192gb)
    30819089  # F187N nrcb4 iter1 (192gb)
    30819091 30819093 30819095 30819097  # F210M nrca1-4 (192gb)
    30819099 30819101 30819103 30819105  # F210M nrcb1-4 (192gb)
)
all_iter2_plain=(
    30819086  # F150W nrca2 iter2 (192gb)
    30816302 30816304 30818297 30816306 30816308 30821874  # F150W nrca3-4,nrcb1-4 (all 192gb)
    30816288 30816290 30816292 30816294 30816296 30816298 30816520 30816300  # F182M nrca1-4,nrcb1-4 (all 192gb)
    30818287 30818289 30818291 30818293 30818295  # F187N nrca1-4,nrcb3 (192gb)
    30798645  # F187N nrcb1 iter2
    30819088  # F187N nrcb2 iter2 (192gb)
    30819090  # F187N nrcb4 iter2 (192gb)
    30819092 30819094 30819096 30819098  # F210M nrca1-4 iter2 (192gb)
    30819100 30819102 30819104 30819106  # F210M nrcb1-4 iter2 (192gb)
)

# ---------- helpers ----------

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

# ---------- F150W bgsub recovery (96gb, 7 modules; KeyError bug fixed) ----------
echo "=== F150W bgsub recovery (96gb) ===" >&2
filter=F150W; each_suffix=align_o001_crf; array_range="0-23"; mem=${sw_mem}
f150w_bgsub_i1_ids=(); f150w_bgsub_i2_ids=()
for module in nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" "--bgsub"
done

# ---------- F212N ----------
echo "=== F212N ===" >&2
filter=F212N; each_suffix=destreak_o001_crf; array_range="0-23"; mem=${sw_mem}
f212n_i1_ids=(); f212n_i2_ids=()
for module in "${sw_mods[@]}"; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" ""
    f212n_i1_ids+=("${all_iter1_plain[-1]}")
    f212n_i2_ids+=("${all_iter2_plain[-1]}")
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" "--bgsub"
done
f212n_i1dep=$(IFS=:; echo "${f212n_i1_ids[*]}")
f212n_i2dep=$(IFS=:; echo "${f212n_i2_ids[*]}")
submit_mosaics "${filter}" nrca "${f212n_i1dep}" "${f212n_i2dep}"
submit_mosaics "${filter}" nrcb "${f212n_i1dep}" "${f212n_i2dep}"

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

# ---------- Merges (gated on ALL accumulated plain iter1/iter2 IDs) ----------
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

echo "DONE submitting continuation2 for ${target}."
echo "Run iter3 after merge completes:"
echo "  bash run_iter3_cataloging.sh --target=sgrb2 --iter2-merge-dep ${iter2_merge_id}"
