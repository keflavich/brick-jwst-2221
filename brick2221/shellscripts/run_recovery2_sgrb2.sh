#!/usr/bin/env bash
# Recovery for sgrb2 after visualization-bug OOM kills:
#   - F212N all 8 modules plain iter1+iter2 (OOM in zoomcut_list before PSF fitting)
#   - LW nrcalong F300M/F360M/F405N/F466N/F480M plain iter1+iter2 (same)
#   - F466N nrcblong all 24 tasks iter1 + iter2 (task 17 PSF read error; iter2 was cancelled)
#   - New iter2 merge gated on all new iter2 IDs
# The already-completed iter2 catalogs (F150W/F182M/F187N/F210M/LW nrcblong) exist on disk
# and will be picked up by merge_catalogs.py automatically.
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
lw_mem=32gb

sw_mods=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
lw_mods=(nrcalong nrcblong)

all_new_iter2=()

submit_pair_plain() {
    # Submit iter1+iter2 plain (no bgsub); appends iter2 ID to all_new_iter2.
    local filter="$1" module="$2" mem="$3" each_suffix="$4" array_range="$5"
    local dao_args="--daophot --skip-crowdsource"

    local i1
    i1=$(sbatch --parsable \
        --array="${array_range}" \
        --job-name="webb-cat-${target}-${filter}-${module}-eachexp" \
        --output="${logdir}/webb-cat-${target}-${filter}-${module}-eachexp_%j-%A_%a.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=2 --nodes=1 --mem="${mem}" --time=96:00:00 \
        --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} \
--each-exposure --proposal_id=${proposal_id} --target=${target} \
--each-suffix=${each_suffix} ${dao_args}")
    echo "  Submitted iter1 ${i1}  ${target} ${filter} ${module} plain" >&2

    local i2
    i2=$(sbatch --parsable \
        --dependency="afterok:${i1}" \
        --array="${array_range}" \
        --job-name="webb-cat-${target}-iter2-${filter}-${module}-eachexp" \
        --output="${logdir}/webb-cat-${target}-iter2-${filter}-${module}-eachexp_%j-%A_%a.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=2 --nodes=1 --mem="${mem}" --time=96:00:00 \
        --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} \
--each-exposure --proposal_id=${proposal_id} --target=${target} \
--each-suffix=${each_suffix} ${dao_args} \
--iteration-label=iter2 --postprocess-residuals")
    echo "  Submitted iter2 ${i2}  ${target} ${filter} ${module} plain" >&2

    all_new_iter2+=("${i2}")
}

# ---------- F212N plain iter1+iter2 (8 modules, 192gb) ----------
echo "=== F212N plain recovery (192gb, 8 modules) ===" >&2
for module in "${sw_mods[@]}"; do
    submit_pair_plain F212N "${module}" "${sw_mem}" destreak_o001_crf "0-23"
done

# ---------- LW nrcalong: F300M F360M F405N F466N F480M plain (32gb) ----------
for filter in F300M F360M F405N F466N F480M; do
    echo "=== ${filter} nrcalong plain recovery (32gb) ===" >&2
    submit_pair_plain "${filter}" nrcalong "${lw_mem}" align_o001_crf "0-23"
done

# ---------- F466N nrcblong: resubmit all 24 iter1 tasks + iter2 (32gb) ----------
echo "=== F466N nrcblong full resubmit (32gb) ===" >&2
submit_pair_plain F466N nrcblong "${lw_mem}" align_o001_crf "0-23"

# Also include F300M nrcblong iter2 (30825736) which is still pending
# (its iter1 task 23 was running; include it in merge dep if active)
f300m_nrcblong_iter2=30825736
if squeue -j ${f300m_nrcblong_iter2} --format="%i" -h 2>/dev/null | grep -q .; then
    echo "F300M nrcblong iter2 ${f300m_nrcblong_iter2} is still in queue; adding to merge dep" >&2
    all_new_iter2+=("${f300m_nrcblong_iter2}")
fi

# ---------- New iter2 merge ----------
iter2_merge_dep=$(IFS=:; echo "${all_new_iter2[*]}")
echo "" >&2
echo "New iter2 IDs: ${all_new_iter2[*]}" >&2

iter2_merge_id=$(sbatch --parsable \
    --dependency="afterok:${iter2_merge_dep}" \
    --job-name="webb-cat-merge-${target}-iter2" \
    --output="${logdir}/webb-cat-merge-${target}-iter2_%j.log" \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
    --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py \
--merge-singlefields --modules=merged --indiv-merge-methods=daoiterative \
--skip-crowdsource --target=${target} --iteration-label=iter2")
echo "Submitted iter2 merge ${iter2_merge_id}" >&2

echo "" >&2
echo "DONE submitting recovery2 for ${target}." >&2
echo "After iter2 merge ${iter2_merge_id} completes, run iter3:" >&2
echo "  bash brick2221/shellscripts/run_iter3_cataloging.sh --target=sgrb2 --iter2-merge-dep ${iter2_merge_id}" >&2
