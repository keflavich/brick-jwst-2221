#!/usr/bin/env bash
# Recovery + continuation for sgrb2 iter1/iter2 pipeline.
#
# Resubmits modules that had OOM kills at 48gb (now 96gb):
#   F182M: all 8 modules plain (tasks 22-23 OOM at 48gb)
#   F150W: nrca3-4, nrcb1-4 plain; nrca2-4, nrcb1-4 bgsub (KeyError bug + OOM)
#   F187N: nrca1-4 + nrcb3 plain (OOM); nrca1-3 bgsub (OOM)
#
# Also submits:
#   F210M nrcb4: iter2 gated on already-submitted iter1 (30809195) + bgsub pair
#
# F212N, LW filters, and merges are deferred to run_continuation2_sgrb2.sh
# (too many tasks to fit in one QOS batch).
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
sw_mem=96gb   # increased from 48gb; OOM kills observed on dense Sgr B2 exposures
long_mem=32gb

sw_mods=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
lw_mods=(nrcalong nrcblong)

# New plain iter1/iter2 IDs accumulated by this script (for merge dependency).
# Starts empty; submit_pair adds to it for plain (not bgsub) jobs.
all_iter1_plain=()
all_iter2_plain=()

# ---------- helpers ----------

# Submit iter1+iter2 pair; adds plain IDs to global arrays.
# Args: filter module mem each_suffix array_range [bgsub_opt]
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

# ---------- F182M: all 8 modules at 96gb (OOM at 48gb for tasks 22-23) ----------
echo "=== F182M plain recovery (96gb) ===" >&2
filter=F182M; each_suffix=destreak_o001_crf; array_range="0-23"; mem=${sw_mem}
for module in "${sw_mods[@]}"; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" ""
done

# ---------- F150W: nrca3-4 + nrcb1-4 plain at 96gb (OOM); nrca1-2 already OK ----------
echo "=== F150W plain recovery (96gb, 6 modules) ===" >&2
filter=F150W; each_suffix=align_o001_crf; array_range="0-23"; mem=${sw_mem}
for module in nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" ""
done

# ---------- F187N: nrca1-4 + nrcb3 plain at 96gb (OOM); nrcb1,2,4 iter1 still running ----------
echo "=== F187N plain recovery (96gb) ===" >&2
filter=F187N; each_suffix=destreak_o001_crf; array_range="0-47"; mem=${sw_mem}
for module in nrca1 nrca2 nrca3 nrca4 nrcb3; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" ""
done

# ---------- F182M bgsub: 7 modules (skip nrcb4 - still running at 48gb) ----------
echo "=== F182M bgsub recovery (96gb, 7 modules) ===" >&2
filter=F182M; each_suffix=destreak_o001_crf; array_range="0-23"; mem=${sw_mem}
for module in nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" "--bgsub"
done

# ---------- F210M nrcb4: iter2 gated on already-submitted iter1 (30809195) + bgsub pair ----------
echo "=== F210M nrcb4 continuation ===" >&2
filter=F210M; each_suffix=destreak_o001_crf; array_range="0-23"; mem=48gb

# iter2 plain gated on 30809195 (already in queue)
f210m_nrcb4_i2=$(sbatch --parsable \
    --dependency="afterok:30809195" \
    --array="${array_range}" \
    --job-name="webb-cat-${target}-iter2-${filter}-nrcb4-eachexp" \
    --output="${logdir}/webb-cat-${target}-iter2-${filter}-nrcb4-eachexp_%j-%A_%a.log" \
    --account=astronomy-dept --qos=astronomy-dept-b \
    --ntasks=2 --nodes=1 --mem="${mem}" --time=96:00:00 \
    --wrap "${python_exec} ${script} --filternames=${filter} --modules=nrcb4 \
--each-exposure --proposal_id=${proposal_id} --target=${target} \
--each-suffix=${each_suffix} --daophot --skip-crowdsource \
--iteration-label=iter2 --postprocess-residuals")
echo "  Submitted F210M nrcb4 iter2 ${f210m_nrcb4_i2}" >&2
all_iter2_plain+=("${f210m_nrcb4_i2}")
all_iter1_plain+=("30809195")  # already submitted iter1

# bgsub pair for nrcb4
submit_pair "${filter}" nrcb4 "${mem}" "${each_suffix}" "${array_range}" "--bgsub"

# ---------- F187N bgsub: nrca1-3 (OOM at 48gb; nrca4 bgsub iter1 already completed OK) ----------
echo "=== F187N bgsub recovery (96gb, 3 modules) ===" >&2
filter=F187N; each_suffix=destreak_o001_crf; array_range="0-47"; mem=${sw_mem}
for module in nrca1 nrca2 nrca3; do
    submit_pair "${filter}" "${module}" "${mem}" "${each_suffix}" "${array_range}" "--bgsub"
done

# NOTE: F150W bgsub (7 modules) and F210M nrcb4 bgsub deferred to run_continuation2_sgrb2.sh
# to avoid exceeding the 3000-task QOS limit in this batch.

echo "" >&2
echo "=== Recovery submissions complete ===" >&2
echo "New plain iter1 job IDs (${#all_iter1_plain[@]}): ${all_iter1_plain[*]}" >&2
echo "New plain iter2 job IDs (${#all_iter2_plain[@]}): ${all_iter2_plain[*]}" >&2
echo "" >&2
echo "Still-active valid IDs to include in merge dependency:" >&2
echo "  F150W nrca1 iter2: 30796359" >&2
echo "  F150W nrca2 iter2: 30796363 (Dependency on 30796362)" >&2
echo "  F210M nrca1-4,nrcb1-3 iter2: 30808446 30808450 30808454 30808458 30808462 30808466 30808470" >&2
echo "  F187N nrcb1 iter2: 30798645" >&2
echo "  F187N nrcb2 iter2: 30807737" >&2
echo "  F187N nrcb4 iter2: 30807745" >&2
echo "" >&2
echo "Next step: run run_continuation2_sgrb2.sh to submit F212N, LW, and merges." >&2
