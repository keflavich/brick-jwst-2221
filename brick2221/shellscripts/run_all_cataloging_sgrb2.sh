#!/usr/bin/env bash
# Submit iter1 + iter2 per-frame photometry for all Sgr B2 (proposal 5365)
# filters, then merge.  Run run_iter3_cataloging.sh --target=sgrb2 afterwards.
#
# Filter layout:
#   SW destreak (24/mod): F182M, F210M, F212N  — modules nrca1-4, nrcb1-4
#   SW destreak (48/mod): F187N                — modules nrca1-4, nrcb1-4
#   SW align    (48/mod): F150W                — modules nrca1-4, nrcb1-4
#   LW align    (48/mod): F300M F360M F405N F410M F466N F480M — nrcalong, nrcblong
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
mkdir -p "${logdir}"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
proposal_id=5365
target=sgrb2
field=001
short_mem=48gb
long_mem=32gb
BUNDLE_SIZE=${BUNDLE_SIZE:-4}

compute_array_range() {
    local filter="$1" module="$2" dao_args="$3" filt_each_suffix="$4" iter_label="$5"
    local iter_arg=""
    if [[ -n "${iter_label}" ]]; then iter_arg="--iteration-label=${iter_label}"; fi
    "${python_exec}" "${script}" \
        --filternames="${filter}" --modules="${module}" --each-exposure \
        --proposal_id="${proposal_id}" --target="${target}" --each-suffix="${filt_each_suffix}" \
        ${dao_args} ${iter_arg} --bundle-size="${BUNDLE_SIZE}" --list-missing-tasks 2>/dev/null \
        | awk -F: '/^__MISSING_TASKS__:/{sub(/^__MISSING_TASKS__:/,""); print; found=1} END{if(!found) print ""}'
}

# Filter → each_suffix
filter_suffix() {
    case "$1" in
        F182M|F187N|F210M|F212N) echo "destreak_o001_crf" ;;
        *)                        echo "align_o001_crf" ;;
    esac
}

# Filter → array range (upper bound = files_per_module - 1)
# F187N has 48 exposures/module; F150W and all LW have 24; F182M/F210M/F212N have 24.
filter_array_range() {
    case "$1" in
        F187N) echo "0-47" ;;
        *)     echo "0-23" ;;
    esac
}

# Filter → is short wavelength?
is_sw() {
    case "$1" in
        F150W|F182M|F187N|F210M|F212N) return 0 ;;
        *) return 1 ;;
    esac
}

sw_mods=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
lw_mods=(nrcalong nrcblong)

all_iter1_plain_jobids=()
all_iter2_plain_jobids=()

# ---- per-filter submission ----

for filter in F150W F182M F187N F210M F212N F300M F360M F405N F410M F466N F480M; do
    each_suffix=$(filter_suffix "${filter}")
    array_range=$(filter_array_range "${filter}")
    if is_sw "${filter}"; then
        mods=("${sw_mods[@]}")
        mem="${short_mem}"
    else
        mods=("${lw_mods[@]}")
        mem="${long_mem}"
    fi

    plain_iter1_ids=()
    plain_iter2_ids=()

    for module in "${mods[@]}"; do
        for bgsub_opt in "" "--bgsub"; do
            bgsub_tag="${bgsub_opt:+-bgsub}"
            dao_args="--daophot --skip-crowdsource${bgsub_opt:+ }${bgsub_opt}"

            # --- iter1: only the bundled tasks still missing outputs ---
            iter1_range=$(compute_array_range "${filter}" "${module}" "${dao_args}" "${each_suffix}" "")
            iter1_id=""
            if [[ -n "${iter1_range}" ]]; then
                iter1_id=$(sbatch --parsable \
                    --array="${iter1_range}" \
                    --job-name="webb-cat-sgrb2-${filter}-${module}${bgsub_tag}-eachexp" \
                    --output="${logdir}/webb-cat-sgrb2-${filter}-${module}${bgsub_tag}-eachexp_%j-%A_%a.log" \
                    --account=astronomy-dept --qos=astronomy-dept-b \
                    --ntasks=2 --nodes=1 --mem="${mem}" --time=96:00:00 \
                    --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} \
--each-exposure --proposal_id=${proposal_id} --target=${target} \
--each-suffix=${each_suffix} ${dao_args} --bundle-size=${BUNDLE_SIZE} --skip-if-done")
                echo "Submitted iter1 ${iter1_id} (range=${iter1_range}) for ${target} ${filter} ${module} ${bgsub_opt:-plain}" >&2
            else
                echo "iter1 already complete for ${target} ${filter} ${module} ${bgsub_opt:-plain}" >&2
            fi

            # --- iter2: sparse + gated on iter1 ---
            iter2_range=$(compute_array_range "${filter}" "${module}" "${dao_args}" "${each_suffix}" "iter2")
            iter2_id=""
            if [[ -n "${iter2_range}" ]]; then
                iter2_dep_args=()
                [[ -n "${iter1_id}" ]] && iter2_dep_args+=(--dependency="afterok:${iter1_id}")
                iter2_id=$(sbatch --parsable \
                    "${iter2_dep_args[@]}" \
                    --array="${iter2_range}" \
                    --job-name="webb-cat-sgrb2-iter2-${filter}-${module}${bgsub_tag}-eachexp" \
                    --output="${logdir}/webb-cat-sgrb2-iter2-${filter}-${module}${bgsub_tag}-eachexp_%j-%A_%a.log" \
                    --account=astronomy-dept --qos=astronomy-dept-b \
                    --ntasks=2 --nodes=1 --mem="${mem}" --time=96:00:00 \
                    --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} \
--each-exposure --proposal_id=${proposal_id} --target=${target} \
--each-suffix=${each_suffix} ${dao_args} \
--iteration-label=iter2 --postprocess-residuals --bundle-size=${BUNDLE_SIZE} --skip-if-done")
                echo "Submitted iter2 ${iter2_id} (range=${iter2_range}) for ${target} ${filter} ${module} ${bgsub_opt:-plain}" >&2
            else
                echo "iter2 already complete for ${target} ${filter} ${module} ${bgsub_opt:-plain}" >&2
            fi

            if [[ -z "${bgsub_opt}" ]]; then
                [[ -n "${iter1_id}" ]] && plain_iter1_ids+=("${iter1_id}")
                [[ -n "${iter2_id}" ]] && plain_iter2_ids+=("${iter2_id}")
            fi
        done
    done

    # --- Residual mosaics (per-module) after iter1 and iter2, plain only ---
    for module in "${mods[@]}"; do
        # Collect the plain iter1/iter2 ids for this module
        # (Re-query from the arrays already captured above; since we iterate
        # modules in the same order, the index aligns with sw_mods / lw_mods.)
        : # mosaics submitted below using the full plain dep sets for simplicity
    done

    # Aggregate all plain iter1/iter2 deps for the per-filter residual mosaics.
    # Mosaics for SW use 'nrca' and 'nrcb' aggregate modules (expanded in the
    # mosaic function via the sgrb2 branch added to crowdsource_catalogs_long.py).
    # Mosaics for LW use 'nrcalong' and 'nrcblong' directly.
    iter1_dep=$(IFS=:; echo "${plain_iter1_ids[*]}")
    iter2_dep=$(IFS=:; echo "${plain_iter2_ids[*]}")

    if is_sw "${filter}"; then
        mosaic_modules=(nrca nrcb)
    else
        mosaic_modules=("${mods[@]}")
    fi

    # One finalize-only mosaic job per (filter, aggregate module) covers BOTH
    # iter=None and iter=iter2 (previously 2 separate singletons).  Depends on
    # iter2 if available, else iter1.
    for agg_mod in "${mosaic_modules[@]}"; do
        mosaic_dep="${iter2_dep:-${iter1_dep}}"
        dep_arg=""
        [[ -n "${mosaic_dep}" ]] && dep_arg="--dependency=afterok:${mosaic_dep}"
        sbatch ${dep_arg} \
            --job-name="webb-mosaic-sgrb2-${filter}-${agg_mod}" \
            --output="${logdir}/webb-mosaic-sgrb2-${filter}-${agg_mod}_%j.log" \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 \
            --wrap "${python_exec} ${script} --filternames=${filter} --modules=${agg_mod} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${each_suffix} --daophot --skip-crowdsource --finalize-only --iteration-labels=,iter2" >/dev/null
        echo "Submitted folded mosaic for ${filter} ${agg_mod} (both iter labels)" >&2
    done

    all_iter1_plain_jobids+=("${plain_iter1_ids[@]}")
    all_iter2_plain_jobids+=("${plain_iter2_ids[@]}")
done

# ---- iter1 merge ----
if [[ ${#all_iter1_plain_jobids[@]} -gt 0 ]]; then
    iter1_merge_dep=$(IFS=:; echo "${all_iter1_plain_jobids[*]}")
    iter1_merge_id=$(sbatch --parsable \
        --dependency="afterok:${iter1_merge_dep}" \
        --job-name="webb-cat-merge-sgrb2-iter1" \
        --output="${logdir}/webb-cat-merge-sgrb2-iter1_%j.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
        --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py \
--merge-singlefields --modules=merged --indiv-merge-methods=dao \
--skip-crowdsource --target=${target}")
    echo "Submitted iter1 merge ${iter1_merge_id}"
fi

# ---- iter2 merge ----
if [[ ${#all_iter2_plain_jobids[@]} -gt 0 ]]; then
    iter2_merge_dep=$(IFS=:; echo "${all_iter2_plain_jobids[*]}")
    iter2_merge_id=$(sbatch --parsable \
        --dependency="afterok:${iter2_merge_dep}" \
        --job-name="webb-cat-merge-sgrb2-iter2" \
        --output="${logdir}/webb-cat-merge-sgrb2-iter2_%j.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
        --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py \
--merge-singlefields --modules=merged --indiv-merge-methods=daoiterative \
--skip-crowdsource --target=${target} --iteration-label=iter2")
    echo "Submitted iter2 merge ${iter2_merge_id}"
fi

echo "DONE submitting iter1+iter2 chain for ${target}."
echo "Run: run_iter3_cataloging.sh --target=sgrb2 --iter2-merge-dep ${iter2_merge_id:-<iter2_merge_id>}"
