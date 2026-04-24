# Submit per-exposure cataloging jobs for brick/cloudc, using bundle_size to
# fold K consecutive exposures into one SLURM array task and a single
# --finalize-only mosaic job per config (covers iter=None and iter=iter2).
#
# Queued-task math (vs. the pre-bundling version):
#   Per config, old:  2 arrays of 24 tasks + 2 singleton mosaics = 50.
#   Per config, now:  2 arrays of 6 tasks  + 1 singleton mosaic  = 13.
#dao="--daophot --skip-crowdsource"
# enables modifying globally whether you're doing just crowdsource or both (" " = crowdsource only)
daoloop=("--daophot --skip-crowdsource")
# daoloop=("--daophot --skip-crowdsource" " ")
mem=16gb
bundle_size=${BUNDLE_SIZE:-4}

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
logdir=/blue/adamginsburg/adamginsburg/brick_logs/
python_exe=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
analysis_script=/blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py
analysis_dir=/blue/adamginsburg/adamginsburg/jwst/brick/analysis
basepath=/blue/adamginsburg/adamginsburg/jwst/brick

compute_array_range() {
    # Compute a sparse --array spec over bundled task indices for which the
    # expected per-exposure output is missing.  Returns empty string if nothing
    # is needed.
    local filter="$1" module="$2" dao="$3" proposal_id="$4" target="$5" each_suffix="$6" iter_label="$7"
    local extra=""
    if [[ -n "${each_suffix}" ]]; then extra="--each-suffix=${each_suffix}"; fi
    local iter_arg=""
    if [[ -n "${iter_label}" ]]; then iter_arg="--iteration-label=${iter_label}"; fi
    "${python_exe}" "${analysis_script}" \
        --filternames="${filter}" --modules="${module}" --each-exposure \
        ${dao} --proposal_id="${proposal_id}" --target="${target}" ${extra} \
        ${iter_arg} --bundle-size="${bundle_size}" --list-missing-tasks 2>/dev/null \
        | awk -F: '/^__MISSING_TASKS__:/{sub(/^__MISSING_TASKS__:/,""); print; found=1} END{if(!found) print ""}'
}

submit_catalog_and_residual_mosaic() {
    local filter="$1"
    local module="$2"
    local dao="$3"
    local mem="$4"
    local proposal_id="$5"
    local target="$6"
    local each_suffix="$7"

    local extra_args="--proposal_id=${proposal_id} --target=${target}"
    if [[ -n "${each_suffix}" ]]; then
        extra_args="${extra_args} --each-suffix=${each_suffix}"
    fi

    # iter1 array over bundled exposure tasks.
    local iter1_range
    iter1_range=$(compute_array_range "${filter}" "${module}" "${dao}" "${proposal_id}" "${target}" "${each_suffix}" "")
    local array_jobid=""
    if [[ -n "${iter1_range}" ]]; then
        array_jobid=$(sbatch --parsable --array="${iter1_range}" --job-name=webb-cat-${filter}-${module}-eachexp-${target} --output=${logdir}/webb-cat-${filter}-${module}-eachexp-${target}_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "${python_exe} ${analysis_script} --filternames=${filter} --modules=${module} --each-exposure ${dao} ${extra_args} --bundle-size=${bundle_size} --skip-if-done")
    else
        echo "iter1 already complete for ${filter} ${module} ${target}; skipping iter1 submission."
    fi

    if [[ "${dao}" == *"--daophot"* ]]; then
        local iter2_range
        iter2_range=$(compute_array_range "${filter}" "${module}" "${dao}" "${proposal_id}" "${target}" "${each_suffix}" "iter2")
        local iter2_jobid=""
        if [[ -n "${iter2_range}" ]]; then
            local dep_arg=""
            if [[ -n "${array_jobid}" ]]; then dep_arg="--dependency=afterok:${array_jobid}"; fi
            iter2_jobid=$(sbatch --parsable ${dep_arg} --array="${iter2_range}" --job-name=webb-cat-${filter}-${module}-eachexp-${target}-iter2 --output=${logdir}/webb-cat-${filter}-${module}-eachexp-${target}-iter2_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "${python_exe} ${analysis_script} --filternames=${filter} --modules=${module} --each-exposure ${dao} ${extra_args} --iteration-label=iter2 --postprocess-residuals --bundle-size=${bundle_size} --skip-if-done")
        else
            echo "iter2 already complete for ${filter} ${module} ${target}; skipping iter2 submission."
        fi

        local field
        if [[ "${proposal_id}" == "2221" && "${target}" == "brick" ]]; then
            field="001"
        elif [[ "${proposal_id}" == "2221" && "${target}" == "cloudc" ]]; then
            field="002"
        elif [[ "${proposal_id}" == "1182" && "${target}" == "brick" ]]; then
            field="004"
        else
            echo "Skipping residual mosaic submission: unknown field mapping for proposal_id=${proposal_id} target=${target}"
            return
        fi

        # One finalize-only mosaic job covers both iteration labels.
        local mosaic_dep=""
        if [[ -n "${iter2_jobid}" ]]; then
            mosaic_dep="--dependency=afterok:${iter2_jobid}"
        elif [[ -n "${array_jobid}" ]]; then
            mosaic_dep="--dependency=afterok:${array_jobid}"
        fi
        sbatch ${mosaic_dep} --job-name=webb-mosaic-${filter}-${module}-${target} --output=${logdir}/webb-mosaic-${filter}-${module}-${target}_%j.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 --wrap "${python_exe} ${analysis_script} --filternames=${filter} --modules=${module} --each-exposure ${dao} ${extra_args} --finalize-only --iteration-labels=,iter2"
    fi
}

for filter in F410M F405N F466N; do
    for module in nrca nrcb; do
        for dao in "${daoloop[@]}"; do
            submit_catalog_and_residual_mosaic "${filter}" "${module}" "${dao}" "${mem}" "2221" "brick" ""
        done
    done
done


mem=20gb
for filter in F212N F182M F187N; do
    for modnum in 1 2 3 4; do
        for module in nrca${modnum} nrcb${modnum}; do
            for dao in "${daoloop[@]}"; do
                submit_catalog_and_residual_mosaic "${filter}" "${module}" "${dao}" "${mem}" "2221" "brick" ""
            done
        done
    done
done


#mem=16gb
for filter in F356W F444W; do
    for module in nrca nrcb; do
        for dao in "${daoloop[@]}"; do
            submit_catalog_and_residual_mosaic "${filter}" "${module}" "${dao}" "${mem}" "1182" "brick" "destreak_o004_crf"
        done
    done
done

mem=32gb
for filter in F115W F200W; do
    for modnum in 1 2 3 4; do
        for module in nrca${modnum} nrcb${modnum}; do
            for dao in "${daoloop[@]}"; do
                submit_catalog_and_residual_mosaic "${filter}" "${module}" "${dao}" "${mem}" "1182" "brick" "destreak_o004_crf"
            done
        done
    done
done

mem=20gb
for filter in F410M F405N F466N; do
    for module in nrca nrcb; do
        for dao in "--daophot --skip-crowdsource" " "; do
            submit_catalog_and_residual_mosaic "${filter}" "${module}" "${dao}" "${mem}" "2221" "cloudc" "destreak_o002_crf"
        done
    done
done

mem=40gb
for filter in F212N F182M F187N; do
    for modnum in 1 2 3 4; do
        for module in nrca${modnum} nrcb${modnum}; do
            for dao in "--daophot --skip-crowdsource" " "; do
                submit_catalog_and_residual_mosaic "${filter}" "${module}" "${dao}" "${mem}" "2221" "cloudc" "destreak_o002_crf"
            done
        done
    done
done
