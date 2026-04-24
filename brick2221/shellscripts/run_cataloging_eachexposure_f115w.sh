#dao="--daophot --skip-crowdsource"
# enables modifying globally whether you're doing just crowdsource or both (" " = crowdsource only)
# daoloop=("--daophot --skip-crowdsource")
daoloop=("--daophot --skip-crowdsource" " ")
bundle_size=${BUNDLE_SIZE:-4}

python_exe=/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python
analysis_script=/blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py

compute_array_range() {
    local filter="$1" module="$2" dao="$3" iter_label="$4"
    local iter_arg=""
    if [[ -n "${iter_label}" ]]; then iter_arg="--iteration-label=${iter_label}"; fi
    "${python_exe}" "${analysis_script}" \
        --filternames="${filter}" --modules="${module}" --each-exposure \
        ${dao} --proposal_id=1182 --target=brick --each-suffix=destreak_o004_crf \
        ${iter_arg} --bundle-size="${bundle_size}" --list-missing-tasks 2>/dev/null \
        | awk -F: '/^__MISSING_TASKS__:/{sub(/^__MISSING_TASKS__:/,""); print; found=1} END{if(!found) print ""}'
}


mem=64gb
for filter in F115W; do
    for modnum in 1 2 3 4; do
        for module in nrca${modnum} nrcb${modnum}; do
            for dao in "${daoloop[@]}"; do
                range=$(compute_array_range "${filter}" "${module}" "${dao}" "")
                array_jobid=""
                if [[ -n "${range}" ]]; then
                    array_jobid=$(sbatch --parsable --array="${range}" --job-name=webb-cat-${filter}-${module}-eachexp --output=webb-cat-${filter}-${module}-eachexp_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "${python_exe} ${analysis_script} --filternames=${filter} --modules=${module} --each-exposure ${dao}  --proposal_id=1182 --target=brick --each-suffix=destreak_o004_crf --bundle-size=${bundle_size} --skip-if-done")
                fi
                if [[ "${dao}" == *"--daophot"* ]]; then
                    iter2_range=$(compute_array_range "${filter}" "${module}" "${dao}" "iter2")
                    iter2_jobid=""
                    if [[ -n "${iter2_range}" ]]; then
                        dep_arg=""
                        [[ -n "${array_jobid}" ]] && dep_arg="--dependency=afterok:${array_jobid}"
                        iter2_jobid=$(sbatch --parsable ${dep_arg} --array="${iter2_range}" --job-name=webb-cat-${filter}-${module}-eachexp-iter2 --output=webb-cat-${filter}-${module}-eachexp-iter2_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "${python_exe} ${analysis_script} --filternames=${filter} --modules=${module} --each-exposure ${dao}  --proposal_id=1182 --target=brick --each-suffix=destreak_o004_crf --iteration-label=iter2 --postprocess-residuals --bundle-size=${bundle_size} --skip-if-done")
                    fi
                    mosaic_dep=""
                    if [[ -n "${iter2_jobid}" ]]; then
                        mosaic_dep="--dependency=afterok:${iter2_jobid}"
                    elif [[ -n "${array_jobid}" ]]; then
                        mosaic_dep="--dependency=afterok:${array_jobid}"
                    fi
                    sbatch ${mosaic_dep} --job-name=webb-mosaic-${filter}-${module} --output=webb-mosaic-${filter}-${module}_%j.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 --wrap "${python_exe} ${analysis_script} --filternames=${filter} --modules=${module} --each-exposure ${dao} --proposal_id=1182 --target=brick --each-suffix=destreak_o004_crf --finalize-only --iteration-labels=,iter2"
                fi
            done
        done
    done
done

# for filter in F410M F405N F466N; do
#     for module in nrca nrcb; do
#         for dao in "--daophot --skip-crowdsource" " "; do
#             sbatch --array=0-23 --job-name=webb-cat-${filter}-${module}-eachexp-cloudc --output=webb-cat-${filter}-${module}-eachexp-cloudc_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --modules=${module} --each-exposure ${dao} --target=cloudc --each-suffix=destreak_o002_crf"
#         done
#     done
# done
#
# for filter in F212N F182M F187N; do
#     for modnum in 1 2 3 4; do
#         for module in nrca${modnum} nrcb${modnum}; do
#             for dao in "--daophot --skip-crowdsource" " "; do
#                 sbatch --array=0-23 --job-name=webb-cat-${filter}-${module}-eachexp-cloudc --output=webb-cat-${filter}-${module}-eachexp-cloudc_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --modules=${module} --each-exposure ${dao} --target=cloudc --each-suffix=destreak_o002_crf"
#             done
#         done
#     done
# done
