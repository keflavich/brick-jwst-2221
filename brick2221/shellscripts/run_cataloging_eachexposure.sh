#dao="--daophot --skip-crowdsource"
# enables modifying globally whether you're doing just crowdsource or both (" " = crowdsource only)
daoloop=("--daophot --skip-crowdsource")
# daoloop=("--daophot --skip-crowdsource" " ")
mem=16gb

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
logdir=/blue/adamginsburg/adamginsburg/brick_logs/
python_exe=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
analysis_script=/blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py
analysis_dir=/blue/adamginsburg/adamginsburg/jwst/brick/analysis
basepath=/blue/adamginsburg/adamginsburg/jwst/brick

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

    local array_jobid
    array_jobid=$(sbatch --parsable --array=0-23 --job-name=webb-cat-${filter}-${module}-eachexp-${target} --output=${logdir}/webb-cat-${filter}-${module}-eachexp-${target}_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "${python_exe} ${analysis_script} --filternames=${filter} --modules=${module} --each-exposure ${dao} ${extra_args}")

    if [[ "${dao}" == *"--daophot"* ]]; then
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

        sbatch --dependency=afterok:${array_jobid} --job-name=webb-mosaic-${filter}-${module}-${target} --output=${logdir}/webb-mosaic-${filter}-${module}-${target}_%j.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 --wrap "FILTER=${filter} MODULE=${module} PROPOSAL_ID=${proposal_id} FIELD=${field} BASEPATH=${basepath} ANALYSIS_DIR=${analysis_dir} ${python_exe} -c \"import os, sys; sys.path.insert(0, os.environ['ANALYSIS_DIR']); import crowdsource_catalogs_long as c; [c.mosaic_each_exposure_residuals(basepath=os.environ['BASEPATH'], filtername=os.environ['FILTER'], proposal_id=os.environ['PROPOSAL_ID'], field=os.environ['FIELD'], module=os.environ['MODULE'], residual_kind=kind, desat=False, bgsub=False, epsf=False, blur=False, group=False, pupil='clear') for kind in ('basic', 'iterative')]\""
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
