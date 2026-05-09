#!/usr/bin/env bash
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
mkdir -p "$logdir"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
basepath=/orange/adamginsburg/jwst/sickle
proposal_id=3958
field=007

submit_residual_mosaic_only() {
    local filter="$1"
    local module="$2"
    local bgsub_flag="$3"
    local iteration_label="${4:-}"

    local bgsub_py="False"
    local bgsub_suffix=""
    local bgsub_tag=""
    if [[ "${bgsub_flag}" == "--bgsub" ]]; then
        bgsub_py="True"
        bgsub_suffix="_bgsub"
        bgsub_tag="-bgsub"
    fi

    local iter_suffix=""
    local iter_tag=""
    local iter_py="None"
    if [[ -n "${iteration_label}" ]]; then
        iter_suffix="_${iteration_label}"
        iter_tag="-${iteration_label}"
        iter_py="'${iteration_label}'"
    fi

    local kinds=()
    local kind
    local -a modules=()
    if [[ "${filter}" == "F187N" || "${filter}" == "F210M" ]]; then
        modules=(nrcb1 nrcb2 nrcb3 nrcb4)
    else
        modules=(nrcb)
    fi
    for kind in basic iterative; do
        for module_name in "${modules[@]}"; do
            local pattern="${basepath}/${filter}/pipeline/jw0${proposal_id}-o${field}_t001_nircam_clear-${filter,,}-${module_name}_visit*_vgroup*_exp*${bgsub_suffix}${iter_suffix}_daophot_${kind}_residual.fits"
            if compgen -G "${pattern}" > /dev/null; then
                kinds+=("${kind}")
                break
            fi
        done
    done

    if [[ ${#kinds[@]} -eq 0 ]]; then
        echo "Skipping mosaicing-only job for ${filter} ${module} ${bgsub_flag}: no matching residual inputs found"
        return
    fi

    local kinds_csv
    kinds_csv=$(IFS=,; echo "${kinds[*]}")

    sbatch --job-name=webb-mosaic-sickle-${filter}-${module}${bgsub_tag}${iter_tag} \
        --output=${logdir}/webb-mosaic-sickle-${filter}-${module}${bgsub_tag}${iter_tag}_%j.log \
        --account=astronomy-dept \
        --qos=astronomy-dept-b \
        --ntasks=1 \
        --nodes=1 \
        --mem=24gb \
        --time=24:00:00 \
        --wrap "FILTER=${filter} MODULE=${module} BASEPATH=${basepath} ANALYSIS_DIR=${analysis_dir} KINDS=${kinds_csv} ${python_exec} -c \"import os, sys; sys.path.insert(0, os.environ['ANALYSIS_DIR']); import crowdsource_catalogs_long as c; [c.mosaic_each_exposure_residuals(basepath=os.environ['BASEPATH'], filtername=os.environ['FILTER'], proposal_id='${proposal_id}', field='${field}', module=os.environ['MODULE'], residual_kind=kind, desat=False, bgsub=${bgsub_py}, epsf=False, blur=False, group=False, pupil='clear', iteration_label=${iter_py}) for kind in os.environ['KINDS'].split(',')]\""

    echo "Submitted mosaicing-only job for ${filter} ${module} ${bgsub_flag} ${iteration_label} kinds=${kinds_csv}"
}

# NOTE: Sickle is NRCB-only (SUB640 for proposal 3958 field 007)
# Short-wavelength filters use nrcb1-4; long-wavelength filters use nrcb.

for filter in F187N F210M; do
    module=nrcb
    for bgsub in " " "--bgsub"; do
        submit_residual_mosaic_only "${filter}" "${module}" "${bgsub}" ""
        submit_residual_mosaic_only "${filter}" "${module}" "${bgsub}" "iter2"
    done
done

for filter in F335M F470N F480M; do
    module=nrcb
    for bgsub in " " "--bgsub"; do
        submit_residual_mosaic_only "${filter}" "${module}" "${bgsub}" ""
        submit_residual_mosaic_only "${filter}" "${module}" "${bgsub}" "iter2"
    done
done
