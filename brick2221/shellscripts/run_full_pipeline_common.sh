#!/usr/bin/env bash
set -euo pipefail

# Common end-to-end runner for one target:
# 1) First-pass pipeline (no reference catalog)
# 2) Build reference catalog from first-pass products
# 3) Second-pass pipeline (skip step1/2, now with reference catalog)
# 4) Per-exposure cataloging (DAO)
# 5) Merge single-field catalogs

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <target>" >&2
    echo "Targets: cloudef, sgrc, sgrb2, arches, quintuplet, sgra" >&2
    exit 2
fi

target="$1"

logdir=/blue/adamginsburg/adamginsburg/logs/jwst_full_pipeline
mkdir -p "${logdir}"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
pipeline_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/PipelineRerunNIRCAM-LONG.py
catalog_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
merge_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/merge_catalogs.py
refcat_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/make_reference_from_pipeline_catalogs.py
fov_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/make_fov_region_from_mast_i2d.py
fwhm_source=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/fwhm_table.ecsv
DEFAULT_CRDS_PATH=/orange/adamginsburg/jwst/crds

MODULES=${MODULES:-merged}
ARRAY_RANGE=${ARRAY_RANGE:-0-23}

set_target_defaults() {
    local name="$1"
    case "${name}" in
        cloudef)
            DEF_PROPOSAL_ID=2092
            DEF_FIELD=005
            DEF_FILTERS=F162M,F210M,F360M,F480M
            DEF_REF_FILTER=F210M
            ;;
        sgrc)
            DEF_PROPOSAL_ID=4147
            DEF_FIELD=012
            DEF_FILTERS=F115W,F162M,F182M,F212N,F360M,F405N,F470N,F480M
            DEF_REF_FILTER=F212N
            ;;
        sgrb2)
            DEF_PROPOSAL_ID=5365
            DEF_FIELD=001
            DEF_FILTERS=F150W,F182M,F187N,F210M,F212N,F300M,F360M,F405N,F410M,F466N,F480M
            DEF_REF_FILTER=F210M
            ;;
        arches)
            DEF_PROPOSAL_ID=2045
            DEF_FIELD=001
            DEF_FILTERS=F212N,F323N
            DEF_REF_FILTER=F212N
            ;;
        quintuplet)
            DEF_PROPOSAL_ID=2045
            DEF_FIELD=003
            DEF_FILTERS=F212N,F323N
            DEF_REF_FILTER=F212N
            ;;
        sgra)
            DEF_PROPOSAL_ID=1939
            DEF_FIELD=001
            DEF_FILTERS=F115W,F212N,F405N
            DEF_REF_FILTER=F212N
            ;;
        *)
            echo "Unknown target: ${name}. Supported targets: cloudef, sgrc, sgrb2, arches, quintuplet, sgra" >&2
            exit 2
            ;;
    esac
}

ensure_target_setup() {
    local name="$1"
    local proposal_id="$2"
    local field="$3"
    local basepath="/orange/adamginsburg/jwst/${name}"

    if [[ ! -d "${basepath}" ]]; then
        echo "Missing target directory: ${basepath}" >&2
        exit 1
    fi
    if [[ ! -d "${basepath}/crds" ]]; then
        if [[ -d "/orange/adamginsburg/jwst/crds" ]]; then
            ln -s "/orange/adamginsburg/jwst/crds" "${basepath}/crds"
            echo "Linked shared CRDS cache to ${basepath}/crds"
        else
            echo "Missing CRDS directory: ${basepath}/crds and shared fallback /orange/adamginsburg/jwst/crds not found" >&2
            exit 1
        fi
    fi

    mkdir -p "${basepath}/reduction"
    if [[ ! -f "${basepath}/reduction/fwhm_table.ecsv" ]]; then
        cp "${fwhm_source}" "${basepath}/reduction/fwhm_table.ecsv"
        echo "Seeded ${basepath}/reduction/fwhm_table.ecsv from repository default"
    fi

    mkdir -p "${basepath}/catalogs"

    local fov_file="${basepath}/regions_/nircam_${name}_fov.reg"
    if [[ ! -f "${fov_file}" ]]; then
        echo "No FOV region found for ${name}. Building from MAST i2d footprints..."
        "${python_exec}" "${fov_script}" --target="${name}" --proposal-id="${proposal_id}" --field="${field}"
    fi
}

submit_pipeline_filter_jobs() {
    local name="$1"
    local proposal_id="$2"
    local field="$3"
    local filters_csv="$4"
    local modules="$5"
    local skip_step_flag="$6"
    local dep_ids="$7"

    local -a filters=()
    IFS=',' read -r -a filters <<< "${filters_csv}"

    local -a submitted=()
    local filter
    local dep_arg=""
    if [[ -n "${dep_ids}" ]]; then
        dep_arg="--dependency=afterok:${dep_ids}"
    fi

    for filter in "${filters[@]}"; do
        local wrap_cmd="CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${pipeline_script} --proposal_id=${proposal_id} --field=${field} --filternames=${filter} --modules=${modules} ${skip_step_flag}"
        jobid=$(sbatch --parsable ${dep_arg} \
            --job-name="webb-pipe-${name}-${filter}" \
            --output="${logdir}/webb-pipe-${name}-${filter}_%j.log" \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 \
            --wrap "${wrap_cmd}")
        submitted+=("${jobid}")
        echo "Submitted pipeline job ${jobid} for ${name} ${filter} ${skip_step_flag}" >&2
    done

    (IFS=:; echo "${submitted[*]}")
}

submit_target_flow() {
    local name="$1"
    local proposal_id="$2"
    local field="$3"
    local filters_csv="$4"
    local ref_filter="$5"

    ensure_target_setup "${name}" "${proposal_id}" "${field}"

    first_pass_dep=$(submit_pipeline_filter_jobs "${name}" "${proposal_id}" "${field}" "${filters_csv}" "${MODULES}" "" "")

    refcat_jobid=$(sbatch --parsable --dependency=afterok:${first_pass_dep} \
        --job-name="webb-refcat-${name}" \
        --output="${logdir}/webb-refcat-${name}_%j.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=64gb --time=24:00:00 \
        --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${refcat_script} --target=${name} --proposal-id=${proposal_id} --field=${field} --filter=${ref_filter} --generate-catalogs")
    echo "Submitted reference-catalog job ${refcat_jobid} for ${name}"

    second_pass_dep=$(submit_pipeline_filter_jobs "${name}" "${proposal_id}" "${field}" "${filters_csv}" "${MODULES}" "--skip_step1and2" "${refcat_jobid}")

    local each_suffix="destreak_o${field}_crf"
    local -a filters=()
    IFS=',' read -r -a filters <<< "${filters_csv}"

    local -a catalog_jobids=()
    local filter
    for filter in "${filters[@]}"; do
        cat_jobid=$(sbatch --parsable --dependency=afterok:${second_pass_dep} \
            --array="${ARRAY_RANGE}" \
            --job-name="webb-cat-${name}-${filter}-eachexp" \
            --output="${logdir}/webb-cat-${name}-${filter}-eachexp_%j-%A_%a.log" \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
            --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${catalog_script} --filternames=${filter} --modules=${MODULES} --proposal_id=${proposal_id} --target=${name} --each-exposure --each-suffix=${each_suffix} --daophot --skip-crowdsource")
        catalog_jobids+=("${cat_jobid}")
        echo "Submitted catalog array ${cat_jobid} for ${name} ${filter}"
    done

    catalog_dep=$(IFS=:; echo "${catalog_jobids[*]}")
    merge_jobid=$(sbatch --parsable --dependency=afterok:${catalog_dep} \
        --job-name="webb-merge-${name}" \
        --output="${logdir}/webb-merge-${name}_%j.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
        --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${merge_script} --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource --target=${name}")
    echo "Submitted merge job ${merge_jobid} for ${name}"
}

set_target_defaults "${target}"

PROPOSAL_ID=${PROPOSAL_ID:-${DEF_PROPOSAL_ID}}
FIELD=${FIELD:-${DEF_FIELD}}
FILTERS=${FILTERS:-${DEF_FILTERS}}
REF_FILTER=${REF_FILTER:-${DEF_REF_FILTER}}
CRDS_PATH=${CRDS_PATH:-${DEFAULT_CRDS_PATH}}

if [[ ! -d "${CRDS_PATH}" ]]; then
    echo "Configured CRDS_PATH does not exist: ${CRDS_PATH}" >&2
    exit 1
fi

submit_target_flow "${target}" "${PROPOSAL_ID}" "${FIELD}" "${FILTERS}" "${REF_FILTER}"
