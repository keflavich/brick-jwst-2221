#!/usr/bin/env bash
set -euo pipefail

# Compatibility launcher: runs one or more targets through the common full-pipeline script.
# If no target is passed, run all supported targets.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
common_script="${script_dir}/run_full_pipeline_common.sh"

run_targets=("$@")
if [[ ${#run_targets[@]} -eq 0 ]]; then
    run_targets=(cloudef sgrc sgrb2 arches quintuplet sgra)
fi

for target in "${run_targets[@]}"; do
    "${common_script}" "${target}"
done
#!/usr/bin/env bash
set -euo pipefail

# End-to-end runner for cloudef, sgrc, sgrb2, arches, quintuplet, and sgra:
# 1) First-pass pipeline (no reference catalog)
# 2) Build reference catalog from first-pass products
# 3) Second-pass pipeline (skip step1/2, now with reference catalog)
# 4) Per-exposure cataloging (DAO)
# 5) Merge single-field catalogs

logdir=/blue/adamginsburg/adamginsburg/logs/jwst_full_pipeline
mkdir -p "${logdir}"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
pipeline_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/PipelineRerunNIRCAM-LONG.py
catalog_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
merge_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/merge_catalogs.py
refcat_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/make_reference_from_pipeline_catalogs.py
fov_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/make_fov_region_from_mast_i2d.py
fwhm_source=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/fwhm_table.ecsv

# Defaults can be overridden from environment.
CLOUDEF_PROPOSAL_ID=${CLOUDEF_PROPOSAL_ID:-2092}
CLOUDEF_FIELD=${CLOUDEF_FIELD:-005}
CLOUDEF_FILTERS=${CLOUDEF_FILTERS:-F162M,F210M,F360M,F480M}
CLOUDEF_REF_FILTER=${CLOUDEF_REF_FILTER:-F210M}

SGRC_PROPOSAL_ID=${SGRC_PROPOSAL_ID:-4147}
SGRC_FIELD=${SGRC_FIELD:-012}
SGRC_FILTERS=${SGRC_FILTERS:-F115W,F162M,F182M,F212N,F360M,F405N,F470N,F480M}
SGRC_REF_FILTER=${SGRC_REF_FILTER:-F212N}

SGRB2_PROPOSAL_ID=${SGRB2_PROPOSAL_ID:-5365}
SGRB2_FIELD=${SGRB2_FIELD:-001}
SGRB2_FILTERS=${SGRB2_FILTERS:-F150W,F182M,F187N,F210M,F212N,F300M,F360M,F405N,F410M,F466N,F480M}
SGRB2_REF_FILTER=${SGRB2_REF_FILTER:-F210M}

ARCHES_PROPOSAL_ID=${ARCHES_PROPOSAL_ID:-2045}
ARCHES_FIELD=${ARCHES_FIELD:-001}
ARCHES_FILTERS=${ARCHES_FILTERS:-F212N,F323N}
ARCHES_REF_FILTER=${ARCHES_REF_FILTER:-F212N}

QUINTUPLET_PROPOSAL_ID=${QUINTUPLET_PROPOSAL_ID:-2045}
QUINTUPLET_FIELD=${QUINTUPLET_FIELD:-003}
QUINTUPLET_FILTERS=${QUINTUPLET_FILTERS:-F212N,F323N}
QUINTUPLET_REF_FILTER=${QUINTUPLET_REF_FILTER:-F212N}

SGRA_PROPOSAL_ID=${SGRA_PROPOSAL_ID:-1939}
SGRA_FIELD=${SGRA_FIELD:-001}
SGRA_FILTERS=${SGRA_FILTERS:-F115W,F212N,F405N}
SGRA_REF_FILTER=${SGRA_REF_FILTER:-F212N}

MODULES=${MODULES:-merged}
ARRAY_RANGE=${ARRAY_RANGE:-0-23}
BUNDLE_SIZE=${BUNDLE_SIZE:-4}

ensure_target_setup() {
    local target="$1"
    local proposal_id="$2"
    local field="$3"
    local basepath="/orange/adamginsburg/jwst/${target}"

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

    local fov_file="${basepath}/regions_/nircam_${target}_fov.reg"
    if [[ ! -f "${fov_file}" ]]; then
        echo "No FOV region found for ${target}. Building from MAST i2d footprints..."
        "${python_exec}" "${fov_script}" --target="${target}" --proposal-id="${proposal_id}" --field="${field}"
    fi
}

submit_pipeline_filter_jobs() {
    local target="$1"
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
        jobid=$(sbatch --parsable ${dep_arg} \
            --job-name="webb-pipe-${target}-${filter}" \
            --output="${logdir}/webb-pipe-${target}-${filter}_%j.log" \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 \
            --wrap "${python_exec} ${pipeline_script} --proposal_id=${proposal_id} --field=${field} --filternames=${filter} --modules=${modules} ${skip_step_flag}")
        submitted+=("${jobid}")
        echo "Submitted pipeline job ${jobid} for ${target} ${filter} ${skip_step_flag}" >&2
    done

    (IFS=:; echo "${submitted[*]}")
}

submit_target_flow() {
    local target="$1"
    local proposal_id="$2"
    local field="$3"
    local filters_csv="$4"
    local ref_filter="$5"

    ensure_target_setup "${target}" "${proposal_id}" "${field}"

    first_pass_dep=$(submit_pipeline_filter_jobs "${target}" "${proposal_id}" "${field}" "${filters_csv}" "${MODULES}" "" "")

    refcat_jobid=$(sbatch --parsable --dependency=afterok:${first_pass_dep} \
        --job-name="webb-refcat-${target}" \
        --output="${logdir}/webb-refcat-${target}_%j.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=64gb --time=24:00:00 \
        --wrap "${python_exec} ${refcat_script} --target=${target} --proposal-id=${proposal_id} --field=${field} --filter=${ref_filter} --generate-catalogs")
    echo "Submitted reference-catalog job ${refcat_jobid} for ${target}"

    second_pass_dep=$(submit_pipeline_filter_jobs "${target}" "${proposal_id}" "${field}" "${filters_csv}" "${MODULES}" "--skip_step1and2" "${refcat_jobid}")

    local each_suffix="destreak_o${field}_crf"
    local -a filters=()
    IFS=',' read -r -a filters <<< "${filters_csv}"

    local -a catalog_jobids=()
    local filter
    for filter in "${filters[@]}"; do
        local range_hi
        if [[ "${ARRAY_RANGE}" =~ ^0-([0-9]+)$ ]]; then
            local total=$(( ${BASH_REMATCH[1]} + 1 ))
            range_hi=$(( (total + BUNDLE_SIZE - 1) / BUNDLE_SIZE - 1 ))
        else
            range_hi=$(( 24 / BUNDLE_SIZE - 1 ))
        fi
        cat_jobid=$(sbatch --parsable --dependency=afterok:${second_pass_dep} \
            --array="0-${range_hi}" \
            --job-name="webb-cat-${target}-${filter}-eachexp" \
            --output="${logdir}/webb-cat-${target}-${filter}-eachexp_%j-%A_%a.log" \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
            --wrap "${python_exec} ${catalog_script} --filternames=${filter} --modules=${MODULES} --proposal_id=${proposal_id} --target=${target} --each-exposure --each-suffix=${each_suffix} --daophot --skip-crowdsource --bundle-size=${BUNDLE_SIZE} --skip-if-done")
        catalog_jobids+=("${cat_jobid}")
        echo "Submitted catalog array ${cat_jobid} for ${target} ${filter}"
    done

    catalog_dep=$(IFS=:; echo "${catalog_jobids[*]}")
    merge_jobid=$(sbatch --parsable --dependency=afterok:${catalog_dep} \
        --job-name="webb-merge-${target}" \
        --output="${logdir}/webb-merge-${target}_%j.log" \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
        --wrap "${python_exec} ${merge_script} --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource --target=${target}")
    echo "Submitted merge job ${merge_jobid} for ${target}"
}

run_targets=("$@")
if [[ ${#run_targets[@]} -eq 0 ]]; then
    run_targets=(cloudef sgrc sgrb2 arches quintuplet sgra)
fi

for target in "${run_targets[@]}"; do
    case "${target}" in
        cloudef)
            submit_target_flow "cloudef" "${CLOUDEF_PROPOSAL_ID}" "${CLOUDEF_FIELD}" "${CLOUDEF_FILTERS}" "${CLOUDEF_REF_FILTER}"
            ;;
        sgrc)
            submit_target_flow "sgrc" "${SGRC_PROPOSAL_ID}" "${SGRC_FIELD}" "${SGRC_FILTERS}" "${SGRC_REF_FILTER}"
            ;;
        sgrb2)
            submit_target_flow "sgrb2" "${SGRB2_PROPOSAL_ID}" "${SGRB2_FIELD}" "${SGRB2_FILTERS}" "${SGRB2_REF_FILTER}"
            ;;
        arches)
            submit_target_flow "arches" "${ARCHES_PROPOSAL_ID}" "${ARCHES_FIELD}" "${ARCHES_FILTERS}" "${ARCHES_REF_FILTER}"
            ;;
        quintuplet)
            submit_target_flow "quintuplet" "${QUINTUPLET_PROPOSAL_ID}" "${QUINTUPLET_FIELD}" "${QUINTUPLET_FILTERS}" "${QUINTUPLET_REF_FILTER}"
            ;;
        sgra)
            submit_target_flow "sgra" "${SGRA_PROPOSAL_ID}" "${SGRA_FIELD}" "${SGRA_FILTERS}" "${SGRA_REF_FILTER}"
            ;;
        *)
            echo "Unknown target: ${target}. Supported targets: cloudef, sgrc, sgrb2, arches, quintuplet, sgra" >&2
            exit 2
            ;;
    esac

done
