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
    echo "Targets: cloudef, sgrc, sgrb2, arches, quintuplet, sgra, gc2211" >&2
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
# ARRAY_RANGE is only used as a fallback before destreak files exist
# (first-pass scheduling).  Once the per-filter destreak files are on
# disk the catalog array's size is auto-detected from the actual file
# count (see submit_target_flow below); the fallback never determines
# how many tasks actually run.  Override per-filter sizing with
# ARRAY_RANGE_OVERRIDE if you need to.
ARRAY_RANGE=${ARRAY_RANGE:-0-47}
BUNDLE_SIZE=${BUNDLE_SIZE:-1}
# Per-source PSF fitting parallelism for the catalog arrays.  Default 8
# workers at 4GB/worker -> --cpus-per-task=8 --mem=32gb (matches the old
# serial 32gb but spreads compute over 8 cores).  Set PARALLEL_WORKERS=1
# to opt out (original serial behavior).
PARALLEL_WORKERS=${PARALLEL_WORKERS:-8}
MEM_PER_WORKER_GB=${MEM_PER_WORKER_GB:-4}
PARALLEL_CHUNK_SIZE=${PARALLEL_CHUNK_SIZE:-100}

set_target_defaults() {
    local name="$1"
    # DEF_MERGE_REF_FILTER = the filter used as astrometric reference
    # when merge_catalogs.py builds cross-filter matches.  Defaults to
    # f405n (correct for brick/sgrb2/sgrc-class targets that include
    # F405N).  Targets without F405N must override.
    DEF_MERGE_REF_FILTER=f405n
    case "${name}" in
        cloudef)
            DEF_PROPOSAL_ID=2092
            # Cloud E (obs 002, t001) and Cloud F (obs 005, t002) are two
            # adjacent NIRCam pointings that must be reduced + cataloged
            # together to produce a single merged cloudef catalog.  Field
            # 005 stays the "primary" field (used as metadata label by
            # mkref and for the FOV region build); FIELDS drives the
            # pipeline + cataloging loop.
            DEF_FIELD=005
            DEF_FIELDS=002,005
            DEF_FILTERS=F162M,F210M,F360M,F480M
            DEF_REF_FILTER=F210M
            DEF_MERGE_REF_FILTER=f360m
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
            DEF_MERGE_REF_FILTER=f212n
            ;;
        quintuplet)
            DEF_PROPOSAL_ID=2045
            DEF_FIELD=003
            DEF_FILTERS=F212N,F323N
            DEF_REF_FILTER=F212N
            DEF_MERGE_REF_FILTER=f212n
            ;;
        sgra)
            DEF_PROPOSAL_ID=1939
            DEF_FIELD=001
            DEF_FILTERS=F115W,F212N,F405N
            DEF_REF_FILTER=F212N
            ;;
        gc2211)
            # Asteroid survey proposal 2211 has 5 GC pointings.  The
            # default below covers obs=028 (F150W+F277W, 240 frames);
            # the run_full_pipeline_gc2211.sh wrapper overrides FIELD
            # and FILTERS per obs.  obs IDs: 023, 028, 046, 049, 050.
            DEF_PROPOSAL_ID=2211
            DEF_FIELD=028
            DEF_FILTERS=F150W,F200W,F277W
            DEF_REF_FILTER=F200W
            DEF_MERGE_REF_FILTER=f277w
            ;;
        *)
            echo "Unknown target: ${name}. Supported targets: cloudef, sgrc, sgrb2, arches, quintuplet, sgra, gc2211" >&2
            exit 2
            ;;
    esac
}

ensure_target_setup() {
    local name="$1"
    local proposal_id="$2"
    local field="$3"
    # Optional 4th arg: CSV of all fields for multi-pointing targets
    # (e.g. cloudef 002,005).  Falls back to single ``field`` so existing
    # call sites are unaffected.
    local fields_csv="${4:-${field}}"
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
    mkdir -p "${basepath}/psfs"

    local fov_file="${basepath}/regions_/nircam_${name}_fov.reg"
    if [[ ! -f "${fov_file}" ]]; then
        echo "No FOV region found for ${name}. Building from MAST i2d footprints..."
        "${python_exec}" "${fov_script}" --target="${name}" --proposal-id="${proposal_id}" --field="${fields_csv}"
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
            --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
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
    # Optional 6th arg: CSV of extra fields to also flow through the
    # pipeline + per-exposure cataloging (e.g. cloudef obs 002 + 005).
    # When set, pipeline submissions get --field=<csv> (the pipeline
    # itself iterates fields), mkref still uses the primary ${field} as a
    # metadata label, and the per-exposure catalog array is duplicated
    # per field (each_suffix=destreak_o<field>_crf is field-specific).
    local fields_csv="${6:-${field}}"

    ensure_target_setup "${name}" "${proposal_id}" "${field}" "${fields_csv}"

    first_pass_dep=$(submit_pipeline_filter_jobs "${name}" "${proposal_id}" "${fields_csv}" "${filters_csv}" "${MODULES}" "" "")

    refcat_jobid=$(sbatch --parsable --dependency=afterok:${first_pass_dep} \
        --job-name="webb-refcat-${name}" \
        --output="${logdir}/webb-refcat-${name}_%j.log" \
        --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
        --ntasks=1 --nodes=1 --mem=64gb --time=24:00:00 \
        --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${refcat_script} --target=${name} --proposal-id=${proposal_id} --field=${field} --filter=${ref_filter} --generate-catalogs")
    echo "Submitted reference-catalog job ${refcat_jobid} for ${name}"

    second_pass_dep=$(submit_pipeline_filter_jobs "${name}" "${proposal_id}" "${fields_csv}" "${filters_csv}" "${MODULES}" "--skip_step1and2" "${refcat_jobid}")

    local -a fields=()
    IFS=',' read -r -a fields <<< "${fields_csv}"
    local -a filters=()
    IFS=',' read -r -a filters <<< "${filters_csv}"

    local -a catalog_jobids=()
    local filter
    local fld
    for fld in "${fields[@]}"; do
    local each_suffix="destreak_o${fld}_crf"
    for filter in "${filters[@]}"; do
        # Bundle BUNDLE_SIZE exposures per task to cut queued tasks.
        # Auto-size the per-filter cat array from the actual destreak
        # file count: dither pattern + detector layout varies by target,
        # field, instrument, and wavelength, so the right value is just
        # "however many destreak FITS this filter wrote".  Resolution
        # order:
        #   1) ARRAY_RANGE_OVERRIDE env var (manual override)
        #   2) actual file count on disk (preferred)
        #   3) ARRAY_RANGE env fallback (used only when files don't yet
        #      exist, e.g. first-pass scheduling before pipeline runs)
        # The cat script silently skips indices beyond the actual file
        # count, so an oversize fallback is harmless apart from a few
        # idle queued tasks.
        local range_hi
        local n_files
        # Use shopt nullglob via a subshell so an empty match doesn't
        # trip set -e / pipefail (ls returns 1 on no-match, killing the
        # whole flow before cat arrays are submitted).
        n_files=$( (shopt -s nullglob; files=( "/orange/adamginsburg/jwst/${name}/${filter}/pipeline/"*"${each_suffix}.fits" ); echo ${#files[@]}) )
        if [[ -n "${ARRAY_RANGE_OVERRIDE:-}" && "${ARRAY_RANGE_OVERRIDE}" =~ ^0-([0-9]+)$ ]]; then
            local total=$(( ${BASH_REMATCH[1]} + 1 ))
            range_hi=$(( (total + BUNDLE_SIZE - 1) / BUNDLE_SIZE - 1 ))
        elif (( n_files > 0 )); then
            range_hi=$(( (n_files + BUNDLE_SIZE - 1) / BUNDLE_SIZE - 1 ))
        elif [[ "${ARRAY_RANGE}" =~ ^0-([0-9]+)$ ]]; then
            local total=$(( ${BASH_REMATCH[1]} + 1 ))
            range_hi=$(( (total + BUNDLE_SIZE - 1) / BUNDLE_SIZE - 1 ))
        else
            # Final-resort fallback if ARRAY_RANGE is malformed.  Caller
            # should fix the env var; we just keep submitting a sane
            # default rather than 0 tasks.
            range_hi=$(( 24 / BUNDLE_SIZE - 1 ))
        fi
        echo "Cat array for ${name} ${filter} obs ${fld}: n_files=${n_files} array=0-${range_hi} (bundle=${BUNDLE_SIZE})" >&2
        local cat_mem_gb=$(( MEM_PER_WORKER_GB * PARALLEL_WORKERS ))
        local parallel_args=""
        if (( PARALLEL_WORKERS > 1 )); then
            parallel_args="--parallel-workers=${PARALLEL_WORKERS} --parallel-chunk-size=${PARALLEL_CHUNK_SIZE}"
        fi
        cat_jobid=$(sbatch --parsable --dependency=afterok:${second_pass_dep} \
            --array="0-${range_hi}" \
            --job-name="webb-cat-${name}-${filter}-o${fld}-eachexp" \
            --output="${logdir}/webb-cat-${name}-${filter}-o${fld}-eachexp_%j-%A_%a.log" \
            --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
            --ntasks=1 --cpus-per-task=${PARALLEL_WORKERS} --nodes=1 --mem=${cat_mem_gb}gb --time=96:00:00 \
            --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ${python_exec} ${catalog_script} --filternames=${filter} --modules=${MODULES} --proposal_id=${proposal_id} --field=${fld} --target=${name} --each-exposure --each-suffix=${each_suffix} --daophot --skip-crowdsource --bundle-size=${BUNDLE_SIZE} --skip-if-done ${parallel_args}")
        catalog_jobids+=("${cat_jobid}")
        echo "Submitted catalog array ${cat_jobid} for ${name} ${filter} obs ${fld}"
    done
    done

    catalog_dep=$(IFS=:; echo "${catalog_jobids[*]}")
    local merge_ref_arg=""
    if [[ -n "${DEF_MERGE_REF_FILTER:-}" ]]; then
        merge_ref_arg="--ref-filter=${DEF_MERGE_REF_FILTER}"
    fi
    merge_jobid=$(sbatch --parsable --dependency=afterok:${catalog_dep} \
        --job-name="webb-merge-${name}" \
        --output="${logdir}/webb-merge-${name}_%j.log" \
        --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
        --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
        --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${merge_script} --merge-singlefields --modules=merged --indiv-merge-methods=dao,daoiterative --skip-crowdsource --target=${name} ${merge_ref_arg}")
    echo "Submitted merge job ${merge_jobid} for ${name}"
}

set_target_defaults "${target}"

PROPOSAL_ID=${PROPOSAL_ID:-${DEF_PROPOSAL_ID}}
FIELD=${FIELD:-${DEF_FIELD}}
FIELDS=${FIELDS:-${DEF_FIELDS:-${FIELD}}}
FILTERS=${FILTERS:-${DEF_FILTERS}}
REF_FILTER=${REF_FILTER:-${DEF_REF_FILTER}}
CRDS_PATH=${CRDS_PATH:-${DEFAULT_CRDS_PATH}}

if [[ ! -d "${CRDS_PATH}" ]]; then
    echo "Configured CRDS_PATH does not exist: ${CRDS_PATH}" >&2
    exit 1
fi

submit_target_flow "${target}" "${PROPOSAL_ID}" "${FIELD}" "${FILTERS}" "${REF_FILTER}" "${FIELDS}"
