#!/usr/bin/env bash
set -euo pipefail

# Common end-to-end runner for one target (cloudef, sgrc, sgrb2, arches,
# quintuplet, sgra, gc2211 -- brick has its own historical runner).
#
# Iteration coverage of this script (vs README "Iter1 / Iter2 / Iter3 /
# Iter4 cataloging cycle" section):
#   pipeline (Detector1/Image2/Image3): yes (first-pass + refcat + second-pass)
#   iter1: yes (per-frame DAO catalogs, --each-exposure --daophot)
#   iter2: NO  (run submit_full_chain.sh or run_iter3_cataloging.sh next)
#   iter3: NO
#   iter4: NO
#
# Stages:
# 1) First-pass pipeline (no reference catalog)
# 2) Build reference catalog from first-pass products
# 3) Second-pass pipeline (skip step1/2, now with reference catalog)
# 4) Per-exposure iter1 cataloging (DAO)
# 5) Merge single-field iter1 catalogs
#
# Multi-obs targets must populate DEF_FIELDS (CSV).  cloudef sets
# DEF_FIELDS=002,005; gc2211 obs id is provided per-launch by
# run_full_pipeline_gc2211.sh via FIELD env var.

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <target>" >&2
    echo "Targets: cloudef, sgrc, sgrb2, arches, quintuplet, sgra, gc2211, wd1, wd2, w51" >&2
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
        wd1)
            # Westerlund 1 (Guarcello prop 1905, Cycle 1).  obs=001 is the
            # main cluster pointing.  obs=003 is a control field (CF) ~10
            # arcmin off-cluster, much shallower coverage (F115W+F277W).
            # Default to obs=001 for the full filter set.  Wd1 is at
            # l~339, b~-0.4 -- outside the deep Galactic Center, so Gaia
            # DR3 is the right astrometric reference (NOT GNS).
            DEF_PROPOSAL_ID=1905
            DEF_FIELD=001
            DEF_FILTERS=F115W,F150W,F164N,F187N,F200W,F212N,F277W,F323N,F405N,F444W,F466N
            DEF_REF_FILTER=F212N
            DEF_MERGE_REF_FILTER=f405n
            ;;
        w51)
            # W51 NIRCam imaging (Yoo prop 6151 obs 001).  W51 sits in
            # the disk (l~49, b~-0.4), NOT the Galactic Center, so Gaia DR3
            # is the right astrometric reference (was UKIDSS/VVV in prior runs).
            # Prop 6151 NIRCam (per user 2026-06-13): F140M F162M F182M F187N
            # F210M F335M F360M F405N F410M F480M.  Narrow medium filters
            # are pupil-encoded (F150W2+F162M, F444W+F405N, etc.); pipeline
            # passes --filternames= to PipelineRerunNIRCAM-LONG.py which
            # resolves the FILTER/PUPIL combo via MAST query.
            DEF_PROPOSAL_ID=6151
            DEF_FIELD=001
            DEF_FILTERS=F140M,F162M,F182M,F187N,F210M,F335M,F360M,F405N,F410M,F480M
            DEF_REF_FILTER=F187N
            DEF_MERGE_REF_FILTER=f405n
            ;;
        wd2)
            # Westerlund 2 (Guarcello prop 3523, Cycle 2).  obs=005 is
            # the main cluster; obs=003 is the control field.  Wd2 sits in
            # the Carina arm (l~284, b~-0.3), far from the GC -- Gaia DR3
            # is the correct astrometric reference.
            DEF_PROPOSAL_ID=3523
            DEF_FIELD=005
            DEF_FILTERS=F115W,F150W,F162M,F164N,F182M,F187N,F200W,F212N,F250M,F277W,F300M,F323N,F335M,F405N,F410M,F444W,F466N
            DEF_REF_FILTER=F212N
            DEF_MERGE_REF_FILTER=f405n
            ;;
        *)
            echo "Unknown target: ${name}. Supported targets: cloudef, sgrc, sgrb2, arches, quintuplet, sgra, gc2211, wd1, wd2, w51" >&2
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

    # Gaia-refcat targets (wd1, wd2, w51) already have a static
    # ``catalogs/gaia_refcat.fits`` written by ``build_gaia_refcat.py``;
    # the VVV/GNS-bootstrap refcat builder does not work outside the GC
    # (no GNS/VVV coverage in Carina/disk fields) and would fail with
    # "No sources found in J/A+A/653/A133/central".  Skip it and feed the
    # first-pass dependency straight into the second pass.
    case "${name}" in
        wd1|wd2|w51)
            echo "Skipping reference-catalog build for ${name} (Gaia DR3 refcat is static at catalogs/gaia_refcat.fits)"
            second_pass_dep=$(submit_pipeline_filter_jobs "${name}" "${proposal_id}" "${fields_csv}" "${filters_csv}" "${MODULES}" "--skip_step1and2" "${first_pass_dep}")
            ;;
        *)
            # GC fields inside the Galactic Center but OUTSIDE GALACTICNUCLEUS
            # (GNS, J/A+A/653/A133) coverage must bootstrap the refcat from VVV
            # only: the GNS query returns 0 sources there and hard-fails with
            # "No sources found in J/A+A/653/A133/central".  VVV covers them.
            # Verified outside GNS: cloudef (Cloud E/F at RA~266.47-266.65, Dec
            # ~-28.48--28.50; GNS only covers RA~266.51-266.56, Dec~-28.71--28.76)
            # and sgrc (documented in make_reference --skip-gns help).
            case "${name}" in
                cloudef|sgrc) refcat_skip_gns="--skip-gns" ;;
                *)            refcat_skip_gns="" ;;
            esac
            refcat_jobid=$(sbatch --parsable --dependency=afterok:${first_pass_dep} \
                --job-name="webb-refcat-${name}" \
                --output="${logdir}/webb-refcat-${name}_%j.log" \
                --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
                --ntasks=1 --nodes=1 --mem=64gb --time=24:00:00 \
                --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${refcat_script} --target=${name} --proposal-id=${proposal_id} --field=${field} --filter=${ref_filter} --generate-catalogs ${refcat_skip_gns}")
            echo "Submitted reference-catalog job ${refcat_jobid} for ${name} ${refcat_skip_gns:+(${refcat_skip_gns})}"

            second_pass_dep=$(submit_pipeline_filter_jobs "${name}" "${proposal_id}" "${fields_csv}" "${filters_csv}" "${MODULES}" "--skip_step1and2" "${refcat_jobid}")
            ;;
    esac

    local -a fields=()
    IFS=',' read -r -a fields <<< "${fields_csv}"
    local -a filters=()
    IFS=',' read -r -a filters <<< "${filters_csv}"

    local -a catalog_jobids=()
    local filter
    local fld
    for fld in "${fields[@]}"; do
    # Extended-emission fields run with do_destreak=False (see
    # PipelineRerunNIRCAM-LONG.py), so their per-exposure crf files are
    # named *_align_o<fld>_crf.fits, not *_destreak_o<fld>_crf.fits.
    # Cataloging must follow the same suffix or it globs the stale
    # destreaked crf (or nothing).
    local crf_kind="destreak"
    case "${name}" in
        w51|sickle|wd2) crf_kind="align" ;;
    esac
    local each_suffix="${crf_kind}_o${fld}_crf"
    for filter in "${filters[@]}"; do
        # Manual-iteration cataloging is the default pipeline now: the
        # --each-exposure phases run sequentially in a single in-process
        # job and CANNOT be split across a SLURM array (the catalog script
        # rejects --each-exposure + manual-iterations when
        # SLURM_ARRAY_TASK_ID is set).  So submit ONE job per filter, no
        # --array.  Per-source parallelism still comes from
        # --parallel-workers.  (If you need the old array-parallel
        # per-exposure path, add --legacy-iterations to the wrap command
        # and restore an --array=0-N sizing.)
        local n_files
        # Use shopt nullglob via a subshell so an empty match doesn't
        # trip set -e / pipefail (ls returns 1 on no-match).
        n_files=$( (shopt -s nullglob; files=( "/orange/adamginsburg/jwst/${name}/${filter}/pipeline/"*"${each_suffix}.fits" ); echo ${#files[@]}) )
        if (( n_files == 0 )); then
            echo "No ${each_suffix} files for ${name} ${filter} obs ${fld}; skipping cataloging" >&2
            continue
        fi
        echo "Cat (manual-iteration, single job) for ${name} ${filter} obs ${fld}: n_files=${n_files}" >&2
        # Per-worker memory covers the parallel per-exposure cataloging, but the
        # manual-iteration merge phase (Phase 1 ra/dec/flux stack) loads ALL
        # per-exposure catalogs into one process at once.  For SW filters with
        # many exposures that peaks ~96GB, far above MEM_PER_WORKER_GB*workers,
        # so floor the request at CAT_MIN_MEM_GB to avoid OOM at the merge.
        local cat_mem_gb=$(( MEM_PER_WORKER_GB * PARALLEL_WORKERS ))
        local cat_min_mem_gb=${CAT_MIN_MEM_GB:-128}
        if (( cat_mem_gb < cat_min_mem_gb )); then
            cat_mem_gb=${cat_min_mem_gb}
        fi
        local parallel_args=""
        if (( PARALLEL_WORKERS > 1 )); then
            parallel_args="--parallel-workers=${PARALLEL_WORKERS} --parallel-chunk-size=${PARALLEL_CHUNK_SIZE}"
        fi
        cat_jobid=$(sbatch --parsable --dependency=afterok:${second_pass_dep} \
            --job-name="webb-cat-${name}-${filter}-o${fld}-eachexp" \
            --output="${logdir}/webb-cat-${name}-${filter}-o${fld}-eachexp_%j.log" \
            --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
            --ntasks=1 --cpus-per-task=${PARALLEL_WORKERS} --nodes=1 --mem=${cat_mem_gb}gb --time=96:00:00 \
            --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ${python_exec} ${catalog_script} --filternames=${filter} --modules=${MODULES} --proposal_id=${proposal_id} --field=${fld} --target=${name} --each-exposure --each-suffix=${each_suffix} --daophot --skip-crowdsource --bundle-size=${BUNDLE_SIZE} --skip-if-done ${parallel_args}")
        catalog_jobids+=("${cat_jobid}")
        echo "Submitted catalog job ${cat_jobid} for ${name} ${filter} obs ${fld}"
    done
    done

    catalog_dep=$(IFS=:; echo "${catalog_jobids[*]}")
    local merge_ref_arg=""
    if [[ -n "${DEF_MERGE_REF_FILTER:-}" ]]; then
        merge_ref_arg="--ref-filter=${DEF_MERGE_REF_FILTER}"
    fi
    local merge_workers=${MERGE_WORKERS:-8}
    merge_jobid=$(sbatch --parsable --dependency=afterok:${catalog_dep} \
        --job-name="webb-merge-${name}" \
        --output="${logdir}/webb-merge-${name}_%j.log" \
        --account=astronomy-dept --qos=${SLURM_QOS:-astronomy-dept-b} \
        --ntasks=1 --cpus-per-task=${merge_workers} --nodes=1 --mem=128gb --time=96:00:00 \
        --wrap "CRDS_PATH=${CRDS_PATH} CRDS_SERVER_URL=https://jwst-crds.stsci.edu ${python_exec} ${merge_script} --merge-singlefields --modules=merged --indiv-merge-methods=dao,daoiterative --skip-crowdsource --target=${name} ${merge_ref_arg} --merge-workers=${merge_workers}")
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
