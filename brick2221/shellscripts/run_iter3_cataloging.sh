#!/usr/bin/env bash
# Submit the iter3 cross-band union-seeded photometry for one of the
# three NIRCam targets (sickle / brick / cloudc).  Assumes iter2 has
# already been run and its per-filter daoiterative cross-exposure
# merges exist (they're what feeds the union seed builder).
#
# The script:
#   1. Builds/refreshes ``seed_union_iter3_{target}.fits`` via
#      brick2221/analysis/build_union_seed_catalog.py
#   2. Submits per-frame array jobs for each filter with
#      --iteration-label=iter3 --seed-catalog=<union path>
#   3. Submits residual-mosaic jobs gated on the array jobs
#   4. Submits one merge job gated on every iter3 array
#   5. Submits forced PSF photometry on per-frame satstar residuals gated on
#      the iter3 array jobs (measures undetected sources in each filter)
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
script=${analysis_dir}/crowdsource_catalogs_long.py
seed_builder=${analysis_dir}/build_union_seed_catalog.py
BUNDLE_SIZE=${BUNDLE_SIZE:-4}

# Per-target LW chunking.  Splits the union seed catalog into N image-pixel
# tiles per LW frame (one SLURM array job per chunk index, gated together at
# merge time).  Set to 1 to disable chunking for a target.  brick LW iter3
# at full union-seed density (~430 k seeds/frame) does not finish in 96 h
# walltime, so brick uses 8-way; sgrb2/cloudc LW are dense enough to
# benefit from 4-way chunking.  sickle LW is small and runs fine in one
# pass.  SW filters are not chunked anywhere -- the issue is LW group
# blow-up, not SW seed count.
N_SEED_CHUNKS_LW_brick=${N_SEED_CHUNKS_LW_brick:-8}
N_SEED_CHUNKS_LW_sgrb2=${N_SEED_CHUNKS_LW_sgrb2:-4}
N_SEED_CHUNKS_LW_cloudc=${N_SEED_CHUNKS_LW_cloudc:-4}
N_SEED_CHUNKS_LW_sickle=${N_SEED_CHUNKS_LW_sickle:-1}

# Hard cap on the photutils SourceGrouper group size for iter3-class runs.
# Above ~15, the joint LevMar fit cost grows roughly cubically with no
# meaningful accuracy gain since NIRCam can rarely separate >5-10 truly
# overlapping sources anyway.  In sparse regions nothing changes.
MAX_GROUP_SIZE_ITER3=${MAX_GROUP_SIZE_ITER3:-15}

usage() {
    cat <<'EOF'
Usage: run_iter3_cataloging.sh --target <sickle|brick|cloudc|sgrb2> [--skip-seed-build] [--residual-peaks] [--bgsub] [FILTER ...]

Options:
  --target NAME        Required. One of: sickle, brick, cloudc, sgrb2.
  --skip-seed-build    Do not rebuild the union seed catalog; reuse existing file.
  --residual-peaks     Pass --residual-peaks to the seed builder to inject bright
                       peaks from iter3 residual mosaics as additional seeds.
  --iter2-merge-dep ID Optional slurm jobid; gate everything on that job.
  --bgsub              Also submit background-subtracted (bgsub) catalog jobs.
                       Off by default; plain photometry only is the default.
  FILTER...            Optional list of filters to include. Default: all.
EOF
}

target=""
skip_seed_build=0
residual_peaks=0
run_bgsub=0
iter2_merge_dep=""
filters=()
while (($#)); do
    case "$1" in
        --target)
            target="$2"; shift 2 ;;
        --skip-seed-build)
            skip_seed_build=1; shift ;;
        --residual-peaks)
            residual_peaks=1; shift ;;
        --bgsub)
            run_bgsub=1; shift ;;
        --iter2-merge-dep)
            iter2_merge_dep="$2"; shift 2 ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            filters+=("$1"); shift ;;
    esac
done

if [[ -z "${target}" ]]; then
    echo "--target is required" >&2
    usage
    exit 2
fi

case "${target}" in
    sickle)
        basepath=/orange/adamginsburg/jwst/sickle
        proposal_id=3958
        field=007
        each_suffix=destreak_o007_crf
        default_filters=(F187N F210M F335M F470N F480M)
        logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4)
        modules_long=(nrcb)
        module_long=nrcb
        seed_path=${basepath}/catalogs/seed_union_iter3_sickle.fits
        mem_short=32gb
        mem_long=20gb
        ;;
    brick)
        basepath=/blue/adamginsburg/adamginsburg/jwst/brick
        proposal_id=2221
        field=001
        each_suffix=destreak_o001_crf
        default_filters=(F115W F182M F187N F200W F212N F356W F405N F410M F444W F466N)
        logdir=/blue/adamginsburg/adamginsburg/brick_logs
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4 nrca1 nrca2 nrca3 nrca4)
        modules_long=(merged)
        module_long=merged
        seed_path=${basepath}/catalogs/seed_union_iter3_brick.fits
        mem_short=96gb
        mem_long=64gb
        ;;
    cloudc)
        basepath=/blue/adamginsburg/adamginsburg/jwst/cloudc
        # cloudc data is on proposal 2221 field 002 (per
        # field_to_reg_mapping in crowdsource_catalogs_long.py and
        # obs_filters in merge_catalogs.py).  The earlier values
        # (proposal_id=1182, field=004) collided with brick's
        # proposal-1182 identifier and made the script silently
        # skip cloudc -- compute_array_range got an empty list
        # because crowdsource_catalogs_long.py raised KeyError on
        # ``reg_to_field_mapping['cloudc']``.
        proposal_id=2221
        field=002
        each_suffix=destreak_o002_crf
        default_filters=(F182M F187N F212N F405N F410M F466N)
        logdir=/blue/adamginsburg/adamginsburg/cloudc_logs
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4 nrca1 nrca2 nrca3 nrca4)
        modules_long=(merged)
        module_long=merged
        seed_path=${basepath}/catalogs/seed_union_iter3_cloudc.fits
        mem_short=64gb
        mem_long=48gb
        ;;
    sgrb2)
        basepath=/orange/adamginsburg/jwst/sgrb2
        proposal_id=5365
        field=001
        each_suffix=align_o001_crf
        default_filters=(F150W F182M F187N F210M F212N F300M F360M F405N F410M F466N F480M)
        logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
        modules_short=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
        modules_long=(nrcalong nrcblong)
        module_long=""
        seed_path=${basepath}/catalogs/seed_union_iter3_sgrb2.fits
        mem_short=256gb
        mem_long=96gb
        ;;
    *)
        echo "unknown target: ${target}" >&2
        exit 2
        ;;
esac

mkdir -p "${logdir}"

if [[ ${#filters[@]} -eq 0 ]]; then
    filters=("${default_filters[@]}")
fi

# --- 1. Build union seed catalog --------------------------------------------

seed_dep=""
if [[ ${skip_seed_build} -eq 0 ]]; then
    if [[ -n "${iter2_merge_dep}" ]]; then
        seed_dep_args=(--dependency=afterok:${iter2_merge_dep})
    else
        seed_dep_args=()
    fi
    # The seed builder is cheap enough (<1 hour even for brick) to run as a
    # single-node slurm job.  Downstream array jobs depend on its completion.
    residual_peaks_arg=""
    [[ ${residual_peaks} -eq 1 ]] && residual_peaks_arg=" --residual-peaks"
    seed_jobid=$(sbatch --parsable "${seed_dep_args[@]}" \
        --job-name=webb-seed-iter3-${target} \
        --output=${logdir}/webb-seed-iter3-${target}_%j.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=32gb --time=4:00:00 \
        --wrap "${python_exec} ${seed_builder} --target=${target} --output=${seed_path}${residual_peaks_arg}")
    echo "Submitted seed-builder job ${seed_jobid} for ${target} -> ${seed_path}"
    seed_dep="--dependency=afterok:${seed_jobid}"
else
    if [[ ! -e "${seed_path}" ]]; then
        echo "--skip-seed-build specified but ${seed_path} does not exist" >&2
        exit 2
    fi
    echo "Reusing existing seed catalog ${seed_path}"
fi

# --- 2. Per-filter iter3 catalog arrays + residual mosaics ------------------

compute_array_range() {
    local filter="$1" module="$2" cat_args="$3" filt_each_suffix="$4"
    "${python_exec}" "${script}" \
        --filternames="${filter}" --modules="${module}" --each-exposure \
        --proposal_id="${proposal_id}" --target="${target}" --each-suffix="${filt_each_suffix}" \
        ${cat_args} --bundle-size="${BUNDLE_SIZE}" --list-missing-tasks 2>/dev/null \
        | awk -F: '/^__MISSING_TASKS__:/{sub(/^__MISSING_TASKS__:/,""); print; found=1} END{if(!found) print ""}'
}

# Submit one array job for a single (filter, module, bgsub_opt, chunk) combo.
# When chunk_index >= 0 and n_chunks > 1, the per-frame outputs are tagged
# with _chunkXXofYY and merge_catalogs.py vstacks them back together.
submit_catalog_array() {
    local filter="$1" module="$2" mem="$3" extra_dep="$4" bgsub_opt="$5" \
          filt_each_suffix="${6:-${each_suffix}}" filt_array_range="${7:-}" \
          chunk_index="${8:--1}" n_chunks="${9:-1}"
    local dep_args=()
    if [[ -n "${seed_dep}" ]]; then dep_args+=("${seed_dep}"); fi
    if [[ -n "${extra_dep}" ]]; then dep_args+=("--dependency=afterok:${extra_dep}"); fi
    local bgsub_arg="" bgsub_tag=""
    if [[ "${bgsub_opt}" == "bgsub" ]]; then bgsub_arg=" --bgsub"; bgsub_tag="-bgsub"; fi
    local cap_arg=""
    if [[ ${MAX_GROUP_SIZE_ITER3} -gt 0 ]]; then
        cap_arg=" --max-group-size=${MAX_GROUP_SIZE_ITER3}"
    fi
    local chunk_arg="" chunk_tag=""
    if [[ ${n_chunks} -gt 1 ]]; then
        chunk_arg=" --n-seed-chunks=${n_chunks} --seed-chunk-index=${chunk_index}"
        chunk_tag=$(printf -- "-chunk%02dof%02d" "${chunk_index}" "${n_chunks}")
    fi
    local cat_args="--daophot --skip-crowdsource --iteration-label=iter3 --postprocess-residuals --seed-catalog=${seed_path}${bgsub_arg}${cap_arg}${chunk_arg}"
    # Compute sparse array: only the bundled task indices that still need work.
    local range
    range=$(compute_array_range "${filter}" "${module}" "${cat_args}" "${filt_each_suffix}")
    if [[ -z "${range}" ]]; then
        echo "iter3 already complete for ${target} ${filter} ${module} (${bgsub_opt}${chunk_tag}); skipping submission." >&2
        echo ""
        return
    fi
    local job
    job=$(sbatch --parsable "${dep_args[@]}" \
        --array="${range}" \
        --job-name=webb-cat-${target}-iter3-${filter}-${module}${bgsub_tag}${chunk_tag}-eachexp \
        --output=${logdir}/webb-cat-${target}-iter3-${filter}-${module}${bgsub_tag}${chunk_tag}-eachexp_%j-%A_%a.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 \
        --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${filt_each_suffix} ${cat_args} --bundle-size=${BUNDLE_SIZE} --skip-if-done")
    echo "Submitted iter3 array ${job} for ${target} ${filter} ${module} (${bgsub_opt}${chunk_tag}) range=${range}" >&2
    echo "${job}"
}

submit_mosaic_job() {
    local filter="$1" module="$2" bgsub="$3" dep_ids="$4"
    local bgsub_tag="" bgsub_arg=""
    if [[ "${bgsub}" == "--bgsub" ]]; then bgsub_tag="-bgsub"; bgsub_arg="--bgsub"; fi
    local dep_arg=""
    if [[ -n "${dep_ids}" ]]; then dep_arg="--dependency=afterok:${dep_ids}"; fi
    sbatch ${dep_arg} \
        --job-name=webb-mosaic-${target}-iter3-${filter}-${module}${bgsub_tag} \
        --output=${logdir}/webb-mosaic-${target}-iter3-${filter}-${module}${bgsub_tag}_%j.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 \
        --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${each_suffix} --daophot --skip-crowdsource ${bgsub_arg} --finalize-only --iteration-labels=iter3"
}

all_iter3_jobids=()
for filter in "${filters[@]}"; do
    # Determine per-filter each_suffix (sgrb2 has two suffix types)
    if [[ "${target}" == "sgrb2" ]]; then
        case "${filter}" in
            F182M|F187N|F210M|F212N) current_each_suffix=destreak_o001_crf ;;
            *)                        current_each_suffix=align_o001_crf ;;
        esac
        case "${filter}" in
            F187N) current_array_range="0-47" ;;
            *)     current_array_range="0-23" ;;
        esac
    else
        current_each_suffix="${each_suffix}"
        current_array_range="0-23"
    fi

    # SW filters never chunk; LW filters get the per-target chunk count
    # (defaults: brick=8, sgrb2=4, cloudc=4, sickle=1).  n_chunks=1 means
    # a single submission per (filter, module) and no chunk token in
    # filenames.
    n_chunks=1
    case "${filter}" in
        F115W|F150W|F182M|F187N|F200W|F210M|F212N)
            # Short-wavelength filters: one array per detector module.
            mods=("${modules_short[@]}")
            mem=${mem_short}
            ;;
        *)
            # Long-wavelength filters: one array per long module.
            mods=("${modules_long[@]}")
            mem=${mem_long}
            n_chunks_var="N_SEED_CHUNKS_LW_${target}"
            n_chunks=${!n_chunks_var:-1}
            ;;
    esac
    plain_deps=()
    bgsub_deps=()
    local_bgsub_opts=("plain")
    [[ ${run_bgsub} -eq 1 ]] && local_bgsub_opts+=("bgsub")
    for module in "${mods[@]}"; do
        for bgsub_opt in "${local_bgsub_opts[@]}"; do
            if [[ ${n_chunks} -gt 1 ]]; then
                for ((cidx=0; cidx<n_chunks; cidx++)); do
                    job=$(submit_catalog_array "${filter}" "${module}" "${mem}" "" "${bgsub_opt}" \
                          "${current_each_suffix}" "${current_array_range}" \
                          "${cidx}" "${n_chunks}")
                    if [[ -z "${job}" ]]; then continue; fi
                    all_iter3_jobids+=("${job}")
                    if [[ "${bgsub_opt}" == "plain" ]]; then
                        plain_deps+=("${job}")
                    else
                        bgsub_deps+=("${job}")
                    fi
                done
            else
                job=$(submit_catalog_array "${filter}" "${module}" "${mem}" "" "${bgsub_opt}" \
                      "${current_each_suffix}" "${current_array_range}")
                if [[ -z "${job}" ]]; then continue; fi
                all_iter3_jobids+=("${job}")
                if [[ "${bgsub_opt}" == "plain" ]]; then
                    plain_deps+=("${job}")
                else
                    bgsub_deps+=("${job}")
                fi
            fi
        done
    done
    # Mosaic jobs.  sgrb2 SW: two mosaic families (nrca, nrcb); sgrb2 LW: one
    # per long module.  Other targets: one aggregate module.
    if [[ "${target}" == "sgrb2" ]]; then
        case "${filter}" in
            F115W|F150W|F182M|F187N|F200W|F210M|F212N)
                mosaic_mods=(nrca nrcb)
                ;;
            *)
                mosaic_mods=("${modules_long[@]}")
                ;;
        esac
    else
        agg_module="${module_long}"
        [[ "${mods[*]}" == "${modules_short[*]}" ]] && agg_module="${modules_long[0]:-${module_long}}"
        mosaic_mods=("${agg_module}")
    fi
    for agg_mod in "${mosaic_mods[@]}"; do
        if [[ ${#plain_deps[@]} -gt 0 ]]; then
            plain_dep_ids=$(IFS=:; echo "${plain_deps[*]}")
            submit_mosaic_job "${filter}" "${agg_mod}" "" "${plain_dep_ids}"
        fi
        if [[ ${#bgsub_deps[@]} -gt 0 ]]; then
            bgsub_dep_ids=$(IFS=:; echo "${bgsub_deps[*]}")
            submit_mosaic_job "${filter}" "${agg_mod}" "--bgsub" "${bgsub_dep_ids}"
        fi
    done
done

# --- 3. Merge job gated on every iter3 catalog array ------------------------

if [[ ${#all_iter3_jobids[@]} -gt 0 ]]; then
    merge_dep=$(IFS=:; echo "${all_iter3_jobids[*]}")
    merge_job=$(sbatch --parsable --dependency=afterok:${merge_dep} \
        --job-name=webb-cat-merge-${target}-iter3 \
        --output=${logdir}/webb-cat-merge-${target}-iter3_%j.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 \
        --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=daoiterative --skip-crowdsource --target=${target} --iteration-label=iter3")
    echo "Submitted iter3 merge job ${merge_job} for ${target}"
fi

# --- 4. Forced photometry gated on all iter3 array jobs ---------------------
# Runs per-frame matched-filter photometry on satstar residuals for every
# source in the iter3 union catalog that was undetected in a given filter.
# Satstar residuals are produced by the iter3 array jobs, so gate on those.

forced_phot_script=${analysis_dir}/forced_photometry_residuals.py
if [[ ${#all_iter3_jobids[@]} -gt 0 ]]; then
    forced_dep=$(IFS=:; echo "${all_iter3_jobids[*]}")
    forced_job=$(sbatch --parsable --dependency=afterok:${forced_dep} \
        --job-name=webb-forced-phot-${target} \
        --output=${logdir}/webb-forced-phot-${target}_%j.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=32gb --time=8:00:00 \
        --wrap "${python_exec} ${forced_phot_script} --target=${target}")
    echo "Submitted forced photometry job ${forced_job} for ${target}"
fi

echo "DONE submitting iter3 chain for ${target}."
