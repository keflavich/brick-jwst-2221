#!/usr/bin/env bash
# Submit the iter2-residbg + iter3-residbg cascade for one target.
#
# Workflow (assumes iter3 photometry + per-frame residuals are
# complete on disk):
#
#   1. Build per-frame ``*_iter3_daophot_iterative_residual_smoothed_bg.fits``
#      via brick2221/analysis/make_iter3_residual_bgmaps.py (3x3 median).
#   2. Submit per-frame array jobs that call crowdsource_catalogs_long.py
#      with ``--use-iter3-residual-bg --iteration-label=iter2residbg``
#      (uses iter1 basic per-frame catalog as the seed; same logic as
#      iter2 but with the residual-derived background subtraction).
#   3. Submit a per-target merge job for iter2-residbg gated on those.
#   4. Submit per-frame array jobs with
#      ``--use-iter3-residual-bg --iteration-label=iter3residbg
#      --seed-catalog=<union>`` (uses the cross-band union seed plus
#      residual bg).
#   5. Submit a per-target merge job for iter3-residbg.
#
# This script reuses the per-target config from
# run_iter3_cataloging.sh.
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
script=${analysis_dir}/crowdsource_catalogs_long.py
bgmap_builder=${analysis_dir}/make_iter3_residual_bgmaps.py
BUNDLE_SIZE=${BUNDLE_SIZE:-4}

# Per-target LW chunking + group-size cap.  Mirrors run_iter3_cataloging.sh
# so iter3residbg (which fits the same union seed) doesn't hit the same
# 96 h walltime wall on brick LW.  iter2residbg uses the per-frame iter1
# basic seed (much sparser) so chunking is unnecessary there; the cap is
# still helpful for residbg-iter2 in dense regions.
N_SEED_CHUNKS_LW_brick=${N_SEED_CHUNKS_LW_brick:-8}
N_SEED_CHUNKS_LW_sgrb2=${N_SEED_CHUNKS_LW_sgrb2:-4}
N_SEED_CHUNKS_LW_cloudc=${N_SEED_CHUNKS_LW_cloudc:-4}
N_SEED_CHUNKS_LW_sickle=${N_SEED_CHUNKS_LW_sickle:-1}
MAX_GROUP_SIZE_ITER3=${MAX_GROUP_SIZE_ITER3:-15}

usage() {
    cat <<'EOF'
Usage: run_residbg_cataloging.sh --target <sickle|brick|cloudc|sgrb2> [--skip-bgmap-build] [FILTER ...]

Runs the iter2-residbg + iter3-residbg cascade.  Assumes iter3
photometry and per-frame residuals are already on disk.

Options:
  --target NAME        Required. One of: sickle, brick, cloudc, sgrb2.
  --skip-bgmap-build   Don't rebuild *_iter3_..._residual_smoothed_bg.fits;
                       reuse what's already on disk.
  FILTER...            Optional list of filters to include.  Default: all
                       per-target defaults.
EOF
}

target=""
skip_bgmap_build=0
filters=()
while (($#)); do
    case "$1" in
        --target) target="$2"; shift 2 ;;
        --skip-bgmap-build) skip_bgmap_build=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) filters+=("$1"); shift ;;
    esac
done

if [[ -z "${target}" ]]; then
    echo "--target is required" >&2; usage; exit 2
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
        module_long=nrcb
        seed_path=${basepath}/catalogs/seed_union_iter3_sickle.fits
        mem_short=32gb; mem_long=20gb
        ;;
    brick)
        basepath=/blue/adamginsburg/adamginsburg/jwst/brick
        proposal_id=2221
        field=001
        each_suffix=destreak_o001_crf
        default_filters=(F115W F182M F187N F200W F212N F356W F405N F410M F444W F466N)
        logdir=/blue/adamginsburg/adamginsburg/brick_logs
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4 nrca1 nrca2 nrca3 nrca4)
        module_long=merged
        seed_path=${basepath}/catalogs/seed_union_iter3_brick.fits
        mem_short=96gb; mem_long=64gb
        ;;
    cloudc)
        basepath=/blue/adamginsburg/adamginsburg/jwst/cloudc
        proposal_id=2221
        field=002
        each_suffix=destreak_o002_crf
        default_filters=(F182M F187N F212N F405N F410M F466N)
        logdir=/blue/adamginsburg/adamginsburg/cloudc_logs
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4 nrca1 nrca2 nrca3 nrca4)
        module_long=merged
        seed_path=${basepath}/catalogs/seed_union_iter3_cloudc.fits
        mem_short=64gb; mem_long=48gb
        ;;
    sgrb2)
        basepath=/orange/adamginsburg/jwst/sgrb2
        proposal_id=5365
        field=001
        each_suffix=align_o001_crf
        default_filters=(F150W F182M F187N F210M F212N F300M F360M F405N F410M F466N F480M)
        logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
        modules_short=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4)
        module_long=""
        seed_path=${basepath}/catalogs/seed_union_iter3_sgrb2.fits
        mem_short=256gb; mem_long=96gb
        ;;
    *) echo "unknown target: ${target}" >&2; exit 2 ;;
esac

mkdir -p "${logdir}"
[[ ${#filters[@]} -eq 0 ]] && filters=("${default_filters[@]}")

# --- 1. Build per-frame smoothed-bg files -----------------------------------
bg_dep=""
if [[ ${skip_bgmap_build} -eq 0 ]]; then
    filter_args=""
    if [[ ${#filters[@]} -lt ${#default_filters[@]} ]]; then
        for f in "${filters[@]}"; do filter_args="${filter_args} --filter=${f}"; done
    fi
    bg_jobid=$(sbatch --parsable \
        --job-name=webb-bgmap-iter3-${target} \
        --output=${logdir}/webb-bgmap-iter3-${target}_%j.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=8gb --time=2:00:00 \
        --wrap "${python_exec} ${bgmap_builder} --target=${target}${filter_args}")
    echo "Submitted bgmap-builder job ${bg_jobid} for ${target}"
    bg_dep="--dependency=afterok:${bg_jobid}"
else
    echo "Reusing existing iter3 residual smoothed-bg files."
fi

# --- 2/4. Per-filter array submissions for residbg passes -------------------

submit_array() {
    local iteration_label="$1" seed_arg="$2" filter="$3" module="$4" mem="$5" \
          chunk_index="${6:--1}" n_chunks="${7:-1}"
    local cap_arg=""
    if [[ ${MAX_GROUP_SIZE_ITER3} -gt 0 ]]; then
        cap_arg=" --max-group-size=${MAX_GROUP_SIZE_ITER3}"
    fi
    local chunk_arg="" chunk_tag=""
    if [[ ${n_chunks} -gt 1 ]]; then
        chunk_arg=" --n-seed-chunks=${n_chunks} --seed-chunk-index=${chunk_index}"
        chunk_tag=$(printf -- "-chunk%02dof%02d" "${chunk_index}" "${n_chunks}")
    fi
    local cat_args="--daophot --skip-crowdsource --use-iter3-residual-bg --iteration-label=${iteration_label} --postprocess-residuals${seed_arg}${cap_arg}${chunk_arg}"
    local job
    job=$(sbatch --parsable ${bg_dep:-} \
        --array=0-23 \
        --job-name=webb-cat-${target}-${iteration_label}-${filter}-${module}${chunk_tag}-eachexp \
        --output=${logdir}/webb-cat-${target}-${iteration_label}-${filter}-${module}${chunk_tag}-eachexp_%j-%A_%a.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 \
        --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${each_suffix} ${cat_args} --bundle-size=${BUNDLE_SIZE} --skip-if-done")
    echo "Submitted ${iteration_label} array ${job} for ${target} ${filter} ${module}${chunk_tag}" >&2
    echo "${job}"
}

submit_merge() {
    local iteration_label="$1" deps="$2"
    local dep_arg=""
    [[ -n "${deps}" ]] && dep_arg="--dependency=afterok:${deps}"
    local logfile=${logdir}/webb-cat-merge-${target}-${iteration_label}_%j.log
    local job
    job=$(sbatch --parsable ${dep_arg} \
        --job-name=webb-cat-merge-${target}-${iteration_label} \
        --output=${logfile} \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=256gb --time=24:00:00 \
        --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=daoiterative --skip-crowdsource --target=${target} --iteration-label=${iteration_label}")
    echo "Submitted ${iteration_label} merge ${job} for ${target}"
}

# Loop: iter2-residbg (no seed; uses iter1 basic seed inferred), then iter3-residbg
for iter_label in iter2residbg iter3residbg; do
    seed_arg=""
    if [[ "${iter_label}" == "iter3residbg" ]]; then
        seed_arg=" --seed-catalog=${seed_path}"
    fi
    deps=()
    for filter in "${filters[@]}"; do
        # SW filters never chunk; LW chunks only for iter3residbg (which uses
        # the full union seed).  iter2residbg uses the iter1 basic seed,
        # which is much sparser, so chunking gains nothing there.
        n_chunks=1
        case "${filter}" in
            F115W|F150W|F182M|F187N|F200W|F210M|F212N)
                mods=("${modules_short[@]}"); mem=${mem_short} ;;
            *)
                mods=("${module_long}"); mem=${mem_long}
                if [[ "${iter_label}" == "iter3residbg" ]]; then
                    n_chunks_var="N_SEED_CHUNKS_LW_${target}"
                    n_chunks=${!n_chunks_var:-1}
                fi
                ;;
        esac
        for module in "${mods[@]}"; do
            [[ -z "${module}" ]] && continue
            if [[ ${n_chunks} -gt 1 ]]; then
                for ((cidx=0; cidx<n_chunks; cidx++)); do
                    j=$(submit_array "${iter_label}" "${seed_arg}" "${filter}" "${module}" "${mem}" \
                        "${cidx}" "${n_chunks}")
                    deps+=("${j}")
                done
            else
                j=$(submit_array "${iter_label}" "${seed_arg}" "${filter}" "${module}" "${mem}")
                deps+=("${j}")
            fi
        done
    done
    if [[ ${#deps[@]} -gt 0 ]]; then
        merge_dep=$(IFS=:; echo "${deps[*]}")
        submit_merge "${iter_label}" "${merge_dep}"
    fi
done

echo "DONE submitting residbg chain for ${target}."
