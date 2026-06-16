#!/usr/bin/env bash
# ============================================================
# DEPRECATED -- legacy iter1-4 cataloging pipeline.
# Superseded by the manual-iteration pipeline (default):
#   submit_manual_pipeline.sh  -> jwst_gc_pipeline.photometry.cataloging
# Retired to 'legacy pipeline/'. Kept for reference only.
# ============================================================
echo "DEPRECATED: $(basename "$0") belongs to the legacy iter1-4 cataloging" >&2
echo "pipeline, superseded by submit_manual_pipeline.sh (manual-iteration path)." >&2
echo "This script has been retired and no longer runs. Recover from git if needed." >&2
exit 1
# Append the iter4resbgrefit final step after an iter3 run.
#
# Iteration coverage of this script (vs README "Iter1 / Iter2 / Iter3 /
# Iter4 cataloging cycle" section):
#   pipeline / iter1 / iter2 / iter3: NO  (must already exist on disk)
#   iter4: yes (--iteration-label=iter4resbgrefit;
#              additive refinement, does NOT touch iter1/2/3 outputs)
#
# Purely additive refinement pass (does NOT touch iter1/2/3 or the
# iter2residbg/iter3residbg products).  Per filter it:
#
#   1. Builds the MERGED iter3 residual mosaic (module='merged') from the
#      per-frame iter3 residuals (mosaic_each_exposure_residuals).
#   2. Median-smooths that merged mosaic into the merged background image
#      (make_iter3_residual_bgmaps.py).
#   3. Re-fits the EXACT per-frame iter3 catalog as seeds, on the
#      residual-bg-subtracted data, with the iter3 tight xy_bounds
#      (centroids move <1 px; flux free).  Writes a residual built from the
#      ORIGINAL (non-bg-subtracted) data.  --iteration-label=iter4resbgrefit
#      (outputs carry the _resbgsub token too).
#   4. Merges the per-frame iter4resbgrefit catalogs.
#
# Stages chain via SLURM afterok.  Stage 1 can be gated on external jobs
# (e.g. the in-flight sickle run) via --after.
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
script=${analysis_dir}/crowdsource_catalogs_long.py
bgmap_builder=${analysis_dir}/make_iter3_residual_bgmaps.py
BUNDLE_SIZE=${BUNDLE_SIZE:-1}
ITER_LABEL=iter4resbgrefit

usage() {
    cat <<'EOF'
Usage: run_iter4resbgrefit.sh --target <sickle|brick|cloudc|sgrb2> \
           [--after JOBID[:JOBID...]] [--skip-bgmap-build] [FILTER ...]

Appends the iter4resbgrefit refit step after iter3.  Assumes iter3
photometry and per-frame iter3 residuals/catalogs are already on disk.

Options:
  --target NAME        Required. One of: sickle, brick, cloudc, sgrb2.
  --after DEPS         Colon-separated SLURM job IDs to gate stage 1 on
                       (afterok), e.g. the in-flight run's jobs.
  --skip-mosaic        Reuse existing merged iter3 residual mosaics.
  --skip-bgmap-build   Reuse existing merged smoothed-bg images.
  FILTER...            Optional filter list. Default: per-target defaults.
EOF
}

target=""
after_deps=""
skip_mosaic=0
skip_bgmap_build=0
filters=()
while (($#)); do
    case "$1" in
        --target) target="$2"; shift 2 ;;
        --after) after_deps="$2"; shift 2 ;;
        --skip-mosaic) skip_mosaic=1; shift ;;
        --skip-bgmap-build) skip_bgmap_build=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) filters+=("$1"); shift ;;
    esac
done

if [[ -z "${target}" ]]; then
    echo "--target is required" >&2; usage; exit 2
fi

ref_filter_arg=""
case "${target}" in
    sickle)
        basepath=/orange/adamginsburg/jwst/sickle
        proposal_id=3958; field=007; each_suffix=destreak_o007_crf
        default_filters=(F187N F210M F335M F470N F480M)
        logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4); module_long=nrcb
        mem_short=32gb; mem_long=20gb
        ref_filter_arg="--ref-filter=f470n"
        ;;
    brick)
        basepath=/blue/adamginsburg/adamginsburg/jwst/brick
        proposal_id=2221; field=001; each_suffix=destreak_o001_crf
        default_filters=(F115W F182M F187N F200W F212N F356W F405N F410M F444W F466N)
        logdir=/blue/adamginsburg/adamginsburg/brick_logs
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4 nrca1 nrca2 nrca3 nrca4); module_long=merged
        mem_short=96gb; mem_long=64gb
        ;;
    cloudc)
        basepath=/blue/adamginsburg/adamginsburg/jwst/cloudc
        proposal_id=2221; field=002; each_suffix=destreak_o002_crf
        default_filters=(F182M F187N F212N F405N F410M F466N)
        logdir=/blue/adamginsburg/adamginsburg/cloudc_logs
        modules_short=(nrcb1 nrcb2 nrcb3 nrcb4 nrca1 nrca2 nrca3 nrca4); module_long=merged
        mem_short=64gb; mem_long=48gb
        ;;
    sgrb2)
        basepath=/orange/adamginsburg/jwst/sgrb2
        proposal_id=5365; field=001; each_suffix=align_o001_crf
        default_filters=(F150W F182M F187N F210M F212N F300M F360M F405N F410M F466N F480M)
        logdir=/blue/adamginsburg/adamginsburg/logs/sgrb2_jwst/
        modules_short=(nrca1 nrca2 nrca3 nrca4 nrcb1 nrcb2 nrcb3 nrcb4); module_long=""
        mem_short=256gb; mem_long=96gb
        ;;
    *) echo "unknown target: ${target}" >&2; exit 2 ;;
esac

mkdir -p "${logdir}"
[[ ${#filters[@]} -eq 0 ]] && filters=("${default_filters[@]}")

is_sw_filter() {
    case "$1" in
        F115W|F150W|F182M|F187N|F200W|F210M|F212N) return 0 ;;
        *) return 1 ;;
    esac
}

# Module token of the whole-field iter3 residual mosaic used as the bg for a
# filter.  SW filters co-add four detectors -> 'merged'; LW filters use the
# long-wavelength module (e.g. sickle 'nrcb', brick 'merged').
bg_mosaic_module() {
    if is_sw_filter "$1"; then echo "merged"; else echo "${module_long}"; fi
}

# --- Stage 1: merged iter3 residual mosaic, one job per filter --------------
mosaic_dep_arg=""
# afterany (not afterok): --after is an ordering gate ("run after the rest of
# the run finishes"), not a data dependency -- the refit only needs the iter3
# residuals, which are already complete.
[[ -n "${after_deps}" ]] && mosaic_dep_arg="--dependency=afterany:${after_deps}"
mosaic_jobs=()
if [[ ${skip_mosaic} -eq 0 ]]; then
    for filter in "${filters[@]}"; do
        bg_mod=$(bg_mosaic_module "${filter}")
        [[ -z "${bg_mod}" ]] && { echo "no LW module for ${target} ${filter}; skipping" >&2; continue; }
        job=$(FILTER="${filter}" BASEPATH="${basepath}" ANALYSIS_DIR="${analysis_dir}" \
            PROPOSAL_ID="${proposal_id}" FIELD="${field}" MODULE="${bg_mod}" \
            sbatch --parsable ${mosaic_dep_arg} \
            --job-name=webb-mosaic-${target}-iter3-${filter}-${bg_mod} \
            --output=${logdir}/webb-mosaic-${target}-iter3-${filter}-${bg_mod}_%j.log \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=1 --nodes=1 --mem=64gb --time=8:00:00 \
            --wrap "${python_exec} -c \"import os,sys; sys.path.insert(0, os.environ['ANALYSIS_DIR']); import crowdsource_catalogs_long as c; c.mosaic_each_exposure_residuals(basepath=os.environ['BASEPATH'], filtername=os.environ['FILTER'], proposal_id=os.environ['PROPOSAL_ID'], field=os.environ['FIELD'], module=os.environ['MODULE'], residual_kind='iterative', iteration_label='iter3')\"")
        echo "Submitted mosaic job ${job} for ${target} ${filter} (module=${bg_mod})" >&2
        mosaic_jobs+=("${job}")
    done
else
    echo "Reusing existing merged iter3 residual mosaics."
fi

# --- Stage 2: merged smoothed-bg build (one job, after all mosaics) ---------
bg_dep=""
if [[ ${skip_bgmap_build} -eq 0 ]]; then
    bgmap_dep_arg=""
    if [[ ${#mosaic_jobs[@]} -gt 0 ]]; then
        bgmap_dep_arg="--dependency=afterok:$(IFS=:; echo "${mosaic_jobs[*]}")"
    elif [[ -n "${after_deps}" ]]; then
        bgmap_dep_arg="--dependency=afterany:${after_deps}"
    fi
    filter_args=""
    if [[ ${#filters[@]} -lt ${#default_filters[@]} ]]; then
        for f in "${filters[@]}"; do filter_args="${filter_args} --filter=${f}"; done
    fi
    bg_jobid=$(sbatch --parsable ${bgmap_dep_arg} \
        --job-name=webb-bgmap-iter3merged-${target} \
        --output=${logdir}/webb-bgmap-iter3merged-${target}_%j.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=16gb --time=2:00:00 \
        --wrap "${python_exec} ${bgmap_builder} --target=${target}${filter_args}")
    echo "Submitted merged-bgmap job ${bg_jobid} for ${target}"
    bg_dep="--dependency=afterok:${bg_jobid}"
else
    echo "Reusing existing merged smoothed-bg images."
    [[ ${#mosaic_jobs[@]} -gt 0 ]] && bg_dep="--dependency=afterok:$(IFS=:; echo "${mosaic_jobs[*]}")"
fi

# --- Stage 3: per-frame iter4resbgrefit array jobs --------------------------
refit_deps=()
for filter in "${filters[@]}"; do
    bg_mod=$(bg_mosaic_module "${filter}")
    [[ -z "${bg_mod}" ]] && continue
    if is_sw_filter "${filter}"; then
        mods=("${modules_short[@]}"); mem=${mem_short}
    else
        mods=("${module_long}"); mem=${mem_long}
    fi
    cat_args="--daophot --skip-crowdsource --use-iter3-residual-bg --resbg-mosaic-module=${bg_mod} --iteration-label=${ITER_LABEL} --postprocess-residuals"
    for module in "${mods[@]}"; do
        [[ -z "${module}" ]] && continue
        job=$(sbatch --parsable ${bg_dep:-} \
            --array=0-23 \
            --job-name=webb-cat-${target}-${ITER_LABEL}-${filter}-${module}-eachexp \
            --output=${logdir}/webb-cat-${target}-${ITER_LABEL}-${filter}-${module}-eachexp_%j-%A_%a.log \
            --account=astronomy-dept --qos=astronomy-dept-b \
            --ntasks=1 --nodes=1 --mem=${mem} --time=96:00:00 \
            --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${each_suffix} ${cat_args} --bundle-size=${BUNDLE_SIZE} --skip-if-done")
        echo "Submitted ${ITER_LABEL} array ${job} for ${target} ${filter} ${module} (bg=${bg_mod})" >&2
        refit_deps+=("${job}")
    done
done

# --- Stage 4: merge ---------------------------------------------------------
if [[ ${#refit_deps[@]} -gt 0 ]]; then
    merge_dep=$(IFS=:; echo "${refit_deps[*]}")
    merge_job=$(sbatch --parsable --dependency=afterok:${merge_dep} \
        --job-name=webb-cat-merge-${target}-${ITER_LABEL} \
        --output=${logdir}/webb-cat-merge-${target}-${ITER_LABEL}_%j.log \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --ntasks=1 --nodes=1 --mem=256gb --time=24:00:00 \
        --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=daoiterative --skip-crowdsource --target=${target} --iteration-label=${ITER_LABEL} --use-iter3-residual-bg ${ref_filter_arg}")
    echo "Submitted ${ITER_LABEL} merge ${merge_job} for ${target}"
fi

echo "DONE submitting ${ITER_LABEL} chain for ${target}."
