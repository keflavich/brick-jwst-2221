#!/usr/bin/env bash
set -euo pipefail

export STPSF_PATH=/orange/adamginsburg/repos/webbpsf/data/
logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
mkdir -p "$logdir"

usage() {
    cat <<'EOF'
Usage: run_all_cataloging_sickle.sh [--local] [--local-max-parallel N] [FILTER ...]

Without arguments, submit cataloging jobs for all sickle filters.
Pass one or more filters to limit the run, for example:
  run_all_cataloging_sickle.sh F480M

Use --local to run the per-frame jobs in parallel on the current node instead of submitting SLURM array jobs.
Use --local-max-parallel to limit concurrent local frame jobs (default: 24).
EOF
}

run_mode=sbatch
local_max_parallel=24
filters=()
while (($#)); do
    case "$1" in
        --local)
            run_mode=local
            shift
            ;;
        --local-max-parallel)
            if [[ $# -lt 2 ]]; then
                echo "--local-max-parallel requires an integer argument" >&2
                exit 2
            fi
            local_max_parallel="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            filters+=("$1")
            shift
            ;;
    esac
done

if ! [[ "${local_max_parallel}" =~ ^[0-9]+$ ]] || [[ "${local_max_parallel}" -lt 1 ]]; then
    echo "--local-max-parallel must be a positive integer (got: ${local_max_parallel})" >&2
    exit 2
fi

if [[ ${#filters[@]} -eq 0 ]]; then
    filters=(F187N F210M F335M F470N F480M)
fi

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py
analysis_dir=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis
basepath=/orange/adamginsburg/jwst/sickle
each_suffix=destreak_o007_crf
proposal_id=3958
target=sickle
field=007
dao_catalog_jobids=()
local_catalog_jobs=0
BUNDLE_SIZE=${BUNDLE_SIZE:-4}

compute_array_range() {
    local filter="$1" module="$2" dao_args="$3" iter_label="$4"
    local iter_arg=""
    if [[ -n "${iter_label}" ]]; then iter_arg="--iteration-label=${iter_label}"; fi
    "${python_exec}" "${script}" \
        --filternames="${filter}" --modules="${module}" --each-exposure \
        --proposal_id="${proposal_id}" --target="${target}" --each-suffix="${each_suffix}" \
        ${dao_args} ${iter_arg} --bundle-size="${BUNDLE_SIZE}" --list-missing-tasks 2>/dev/null \
        | awk -F: '/^__MISSING_TASKS__:/{sub(/^__MISSING_TASKS__:/,""); print; found=1} END{if(!found) print ""}'
}

submit_local_catalog_array() {
    local filter="$1"
    local module="$2"
    local dao="$3"
    local mem="$4"

    local -a pids=()
    local -a pid_task_ids=()
    local -a failed_task_ids=()
    local task_id
    for task_id in $(seq 0 23); do
        while [[ ${#pids[@]} -ge ${local_max_parallel} ]]; do
            local first_pid="${pids[0]}"
            local first_task_id="${pid_task_ids[0]}"
            if ! wait "${first_pid}"; then
                failed_task_ids+=("${first_task_id}")
            fi
            pids=("${pids[@]:1}")
            pid_task_ids=("${pid_task_ids[@]:1}")
        done

        local task_log="${logdir}/webb-cat-sickle-${filter}-${module}-eachexp_local_${task_id}.log"
        (
            export SLURM_ARRAY_TASK_ID="${task_id}"
            export SLURM_ARRAY_TASK_COUNT=24
            export SLURM_JOB_NAME="webb-cat-sickle-${filter}-${module}-eachexp-local"
            export SLURM_CPUS_PER_TASK=1
            export OMP_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1
            export VECLIB_MAXIMUM_THREADS=1
            "${python_exec}" "${script}" --filternames="${filter}" --modules="${module}" --each-exposure --proposal_id="${proposal_id}" --target="${target}" --each-suffix="${each_suffix}" ${dao}
        ) >"${task_log}" 2>&1 &
        pids+=("$!")
        pid_task_ids+=("${task_id}")
    done

    echo "Launched 24 local tasks for ${filter} ${module} with args: ${dao} (max parallel: ${local_max_parallel})"
    while [[ ${#pids[@]} -gt 0 ]]; do
        local first_pid="${pids[0]}"
        local first_task_id="${pid_task_ids[0]}"
        if ! wait "${first_pid}"; then
            failed_task_ids+=("${first_task_id}")
        fi
        pids=("${pids[@]:1}")
        pid_task_ids=("${pid_task_ids[@]:1}")
    done

    if [[ ${#failed_task_ids[@]} -gt 0 ]]; then
        echo "Local tasks failed for ${filter} ${module} task IDs: ${failed_task_ids[*]}" >&2
        return 1
    fi

    echo "Completed local tasks for ${filter} ${module}"
}

submit_local_mosaic() {
    local filter="$1"
    local module="$2"
    local dao="$3"

    if [[ "${dao}" == *"--daophot"* ]]; then
        FILTER="${filter}" MODULE="${module}" BASEPATH="${basepath}" ANALYSIS_DIR="${analysis_dir}" "${python_exec}" -c "import os, sys; sys.path.insert(0, os.environ['ANALYSIS_DIR']); import crowdsource_catalogs_long as c; [c.mosaic_each_exposure_residuals(basepath=os.environ['BASEPATH'], filtername=os.environ['FILTER'], proposal_id='${proposal_id}', field='${field}', module=os.environ['MODULE'], residual_kind=kind, desat=False, bgsub=False, epsf=False, blur=False, group=False, pupil='clear') for kind in ('basic', 'iterative')]"
        echo "Completed local residual mosaics for ${filter} ${module}"
    fi
}

submit_catalog_job() {
    local filter="$1"
    local module="$2"
    local dao="$3"
    local mem="$4"
    local dependency_jobid="${5:-}"

    submitted_array_jobid=""

    if [[ "${run_mode}" == "local" ]]; then
        submit_local_catalog_array "${filter}" "${module}" "${dao}" "${mem}"
        submitted_array_jobid="local"
        return
    fi

    # Determine iteration label from dao args (iter2 is identified by
    # --iteration-label=iter2; otherwise iter1/plain).
    local iter_label_for_range=""
    if [[ "${dao}" == *"--iteration-label=iter2"* ]]; then
        iter_label_for_range="iter2"
    fi
    # Strip the iteration-label flag for range computation (we already passed
    # it via iter_label_for_range).
    local dao_for_range="${dao//--iteration-label=iter2/}"
    dao_for_range="${dao_for_range//--postprocess-residuals/}"
    local range
    range=$(compute_array_range "${filter}" "${module}" "${dao_for_range}" "${iter_label_for_range}")
    if [[ -z "${range}" ]]; then
        echo "No outstanding tasks for ${filter} ${module} ${dao}; skipping." >&2
        submitted_array_jobid=""
        return
    fi

    local -a dep_args=()
    if [[ -n "${dependency_jobid}" ]]; then
        dep_args+=(--dependency="afterok:${dependency_jobid}")
    fi

    submitted_array_jobid=$(sbatch --parsable "${dep_args[@]}" --array="${range}" --job-name=webb-cat-sickle-${filter}-${module}-eachexp --output=${logdir}/webb-cat-sickle-${filter}-${module}-eachexp_%j-%A_%a.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${each_suffix} ${dao} --bundle-size=${BUNDLE_SIZE} --skip-if-done")
    echo "Submitted array job ${submitted_array_jobid} (range=${range}) for ${filter} ${module} with args: ${dao}"
}

submit_mosaic_job() {
    # One finalize-only job covers both iter=None and iter=iter2.  Kept the
    # bgsub flag and iteration_label args for call-site compatibility but only
    # the first iter2 invocation per (filter, module, bgsub) actually submits:
    # subsequent calls are no-ops (detected via the ITER_LABEL sentinel).
    local filter="$1"
    local module="$2"
    local bgsub_flag="$3"
    local dep_ids="$4"
    local iteration_label="${5:-}"

    # Only submit on the iter2 call; the plain/iter=None call is folded in.
    if [[ -n "${iteration_label}" && "${iteration_label}" != "iter2" ]]; then
        return
    fi
    if [[ -z "${iteration_label}" ]]; then
        # The plain-iter call is now folded into the iter2 submission; skip.
        return
    fi

    local bgsub_arg=""
    local bgsub_tag=""
    if [[ "${bgsub_flag}" == "--bgsub" ]]; then
        bgsub_arg="--bgsub"
        bgsub_tag="-bgsub"
    fi

    if [[ "${run_mode}" == "local" ]]; then
        "${python_exec}" "${script}" --filternames="${filter}" --modules="${module}" --each-exposure --proposal_id="${proposal_id}" --target="${target}" --each-suffix="${each_suffix}" --daophot --skip-crowdsource ${bgsub_arg} --finalize-only --iteration-labels=,iter2
        echo "Completed local residual mosaics for ${filter} ${module} ${bgsub_flag} (both iter labels)"
        return
    fi

    local dep_arg=""
    [[ -n "${dep_ids}" ]] && dep_arg="--dependency=afterok:${dep_ids}"
    sbatch ${dep_arg} --job-name=webb-mosaic-sickle-${filter}-${module}${bgsub_tag} --output=${logdir}/webb-mosaic-sickle-${filter}-${module}${bgsub_tag}_%j.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${each_suffix} --daophot --skip-crowdsource ${bgsub_arg} --finalize-only --iteration-labels=,iter2"
    echo "Submitted folded residual mosaic job for ${filter} ${module} ${bgsub_flag} (both iter labels)"
}

submit_catalog_and_residual_mosaic() {
    local filter="$1"
    local module="$2"
    local dao="$3"
    local mem="$4"

    if [[ "${run_mode}" == "local" ]]; then
        submit_local_catalog_array "${filter}" "${module}" "${dao}" "${mem}"
        submit_local_mosaic "${filter}" "${module}" "${dao}"
        if [[ "${dao}" == *"--daophot"* && "${dao}" != *"--bgsub"* ]]; then
            local_catalog_jobs=1
        fi
        return
    fi

    local array_jobid
    array_jobid=$(sbatch --parsable --array=0-23 --job-name=webb-cat-sickle-${filter}-${module}-eachexp --output=${logdir}/webb-cat-sickle-${filter}-${module}-eachexp_%j-%A_%a.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "${python_exec} ${script} --filternames=${filter} --modules=${module} --each-exposure --proposal_id=${proposal_id} --target=${target} --each-suffix=${each_suffix} ${dao}")
    
    echo "Submitted array job ${array_jobid} for ${filter} ${module} with args: ${dao}"

    if [[ "${dao}" == *"--daophot"* && "${dao}" != *"--bgsub"* ]]; then
        dao_catalog_jobids+=("${array_jobid}")
    fi

    if [[ "${dao}" == *"--daophot"* ]]; then
        sbatch --dependency=afterok:${array_jobid} --job-name=webb-mosaic-sickle-${filter}-${module} --output=${logdir}/webb-mosaic-sickle-${filter}-${module}_%j.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 --wrap "FILTER=${filter} MODULE=${module} BASEPATH=${basepath} ANALYSIS_DIR=${analysis_dir} ${python_exec} -c \"import os, sys; sys.path.insert(0, os.environ['ANALYSIS_DIR']); import crowdsource_catalogs_long as c; [c.mosaic_each_exposure_residuals(basepath=os.environ['BASEPATH'], filtername=os.environ['FILTER'], proposal_id='${proposal_id}', field='${field}', module=os.environ['MODULE'], residual_kind=kind, desat=False, bgsub=False, epsf=False, blur=False, group=False, pupil='clear') for kind in ('basic', 'iterative')]\""
        echo "Will submit residual mosaic job after array job ${array_jobid} completes"
    fi
}

# NOTE: Sickle is NRCB-only (SUB640 subarray for proposal 3958 field 007)
# Only process NRCB modules; nrca data does not exist

short_mem=32gb
for filter in "${filters[@]}"; do
    case "${filter}" in
        F187N|F210M)
            plain_deps=()
            bgsub_deps=()
            plain_deps_iter2=()
            bgsub_deps_iter2=()
            for modnum in 1 2 3 4; do
                module=nrcb${modnum}
                for dao in "--daophot --skip-crowdsource"; do
                    for bgsub in " " "--bgsub"; do
                        cat_args="${dao}${bgsub:+ }${bgsub}"
                        submit_catalog_job "${filter}" "${module}" "${cat_args}" "${short_mem}"
                        if [[ "${dao}" == *"--daophot"* && -n "${submitted_array_jobid}" ]]; then
                            if [[ "${bgsub}" == "--bgsub" ]]; then
                                bgsub_deps+=("${submitted_array_jobid}")
                            else
                                plain_deps+=("${submitted_array_jobid}")
                            fi

                            iter2_args="${cat_args} --iteration-label=iter2 --postprocess-residuals"
                            submit_catalog_job "${filter}" "${module}" "${iter2_args}" "${short_mem}" "${submitted_array_jobid}"
                            if [[ "${bgsub}" == "--bgsub" ]]; then
                                bgsub_deps_iter2+=("${submitted_array_jobid}")
                            else
                                plain_deps_iter2+=("${submitted_array_jobid}")
                            fi
                        fi
                    done
                done
            done
            if [[ ${#plain_deps[@]} -gt 0 ]]; then
                plain_dep_ids=$(IFS=:; echo "${plain_deps[*]}")
                submit_mosaic_job "${filter}" "nrcb" "" "${plain_dep_ids}" ""
            fi
            if [[ ${#bgsub_deps[@]} -gt 0 ]]; then
                bgsub_dep_ids=$(IFS=:; echo "${bgsub_deps[*]}")
                submit_mosaic_job "${filter}" "nrcb" "--bgsub" "${bgsub_dep_ids}" ""
            fi
            if [[ ${#plain_deps_iter2[@]} -gt 0 ]]; then
                plain_dep_ids_iter2=$(IFS=:; echo "${plain_deps_iter2[*]}")
                submit_mosaic_job "${filter}" "nrcb" "" "${plain_dep_ids_iter2}" "iter2"
            fi
            if [[ ${#bgsub_deps_iter2[@]} -gt 0 ]]; then
                bgsub_dep_ids_iter2=$(IFS=:; echo "${bgsub_deps_iter2[*]}")
                submit_mosaic_job "${filter}" "nrcb" "--bgsub" "${bgsub_dep_ids_iter2}" "iter2"
            fi
            ;;
    esac
done

long_mem=20gb
for filter in "${filters[@]}"; do
    case "${filter}" in
        F335M|F470N|F480M)
            module=nrcb
            plain_deps=()
            bgsub_deps=()
            plain_deps_iter2=()
            bgsub_deps_iter2=()
            for dao in "--daophot --skip-crowdsource"; do
                for bgsub in " " "--bgsub"; do
                    cat_args="${dao}${bgsub:+ }${bgsub}"
                    submit_catalog_job "${filter}" "${module}" "${cat_args}" "${long_mem}"
                    if [[ "${dao}" == *"--daophot"* && -n "${submitted_array_jobid}" ]]; then
                        if [[ "${bgsub}" == "--bgsub" ]]; then
                            bgsub_deps+=("${submitted_array_jobid}")
                        else
                            plain_deps+=("${submitted_array_jobid}")
                        fi

                        iter2_args="${cat_args} --iteration-label=iter2 --postprocess-residuals"
                        submit_catalog_job "${filter}" "${module}" "${iter2_args}" "${long_mem}" "${submitted_array_jobid}"
                        if [[ "${bgsub}" == "--bgsub" ]]; then
                            bgsub_deps_iter2+=("${submitted_array_jobid}")
                        else
                            plain_deps_iter2+=("${submitted_array_jobid}")
                        fi

                        if [[ "${bgsub}" != "--bgsub" ]]; then
                            dao_catalog_jobids+=("${submitted_array_jobid}")
                        fi
                    fi
                done
            done

            if [[ ${#plain_deps[@]} -gt 0 ]]; then
                plain_dep_ids=$(IFS=:; echo "${plain_deps[*]}")
                submit_mosaic_job "${filter}" "${module}" "" "${plain_dep_ids}" ""
            fi
            if [[ ${#bgsub_deps[@]} -gt 0 ]]; then
                bgsub_dep_ids=$(IFS=:; echo "${bgsub_deps[*]}")
                submit_mosaic_job "${filter}" "${module}" "--bgsub" "${bgsub_dep_ids}" ""
            fi
            if [[ ${#plain_deps_iter2[@]} -gt 0 ]]; then
                plain_dep_ids_iter2=$(IFS=:; echo "${plain_deps_iter2[*]}")
                submit_mosaic_job "${filter}" "${module}" "" "${plain_dep_ids_iter2}" "iter2"
            fi
            if [[ ${#bgsub_deps_iter2[@]} -gt 0 ]]; then
                bgsub_dep_ids_iter2=$(IFS=:; echo "${bgsub_deps_iter2[*]}")
                submit_mosaic_job "${filter}" "${module}" "--bgsub" "${bgsub_dep_ids_iter2}" "iter2"
            fi
            ;;
    esac
done

if [[ ${#dao_catalog_jobids[@]} -gt 0 ]]; then
    merge_dep=$(IFS=:; echo "${dao_catalog_jobids[*]}")
    sbatch --dependency=afterok:${merge_dep} --job-name=webb-cat-merge-sickle --output=${logdir}/webb-cat-merge-sickle_%j.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 --wrap "${python_exec} ${analysis_dir}/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource --target=sickle"
    echo "Submitted sickle merge job after daophot catalog jobs: ${merge_dep}"
fi

if [[ "${run_mode}" == "local" && ${local_catalog_jobs} -eq 1 ]]; then
    echo "Local mode completed for at least one daophot job. Run merge_catalogs.py separately if you want the cross-filter merge now."
fi

