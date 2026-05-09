#!/usr/bin/env bash
set -euo pipefail

logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
mkdir -p "$logdir"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
pipeline_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/PipelineRerunNIRCAM-LONG.py

# Defaults for Sickle long-wave processing.
proposal_id=3958
field=007

# Sickle short-wave filters use detector-specific module names.
# sickle does not use nrca
modules_short=nrcb
modules_long=nrcb

# Set SKIP_STEP1AND2=1 in the environment to rerun from existing _cal files.
skip_step_arg=""
if [[ "${SKIP_STEP1AND2:-0}" == "1" ]]; then
    skip_step_arg="--skip_step1and2"
fi

for filter in F187N F210M F335M F470N F480M; do
    if [[ "$filter" == "F187N" || "$filter" == "F210M" ]]; then
        modules="$modules_short"
    else
        modules="$modules_long"
    fi

    sbatch \
        --job-name=webb-long-sickle-${filter} \
        --output=${logdir}/webb-long-sickle-${filter}-%j.log \
        --account=astronomy-dept \
        --qos=astronomy-dept-b \
        --ntasks=8 \
        --nodes=1 \
        --mem=256gb \
        --time=96:00:00 \
        --wrap "${python_exec} ${pipeline_script} --proposal_id=${proposal_id} --field=${field} --filternames=${filter} --modules=${modules} ${skip_step_arg}"
done
