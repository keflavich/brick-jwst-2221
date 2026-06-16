#!/usr/bin/env bash
set -euo pipefail

logdir=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/
mkdir -p "$logdir"

python_exec=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
pipeline_script=/orange/adamginsburg/repos/brick-jwst-2221/brick2221/reduction/PipelineMIRI.py

proposal_id=3958
# MIRI fields in program 3958:
#   001=t001, 002=t002  -> the SICKLE  (land in /orange/.../jwst/sickle/)
#   003=t003            -> the BRICK   (land in /orange/.../jwst/brick/)
# NOTE: obs 003 is NOT the sickle -- it is the brick MIRI field.  Program
# 3958 is shared, so PipelineMIRI routes o003 to the brick/ tree (see
# field_to_reg_mapping in jwst_gc_pipeline/reduction/PipelineMIRI.py) to keep
# its images + catalogs out of sickle/ and avoid name clashes.  This runner
# can still submit all three obs; o003 simply lands under brick/.
fields=${FIELDS:-001,002,003}
filters=${FILTERS:-F770W,F1130W,F1500W}

# Optional explicit reference catalog path.
# Default inside PipelineMIRI.py prefers:
# /orange/adamginsburg/jwst/sickle/catalogs/pipeline_based_nircam-f210m_reference_astrometric_catalog.fits
reference_catalog=${REFERENCE_CATALOG:-}

# Set SKIP_STEP1AND2=1 to rerun from existing _cal files.
skip_step_arg=""
if [[ "${SKIP_STEP1AND2:-0}" == "1" ]]; then
    skip_step_arg="--skip_step1and2"
fi

refcat_arg=""
if [[ -n "$reference_catalog" ]]; then
    refcat_arg="--reference_catalog=${reference_catalog}"
fi

IFS=',' read -r -a field_list <<< "$fields"
IFS=',' read -r -a filter_list <<< "$filters"

for field in "${field_list[@]}"; do
    for filter in "${filter_list[@]}"; do
        sbatch \
            --job-name=webb-miri-sickle-${filter}-o${field} \
            --output=${logdir}/webb-miri-sickle-${filter}-o${field}-%j.log \
            --account=astronomy-dept \
            --qos=astronomy-dept-b \
            --ntasks=8 \
            --nodes=1 \
            --mem=192gb \
            --time=48:00:00 \
            --wrap "${python_exec} ${pipeline_script} --proposal_id=${proposal_id} --field=${field} --filternames=${filter} ${skip_step_arg} ${refcat_arg}"
    done
done
