#!/bin/bash
#SBATCH --job-name=cloudc-2526-G0-F770W-PipelineMIRI
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/cloudc-2526-G0-F770W-PipelineMIRI_%j.log

# Full PipelineMIRI (download + Detector1 + Image2 + Image3) for the prop 2526
# "G0" CMZ cloud-c filament F770W pointing (obs o021 / t013, RA 266.582
# Dec -28.632; ~5.4' from the 2221 cloudc field -- distinct pointing, same region
# tree).  PipelineMIRI maps proposal 2526 field 021 -> regionname 'cloudc' and
# uses cloudc's NIRCam f182m absolute refcat (added to
# REFERENCE_ASTROMETRIC_CATALOG_CANDIDATES_BY_FIELD; twomass last resort).
# Products land under /orange/adamginsburg/jwst/cloudc/F770W/pipeline/ named
# jw02526-o021_*.  Nothing downloaded yet, so NO --skip_download_for_existing.
cd /orange/adamginsburg/jwst/cloudc
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/PipelineMIRI.py \
    -f F770W -d 021 -p 2526
