#!/bin/bash
#SBATCH --job-name=cloudc-F2550W-PipelineMIRI
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/cloudc-F2550W-PipelineMIRI_%j.log

# Full PipelineMIRI (Detector1+Image2+Image3) for cloudc F2550W (2221 obs 001),
# 72 uncal exposures, using the new defaults validated on brick F2550W
# 2026-06-10: skymatch subtract=True + outlier snr='30.0 25.0'.

cd /orange/adamginsburg/jwst/cloudc
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/PipelineMIRI.py \
    -f F2550W -d 001 -p 2221 --skip_download_for_existing
