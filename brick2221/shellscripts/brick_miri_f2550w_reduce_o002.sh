#!/bin/bash
#SBATCH --job-name=brick-F2550W-PipelineMIRI
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/brick-F2550W-PipelineMIRI_%j.log

# FRESH re-reduction of brick F2550W (prop 2221 obs 002).  The existing crf are
# CAL_VER 1.14.1 / 2024-05-07 (2-year-stale calibration); the rightmost mosaic
# tile shows artifacts/speckle.  Re-reduce with the current pipeline + good
# outlier params (marshall OFF -> subtract=True, snr 30/25), same as w51.
# EAST trim is now ADAPTIVE (per-frame glow detection) -- leave MIRI_TRIM_EAST
# unset (floor=0) so clean east edges (e.g. visit-001 _08/10/12101 covering the
# detector-gap notch) are preserved while glowing frames are trimmed.
cd /orange/adamginsburg/jwst/brick
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/PipelineMIRI.py \
    -f F2550W -d 002 -p 2221 -s
