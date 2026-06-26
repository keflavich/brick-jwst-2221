#!/bin/bash
#SBATCH --job-name=w51-F2100W-PipelineMIRI
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/w51-F2100W-PipelineMIRI_%j.log

# FRESH PipelineMIRI re-reduction of w51 F2100W (prop 6151 obs 002).
# WHY: the existing w51 MIRI products are a stale May-2025 reduction
# (CAL_VER 1.17.1) made BEFORE the 2026-06-10 outlier_detection fix (skymatch
# subtract=True + gentler snr 30/25).  The old over-aggressive outlier flagging
# nukes 47-93% of each frame -> 35% NaN mosaic (per-frame NaN grows
# rate 9% -> cal 38% -> outlier 80% -> crf 93%).  Re-reducing with the CURRENT
# defaults (marshall_tuning OFF -> subtract=True, outlier snr '30.0 25.0')
# should restore coverage, as it did for brick F2550W.
# No --skip_download_for_existing: the F2100W uncal is absent here (and the prior
# uncal was flagged corrupt 2026-06-10), so download fresh from MAST.
cd /orange/adamginsburg/jwst/w51
# -s (skip_step1and2): the 8 _cal files already exist from the first attempt
# (which failed only at the Image3 align-copy, now fixed via shutil.copyfile);
# reuse them and just redo Image3 (tweakreg/skymatch subtract=True / outlier
# snr 30-25 / resample) -- the steps that actually fix the NaN.
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/PipelineMIRI.py \
    -f F2100W -d 002 -p 6151 -s
