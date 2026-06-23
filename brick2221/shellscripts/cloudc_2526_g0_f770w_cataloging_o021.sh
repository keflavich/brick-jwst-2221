#!/bin/bash
#SBATCH --job-name=webb-cat-cloudc2526-G0-F770W-o021
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-cloudc2526-G0-F770W-o021_%j.log

# Cataloging of the prop 2526 "G0" CMZ cloud-c filament F770W pointing (obs o021,
# reduced by cloudc_2526_g0_f770w_reduce.sh into the cloudc/ tree).  Worktree
# satstar code (spike amplitude + fake-bright gate + empty-frame guards).  The
# cataloging script's field_to_reg_mapping/obs_filters/nvisits carry 2526->cloudc.
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint
export PYTHONPATH="$WT:$PYTHONPATH"
cd /orange/adamginsburg/jwst/cloudc
rm -f F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_catalog.fits \
      F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_model*.fits \
      F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_flags*.fits \
      F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_residual*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py" \
    --filternames=F770W --modules=mirimage --each-exposure \
    --proposal_id=2526 --field=021 --target=cloudc \
    --each-suffix=o021_crf \
    --daophot --skip-crowdsource \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
