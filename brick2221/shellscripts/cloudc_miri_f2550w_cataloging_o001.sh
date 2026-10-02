#!/bin/bash
#SBATCH --job-name=webb-cat-cloudc-F2550W-o001
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-cloudc-F2550W-o001_%j.log

# Per-obs MIRI F2550W cataloging of cloudc (prop 2221) obs 001, reusing crf.
# cloudc crf were PRODUCT-named; rename_cloudc_f2550w_crf_to_perexposure.py first
# copied them to per-exposure names (jw02221001001_*_mirimage_o001_crf.fits) by
# EXPSTART so get_filenames finds them.  Worktree satstar code; merged data_i2d
# built from crf by the cataloging run.  Canonical MIRI PSFs symlinked into cloudc/psfs.
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint
export PYTHONPATH="$WT:$PYTHONPATH"
source /blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/miri_cataloging_gains.sh  # MIRI cataloging gains (2026-07)
cd /orange/adamginsburg/jwst/cloudc
rm -f F2550W/pipeline/jw02221001001_*_mirimage_o001_crf*satstar_catalog.fits \
      F2550W/pipeline/jw02221001001_*_mirimage_o001_crf*satstar_model*.fits \
      F2550W/pipeline/jw02221001001_*_mirimage_o001_crf*satstar_flags*.fits \
      F2550W/pipeline/jw02221001001_*_mirimage_o001_crf*satstar_residual*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/catalog_long.py" \
    --filternames=F2550W --modules=mirimage --each-exposure \
    --proposal_id=2221 --field=001 --target=cloudc \
    --each-suffix=o001_crf \
    --daophot --skip-crowdsource \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
