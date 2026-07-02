#!/bin/bash
#SBATCH --job-name=webb-cat-w51-F2100W-o002
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-w51-F2100W-o002_%j.log

# Per-obs MIRI F2100W cataloging of w51 (prop 6151) obs 002, reusing
# the existing per-exposure crf (no re-reduction).  Uses the worktree
# jwst-gc-pipeline-wt-miri-joint (branch miri-joint-satstar) -- the satstar code
# with spike-constrained TRUE amplitudes + fake-bright phantom gate validated on
# the sickle.  The merged data_i2d / residual / model are built from the crf by
# the cataloging run itself (ResampleStep); no prior image3 i2d required.
# Canonical MIRI PSFs (fovp101/512/1024) are symlinked into w51/psfs.
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint
export PYTHONPATH="$WT:$PYTHONPATH"
source /blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/miri_cataloging_gains.sh  # MIRI cataloging gains (2026-07)
cd /orange/adamginsburg/jwst/w51
# purge any stale satstar cache so the current worktree code re-fits
rm -f F2100W/pipeline/jw06151*_mirimage_*o002_crf*satstar_catalog.fits \
      F2100W/pipeline/jw06151*_mirimage_*o002_crf*satstar_model*.fits \
      F2100W/pipeline/jw06151*_mirimage_*o002_crf*satstar_flags*.fits \
      F2100W/pipeline/jw06151*_mirimage_*o002_crf*satstar_residual*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py" \
    --filternames=F2100W --modules=mirimage --each-exposure \
    --proposal_id=6151 --field=002 --target=w51 \
    --each-suffix=o002_crf \
    --daophot --skip-crowdsource \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
