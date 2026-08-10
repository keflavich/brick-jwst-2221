#!/bin/bash
#SBATCH --job-name=webb-cat-F770W-miri-o002
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-F770W-miri-o002_%j.log

# Per-obs MIRI F770W cataloging of sickle (prop 3958) obs 002.
# WORKTREE (2026-06-20): the working saturated-star code (spike-constrained TRUE
# amplitudes, large fovp1024 PSF, uncapped, DQ-NaN-mask -- the code that makes A/B
# subtract correctly in the JOINT product) lives ONLY in the worktree
# jwst-gc-pipeline-wt-miri-joint (branch miri-joint-satstar; 30 fix-markers vs 17
# in main).  The OLD o002_pruned launcher ran the MAIN repo -> never got these
# fixes -> A/B stayed unsubtracted in the o002 product.  Use the worktree here too.
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint
export PYTHONPATH="$WT:$PYTHONPATH"
cd /orange/adamginsburg/jwst/sickle
# PURGE cached per-frame satstar catalogs/models so the saturated stars are
# re-fit with the current worktree code (load_or_make_satstar_catalog otherwise
# LOADS the stale cache when the outside-FOV reg is empty -> code changes never
# apply).  Deleting forces a remake.
rm -f F770W/pipeline/jw03958*_mirimage_o002_crf*satstar_catalog.fits \
      F770W/pipeline/jw03958*_mirimage_o002_crf*satstar_model*.fits \
      F770W/pipeline/jw03958*_mirimage_o002_crf*satstar_flags*.fits \
      F770W/pipeline/jw03958*_mirimage_o002_crf*satstar_residual*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/catalog_long.py" \
    --filternames=F770W --modules=mirimage --each-exposure \
    --proposal_id=3958 --field=002 --target=sickle \
    --each-suffix=o002_crf \
    --daophot --skip-crowdsource \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0

# refresh the A/B/C/D satstar diagnostic figure
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/make_satstar_diagnostic.py o002
