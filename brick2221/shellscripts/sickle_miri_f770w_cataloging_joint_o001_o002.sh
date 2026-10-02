#!/bin/bash
#SBATCH --job-name=webb-cat-F770W-miri-joint-o001o002
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-F770W-miri-joint-o001o002_%j.log

# JOINT MIRI F770W cataloging of sickle prop 3958 obs 001 + obs 002 together
# (2026-06-19).  The two sickle MIRI tiles overlap on some bright/saturated
# stars; cataloging them in ONE run pools all 10 per-frame fits, so a star that
# is edge-starved / out-of-FOV in one obs (e.g. the super-saturated "A" at the
# southern edge of o002) is constrained by the other obs's frames where it sits
# mid-detector.  The joint detection coadd + residual/model i2d span both
# pointings -> the correct final MERGED product.
#
# --field=001-002 is the JOINT token: get_filenames expands it to glob BOTH
# obs's frames; everything downstream is already obs-agnostic (merge globs by
# vgroup*, the data_i2d / residual ResampleStep auto-unions the frame WCSs).
# Outputs carry the -o001-002 token, distinct from the per-obs products.
#
# WORKTREE (2026-06-19): MIRI satstar work now lives in the dedicated worktree
# jwst-gc-pipeline-wt-miri-joint (branch miri-joint-satstar) so we stop editing
# the shared jwst-gc-pipeline main and don't conflict with other active agents.
# PYTHONPATH prepends the worktree so `import jwst_gc_pipeline` resolves THERE
# (ahead of the editable-install .pth that points at main), and we run the
# worktree's copy of the script.
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint
export PYTHONPATH="$WT:$PYTHONPATH"
source /blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/miri_cataloging_gains.sh  # MIRI cataloging gains (2026-07)
cd /orange/adamginsburg/jwst/sickle
# PURGE cached per-frame satstar catalogs/models so EVERY run re-fits the
# saturated stars with the current code (2026-06-20).  load_or_make_satstar_catalog
# only overwrites when the outside-FOV reg has near-frame stars (overwrite=
# bool(outside_star_pixels)); with no/empty reg it LOADS the cache, so code
# changes to the satstar fitter silently never apply.  Deleting forces a remake.
rm -f F770W/pipeline/jw03958*_mirimage_o00[12]_crf*satstar_catalog.fits \
      F770W/pipeline/jw03958*_mirimage_o00[12]_crf*satstar_model*.fits \
      F770W/pipeline/jw03958*_mirimage_o00[12]_crf*satstar_flags*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/catalog_long.py" \
    --filternames=F770W --modules=mirimage --each-exposure \
    --proposal_id=3958 --field=001-002 --target=sickle \
    --each-suffix=o001_crf \
    --daophot --skip-crowdsource \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0

# refresh the A/B/C/D satstar diagnostic figure for the JOINT product
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/make_satstar_diagnostic.py o001-002
