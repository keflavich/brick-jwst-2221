#!/bin/bash
#SBATCH --job-name=webb-cat-cloudc2526-G0-F770W-o021
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
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

# Satstar seed-gate recalibration for the cloudc 2526 filament (faint saturated
# stars, real wing-ring cores 300-760 -- the Sickle-calibrated core>=1000 default
# discards them; 28 by-eye-real stars were missing from every catalog product).
# Lower the core floor + enable the neighbour-robust prominence so emission in
# the annulus doesn't crush real stars below the gate.  --deblend-satstars splits
# the giant merged DQ-saturated blob (15086 px) that buries ~10 of the 28.
# Starting values; tune against regions_/f770w_byeyereal_20260629.reg.
# See project_cloudc_f770w_satstar_gate_miscalib.
export MIRI_SATSTAR_SEED_PROM_ROBUST=1
export MIRI_SATSTAR_SEED_CORE_MIN=250
export MIRI_SATSTAR_SEED_PROM_MIN=6
export MIRI_SATSTAR_SEED_CONC_MIN=1.1
# First-group SATURATED DQ correction: the cal/crf SATURATED flag marks pixels
# saturated in ANY ramp group, flooding the bright filament (62705 px / one
# 16231-px DQ blob fusing many real stars) though the ramp fitter recovers their
# flux; only first-group-saturated pixels (720 px / 26 genuine cores) are truly
# unrecoverable.  Correcting it lets daophot fit the embedded stars-on-emission
# normally and leaves only the genuine bright cores to the satstar channel.
export MIRI_FIRSTGROUP_SAT_DQ=1
# Neighbour-robust daophot prominence gate: recovers faint point sources on
# bright emission that the median+MAD prominence (miri_prominence_snr=5) drops
# because a bright neighbour inflates the annulus MAD (cloudc: last 6 by-eye
# stars, prominence 1.7-4.8). Robust 25th-pct floor + lower-half spread.
export MIRI_DAOPHOT_PROM_ROBUST=1
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
    --deblend-satstars \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
