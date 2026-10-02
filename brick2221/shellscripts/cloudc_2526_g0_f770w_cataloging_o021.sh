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
# Progressive prominence gate: STRICT on the raw early rounds (m12/m3=8) so no
# emission seeds propagate, LOOSEN across iterations to the clean bg-subtracted
# residual (m6=3) to recover the faintest real stars (m12/m3=8, m4=5.5, m5=4.25,
# m6=3). Endpoints tunable via MIRI_PROM_SNR_HI / _LO.
export MIRI_PROM_SNR_PROGRESSIVE=1
export MIRI_PROM_SNR_HI=8
export MIRI_PROM_SNR_LO=3
# Per-frame detector-edge detection margin (px). Default 8 masks bright real
# stars near the mosaic boundary out of every covering frame (byeye pt5/6/7:
# 1196-3660, prominence 130-313, lost). Lower to recover them; the prominence
# gate + qfit vetting still reject edge-glow false detections.
export MIRI_EDGE_DETECT_MARGIN=3
# Post-fit bright-phantom gate: rejects spurious SUPER-BRIGHT satstars on
# saturated extended emission (flux>floor AND (ssr>50 OR flux/flux_init>50));
# every pre-fit metric ranks these emission knots >= a real star.  Verified W51
# F770W (7/153 dropped, zero real loss).  Floor is surface-brightness dependent
# -> re-tune.  See project_miri_bright_phantom_gate / keflavich/jwst-gc-pipeline#36.
export MIRI_SATSTAR_PHANTOM_FLUX_FLOOR=1e5
export MIRI_SATSTAR_PHANTOM_SSR_MAX=50
export MIRI_SATSTAR_PHANTOM_RATIO_MAX=50
# FWHM-scaled merge dedup: 0.10" is too tight for the broad MIRI PSF -> daofind
# split-detections survive un-merged, stack models in the coadd -> over-sub
# pockmarks + over-counted catalog.  0.5xFWHM merges only unresolvable dupes.
# See project_miri_f2550w_dedup_pileup / keflavich/jwst-gc-pipeline#44.
export MERGE_DEDUP_FWHM_FRAC=0.5
# Flat-topped saturated-core model: a charge-bled saturated core is a flat-topped
# plateau, so amp*PSF under-subtracts it (bright ring; "every saturated-core star
# undersubtracted" was reported for THIS field).  Replace the model inside a
# geometric core+shoulder footprint by the bg-subtracted data -> core residual ~0,
# wings + flux untouched.  MIRI-only, post-gate.  keflavich/jwst-gc-pipeline#(flattop).
export MIRI_SATSTAR_FLATTOP=1
export MIRI_SATSTAR_FLATTOP_SHOULDER_FWHM=2.0
export MIRI_SATSTAR_FLATTOP_PLATEAU_FRAC=0.15
cd /orange/adamginsburg/jwst/cloudc
rm -f F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_catalog.fits \
      F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_model*.fits \
      F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_flags*.fits \
      F770W/pipeline/jw02526021001_*_mirimage_*o021_crf*satstar_residual*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/catalog_long.py" \
    --filternames=F770W --modules=mirimage --each-exposure \
    --proposal_id=2526 --field=021 --target=cloudc \
    --each-suffix=o021_crf \
    --daophot --skip-crowdsource \
    --deblend-satstars \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
