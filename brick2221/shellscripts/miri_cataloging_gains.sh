#!/bin/bash
# Shared MIRI cataloging "gains" env block -- source this from every MIRI
# cataloging launcher so the 2026-07 improvements apply uniformly.  All are
# env-gated in the pipeline code (default OFF -> no effect on NIRCam / unset
# fields), so sourcing this only turns them on for MIRI runs that opt in.
# Developed + verified on cloudc 2526 F770W (by-eye truth 0 -> 27/28 + 3 edge,
# clean qfit).  See project_cloudc_f770w_satstar_gate_miscalib and
# project_miri_partialsat_divot.
#
# FIELD-AGNOSTIC mechanisms (physically correct for all MIRI):
#  - first-group SATURATED DQ: the cal/crf SATURATED flag marks any pixel
#    saturated in ANY ramp group; on bright emission that floods huge regions and
#    buries real point sources.  Only first-group-saturated pixels are truly
#    unrecoverable.  Recovers stars sitting on saturated emission.
export MIRI_FIRSTGROUP_SAT_DQ=1
#  - neighbour-robust prominence (satstar seed gate + daophot gate): 25th-pct
#    emission floor + lower-half spread, immune to a bright neighbour inflating
#    the annulus MAD.
export MIRI_SATSTAR_SEED_PROM_ROBUST=1
export MIRI_DAOPHOT_PROM_ROBUST=1
#  - progressive prominence schedule: STRICT on raw early rounds (m12/m3=HI) so
#    no emission seeds propagate, LOOSE on clean bg-subtracted m6 (=LO) to recover
#    faint stars.  Endpoints below.
export MIRI_PROM_SNR_PROGRESSIVE=1
export MIRI_PROM_SNR_HI=8
export MIRI_PROM_SNR_LO=3
#  - detector-edge detection margin (px): default 8 masks bright real stars near
#    the mosaic boundary; 3 recovers them (prominence gate + qfit vetting still
#    reject edge-glow false detections).
export MIRI_EDGE_DETECT_MARGIN=3
#
# BRIGHTNESS-DEPENDENT thresholds -- cloudc F770W surface-brightness-calibrated
# STARTING points; with first-group DQ the satstar finder sees only genuine
# bright first-group cores so these are safe, but re-tune per field/filter
# against a by-eye truth region if a field's stars are much brighter/fainter.
export MIRI_SATSTAR_SEED_CORE_MIN=250
export MIRI_SATSTAR_SEED_PROM_MIN=6
export MIRI_SATSTAR_SEED_CONC_MIN=1.1
#
# POST-FIT bright-phantom gate -- rejects spurious SUPER-BRIGHT satstars on
# saturated extended emission (W51 F770W: real stars + emission knots saturate
# into one connected ~1e5-px blob, so every PRE-fit metric ranks the phantom >= a
# real star).  Caught only post-fit: a very bright fit (flux>floor) whose model
# leaves a badly structured residual (ssr_ratio) OR whose flux was extrapolated
# far above its seed (flux/flux_init).  The flux floor is load-bearing -- it
# protects genuine deep-sat stars whose faint wing-seed gives a high ratio.
# Verified W51 F770W: flags 7/153, all emission phantoms, zero real-star loss.
# Floor is surface-brightness-dependent -> re-tune per field/filter.
export MIRI_SATSTAR_PHANTOM_FLUX_FLOOR=1e5
export MIRI_SATSTAR_PHANTOM_SSR_MAX=50
export MIRI_SATSTAR_PHANTOM_RATIO_MAX=50
#
# Merge dedup radius scaled to the PSF FWHM (fraction).  The default 0.10"
# min/max_offset is NIRCam-calibrated and far too tight for the broad MIRI PSF
# (F2550W FWHM 0.80"): daofind splits one star into many detections (~0.14"
# scatter) that survive un-merged, stack their models in the coadd -> deep
# over-subtraction "pockmarks" + a ~3x over-counted catalog.  0.5xFWHM (F2550W
# 0.40", F770W 0.13") merges only physically-unresolvable duplicates.  Per-filter
# MIRI FWHM table in merge_catalogs.py.  See project_miri_f2550w_dedup_pileup.
export MERGE_DEDUP_FWHM_FRAC=0.5
#
# Flat-topped saturated-core model.  STPSF is sharply peaked, but a charge-bled
# saturated core is a flat-topped PLATEAU, so amp*PSF under-subtracts the core
# (bright ring at r~3px; cloudc F770W "every saturated-core star undersubtracted")
# or, when the amplitude is inflated to clear it, over-subtracts (central pit).
# When enabled, an accepted in-FOV satstar's model is replaced inside a geometric
# core+shoulder footprint (radius sqrt(sat_area/pi)+SHOULDER_FWHM*FWHM) by the
# bg-subtracted DATA, driving the core residual to ~0 without touching the PSF
# wings or the reported flux.  MIRI-only; post-gate.  Verified cloudc F770W:
# brightest core residual 1601->226 (=bg), over-sub pit -184->0, bg untouched.
# See flattop_satstar_model + project notes.
export MIRI_SATSTAR_FLATTOP=1
export MIRI_SATSTAR_FLATTOP_SHOULDER_FWHM=2.0
export MIRI_SATSTAR_FLATTOP_PLATEAU_FRAC=0.15
#
# Pedestal-capped saturated-star mergedcat re-render.  A star saturated in only
# SOME frames is re-rendered in the UNsaturated frames at its merged (clipped)
# flux as a peaked point source, over-predicting it and gouging the large MIRI
# thermal-background pedestal into a negative hole (cloudc F2550W bright star at
# 17:46:17.01 -28:35:19.5: -711 below bg; positive-valued so the deep-pit mask
# misses it).  Cap that render to clip(base-bg_coarse,0) -> core residual = bg
# (flat, no hole), wings kept.  MIRI-only.  The render stamp is ALSO FWHM-scaled
# now (auto; MERGE_RENDER_FWHM_MULT default 3) so broad F2550W/F2100W wings are
# no longer clipped by the old 21px stamp.
export MERGE_SATSTAR_RENDER_CAP=1
