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
