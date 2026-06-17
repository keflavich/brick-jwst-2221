# Brick NIRCam Astrometry: accuracy, error origin, and NIRSpec readiness

Author: automated analysis (Claude), 2026-06-16
Scripts: `astrometry_analysis.py` (VVV/Gaia/GNS), `brick_astrometry_diagnostics.py`
(PM-propagated Gaia/VIRAC2/GSC, internal repeatability, spatial maps).
Data: `/orange/adamginsburg/jwst/brick/astrometry_analysis_*` and `astrometry_diag_*`.

JWST Brick (prop 2221, obs 001) epoch = **2022-08-28 (jyear 2022.66)**, MJD 59819.

---

## TL;DR
1. **Internal/relative astrometry is excellent**: bright stars repeat to **~1.5 mas (F212N), ~4–5 mas (F200W)** across filters. The catalog is internally self-consistent; **no module/sector frame jumps** (cross-filter f405n−f182m map is flat to ±10 mas, median ~0). Internal precision meets the 10 mas goal for bright sources in short/medium bands.
2. **There is a real absolute frame offset of ~17–23 mas (RA-dominated, JWST East of reference), brightness-independent.** This exceeds the 10 mas target.
3. **The external reference frames disagree with each other by 23–76 mas** in this field (VVV vs Gaia/GSC ≈ 23 mas; VVV vs GNS ≈ 60 mas). Absolute 10 mas astrometry is **reference-limited**.
4. **Origin**: the F182M reference catalog was tied to **VVV** by `measure_offsets` (shift-only, raw ΔRA without cos δ, convergence threshold 0.01″=10 mas, flux-ratio cleaned) → it converged with ~9 mas residual by its own metric, ~21 mas by naive matching. The merged catalogs add cross-filter transfer (~+8 mas) → ~17–23 mas vs VVV.
5. **For NIRSpec**: FGS uses the **Gaia/GSC** frame, but the catalog is tied to **VVV**, which is ~23 mas off Gaia/GSC here. **Recommendation: re-tie to GSC2.4.x/Gaia DR3 (FGS frame), not VVV**, with a tighter threshold and cos δ fix.

---

## 1. Pipeline reference & astrometric fixing process
- Internal anchor filter: **F182M**.
- Reference catalog: `catalogs/pipeline_based_nircam-f182m_reference_astrometric_catalog.fits`
  (built 2026-04-16 from 229 F182M per-detector pipeline catalogs; cols skycoord/flux/RA/DEC).
- Alignment code: `jwst_gc_pipeline/photometry/measure_offsets.py` (`measure_offsets`), driven from
  `make_reference_from_pipeline_catalogs.py` (`refine_with_vvv`).
- The build's own metadata: `VVV_N_MATCH=3301`, `VVV_DRA_AS=0.0219″`, `VVV_DDEC_AS=0.0258″`, `VVV_ITER=2`.
- Second pass: tweakreg aligns mosaics to this refcat; merged catalogs take per-row positions from
  whichever filter is the per-source reference (mostly f405n/f410m/f444w), transferred to the
  F182M→VVV frame.

### Origin of the systematic offset (complete explanation)
`measure_offsets` has three properties that each leave residual:
1. **Convergence threshold = 0.01″ = 10 mas.** The loop exits once |median ΔRA| AND |median ΔDec|
   < threshold, so up to ~10 mas residual is left **by design**.
2. **Raw ΔRA without cos(dec)** (lines 33/76/79). It zeroes raw ΔRA, not great-circle; minor
   (cos δ=0.877 at δ=−28.7°) but real.
3. **Shift-only** (no rotation/scale/distortion). A linear fit of the refcat→VVV residual shows a
   small gradient (~9 mas across the ±2′ field, rotation ~71 µas/″), so a pure shift leaves a
   field-dependent term.
Plus: **flux-ratio cleaning matters a lot** — `measure_offsets` (flux-cleaned) reports the refcat
as ~9 mas from VVV; a naive 0.2–0.5″ mutual match (no flux selection) reports ~21 mas because
crowded-field mismatches skew the median. **The merged catalogs add cross-filter transfer**, landing
at ~17–23 mas vs VVV regardless of brightness.

---

## 2. Results (qualcuts_oksep2221, 0.2″, flux-cleaned where K available)

| reference | N | systematic vec (mas) | med dRA | med dDec | MAD dRA | MAD dDec | note |
|---|--:|--:|--:|--:|--:|--:|---|
| VVV (Ks) | 2176 | **17.1** | +17.1 | −0.6 | 41.5 | 44.3 | most reliable in GC |
| GSC2.4.2 (Gaia) | 1056 | **22.2** | +20.4 | +8.7 | 50.4 | 52.4 | FGS lineage frame |
| VIRAC2 (Ks, no PM) | 6627 | 36.5 | +22.5 | +28.7 | 40.2 | 40.1 | PM not applied |
| Gaia DR3 (no PM) | 279 | 46.0 | +41.7 | +19.5 | 25.6 | 59.5 | foreground/PM contaminated |
| Gaia DR3 (PM→2022.66) | 279 | 49.0 | +48.7 | +6.2 | 20.9 | 61.4 | PM fixes Dec, not RA |
| GNS (Ks) | 8954 | 101.8 | +96.6 | +32.2 | 36.0 | 24.0 | **GNS frame outlier** |

### Brightness dependence (vs VVV, F212N bins)
med dRA = +17…+23 mas in **every** bin from F212N=12 to 18 → the offset is a **frame shift, not a
faint-source artifact**. MAD shrinks for bright stars (23–27 mas) vs faint (44 mas), reflecting VVV's
own per-source error.

### Internal cross-filter repeatability (reference-independent; the true 10 mas test)
Per-filter position scatter about `skycoord_ref`, **bright** subset:

| filter | mag cut | N | MAD dRA | MAD dDec |
|---|---|--:|--:|--:|
| F212N | <15 | 752 | **1.7** | **1.5** |
| F200W | <14 | 87 | 4.6 | 3.9 |
| F410M | <15 | 872 | 18.7 | 14.7 |
| F405N | all | 361k | 168 | 169 |

Median per-filter offsets are ~0 (no internal systematic). Bright short/medium bands meet 10 mas;
long-wave narrow bands (F405N/F444W/F466N) are crowding/PSF-limited (15–200 mas).

---

## 3. Reference-frame disagreements (no JWST involved)
| pair | med dRA | med dDec | vec (mas) |
|---|--:|--:|--:|
| GSC2.4.2 − VVV | +23.1 | −4.4 | 23.5 |
| GNS − VVV | −56.8 | +18.0 | 59.6 |
| GSC2.4.2 − GNS | +76.1 | −2.1 | 76.1 |

The GC reference frames mutually disagree by 23–76 mas. **VVV↔Gaia/GSC ≈ 23 mas** is the key one for
NIRSpec. **GNS is a ~57 mas outlier vs VVV** (epoch ~2015 + GC bulk proper motion, and/or GNS
calibration in this field) — **do not use GNS as the absolute reference**. Gaia direct in the GC is
sparse (~280 matches) and proper-motion contaminated (foreground stars); GSC (bright Gaia subset)
is the usable Gaia-frame proxy and agrees with VVV at ~23 mas.

### Epoch effect (important for NIRSpec)
VIRAC2 reference epoch is **2014.0**, PMs are **absolute (Gaia DR3-tied)**, pmRA already includes
cos δ (confirmed from the VizieR ReadMe / Smith et al. 2025). Propagating VIRAC2 to the JWST epoch
(2022.66, dt=8.66 yr) **worsened** the agreement (no-PM 36 → PM 61 mas), adding a net median
(+14, +20) mas ≈ bulk PM (1.6, 2.4 mas/yr) × 8.66 yr. The reason: **the JWST catalog is tied to VVV,
whose reference epoch is also ~2014, so the catalog's absolute frame is effectively pinned to ~2014,
not the 2022.66 observation epoch.** Hence no-PM VIRAC (2014.0) is the correct comparison and agrees
with VVV/GSC (~22 mas in RA). Consequence: for current-epoch NIRSpec pointing, the VVV-tying carries
an **epoch error of order (GC bulk PM)×(years since 2014) ~ tens of mas for moving GC stars**, on top
of the frame offset. A Gaia/GSC/VIRAC2 tie *propagated to the observation epoch* avoids this.

---

## 4. Affected catalog versions
- `basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20251211.fits`: ~25 mas vs VVV
  (F115W sparse, 206 ref rows).
- `basic_merged_indivexp_photometry_tables_merged_qualcuts_oksep2221.fits`: ~17 mas vs VVV
  (F115W good, 5763 ref rows). **Photometric ZP anomaly**: median (JWST−VVV)_K = 2.0 mag here vs
  0.1–0.2 for the other catalog — a ~2 mag F212N zeropoint inconsistency to investigate separately.
- `pipeline_based_nircam-f182m_reference_astrometric_catalog.fits` (root): ~9 mas (flux-cleaned) /
  ~21 mas (naive) vs VVV. Built by an **older** code path (meta keys VVV_DRA_AS/VVV_ITER; the
  current `main()` writes different keys), so it predates current bootstrap improvements.
- Legacy: `2023-*_crowdsource_based_nircam-long_reference_*` and `jw01182_VVV_reference_catalog.*`.

---

## 5. Recommendations for NIRSpec-grade pointing (<10 mas absolute)
1. **Choose the operational frame = GSC2.4.x / Gaia DR3** (what FGS uses), not VVV. Re-tie the
   reference catalog to it. (VVV is internally fine but ~23 mas off the FGS frame here.)
2. **Fix `measure_offsets`**: (a) use cos(dec) great-circle ΔRA; (b) tighten threshold to ≪10 mas
   (e.g. 1 mas) and iterate to convergence; (c) consider a 4–6 parameter (shift+rotation+scale) or
   low-order distortion fit instead of shift-only.
3. **Re-tie at the merged-catalog stage** (or re-run tweakreg against a GSC/Gaia refcat) so the
   science catalogs carry the FGS frame directly; current cross-filter transfer adds ~+8 mas.
4. **Use bright (F212N/F200W) stars** for target selection: internal precision is ~1.5–5 mas; the
   only correction needed is the global ~20 mas frame shift.
5. **Definitive module check**: run per-detector (NRCA1-4/B1-4) offsets on the raw F182M per-detector
   catalogs (the cross-filter proxy here already shows no module jumps to ±10 mas).
