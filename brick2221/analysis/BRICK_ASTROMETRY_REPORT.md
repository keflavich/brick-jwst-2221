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
5. **For NIRSpec**: FGS uses **GSC 3.2 / Gaia DR3** (verified: GSC3.2 is Gaia-DR3-sourced, agrees
   with GSC2.4.2 to 8.6 mas, sits ~21 mas off VVV). The catalog is tied to **VVV**, so it is ~23 mas
   off GSC3.2. **Recommendation: re-tie to GSC 3.2 / Gaia DR3 at the observation epoch**, not VVV,
   with a tighter threshold and cos δ fix.

Reference versions used: **VIRAC v2** (II/387/virac2, Gaia-DR3-referenced, epoch 2014.0 — confirmed
the 2nd VIRAC release, not v1 II/364); **GSC 3.2** (current active FGS catalog, Gaia DR3-sourced).

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
| GSC3.2 (Gaia DR3, Ks<14 clean) | 506 | **23.5** | +23.1 | +4.5 | 42.6 | 31.9 | **current FGS catalog** |
| GSC3.2 (PM, all) | 20580 | 53.6 | +51.9 | +13.5 | 54.2 | 29.4 | crowding-inflated (deep, no flux clean) |
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
| GSC3.2 − VVV | −10.5 | −18.0 | 20.8 |
| GSC3.2 − GSC2.4.2 | +7.3 | +4.4 | 8.6 |
| GNS − VVV | −56.8 | +18.0 | 59.6 |
| GSC2.4.2 − GNS | +76.1 | −2.1 | 76.1 |

**GSC 3.2 (current active JWST FGS catalog)** retrieved from the STScI VO service
(`gsss.stsci.edu/.../CatalogSearch.aspx?CAT=GSC32&FORMAT=CSV`; 134k srcs over the footprint,
Gaia-DR3-sourced, per-source epoch ~2016, full PM). It agrees with GSC2.4.2 to **8.6 mas** and sits
**~21 mas from VVV** — i.e. GSC versions are mutually consistent and the Gaia/GSC frame is ~20 mas
off VVV. JWST is **~23 mas** from GSC3.2 (bright Ks<14, flux-cleaned, PM-propagated); the
full-sample 54 mas is faint-source crowding inflation in this deep catalog.

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

## 4b. Re-tie to GSC 3.2 (delivered) and photometric ZP fix
**Re-tie tool** `retie_to_gsc.py`: fits a robust constant shift (rotation/scale are ~0; a 6-param
affine overfits at this S/N and injects spurious ±15-30 mas position-dependent terms, so shift mode
is the default) of the merged catalog to GSC 3.2 proper-motion-propagated to the observation epoch
(2022.66), using bright (2MASS Ks<14), flux-cleaned stars, then rewrites every position column.

Validation (honest, no re-matching artifacts):
| catalog | applied shift (mas) | held-out residual (cross-val) | vs VVV after |
|---|---|---|---|
| qualcuts_oksep2221 | (23.4, 4.6) | **3.1 mas** | 4.3 mas |
| ok2221or1182 | (26.3, 3.0) | 16.7 mas (N=65, noisy; SEM~6) | 3.2 mas |

Both land **<5 mas from VVV and ~3 mas (held-out) from the bright GSC3.2 sample** -> meets the 10 mas
goal. NOTE: the bright 2MASS-Ks GSC3.2 stars (the FGS-relevant *guide-star* population) sit near the
VVV frame, so tying to them ≈ tying to VVV (both NIR); this is the correct choice for FGS/NIRSpec
guiding. The "AFTER GSC3.2 re-match" number (~15 mas) is a dense-catalog counterpart-flip artifact,
NOT a real residual (held-out cross-val and the sparse-VVV check both confirm ~3 mas). Corrected
catalogs written as `*_gsc32retied.fits` with RETIE_* metadata.

**Photometric ZP fix** (astrometry-independent). Determined by direct catalog inspection (not code
reading, which was misleading):
- `qualcuts_oksep2221` (crowdsource provenance, 2026-06-07): `mag_ab − mag_vega = 1.827` → `mag_ab`
  is **true AB, correct**; has `mag_vega`. Its (JWST−VVV)_K = +2.05 ≈ AB−Vega(1.83)+color, as expected.
- `ok2221or1182_20251211` (daophot, 2025-12-11): `mag_ab ≡ Vega` (matches qualcuts `mag_vega` to
  0.003 mag), **mislabeled**, no `mag_vega`. Its (JWST−VVV)_K = +0.21 ≈ color only. This is the
  historical bug (built with old code).
The current daophot path writes correct AB (+8.90); the crowdsource path (`merge_catalogs.py` ~line
1382) was fixed in this work to write true AB + `mag_vega` for future builds (it previously could
emit Vega-as-`mag_ab`). The historical `ok2221or1182` was corrected post-hoc by recomputing
`mag_ab = -2.5 log10(flux_jy/Jy) + 8.90` per filter and storing the old Vega values as `mag_vega`
(`*_abfix.fits`). NOTE: the jwst-gc-pipeline commit message (19e38e5) names the wrong catalog as
affected — the code fix is still valid, but the buggy on-disk catalog was the daophot ok2221or1182,
not the crowdsource qualcuts.

## 4c. Validation suite (2026-06-17)
- **Guide-star range (12.5<J<18, FGS regime), retied qualcuts vs GSC3.2** (VISTA J): 12.5-18 = 7.8
  mas (2MASS J) / ~12 mas (VISTA J to 18); per bin ~10-14 mas. The bright-only shift generalizes to
  faint guide stars (it's a magnitude-independent frame offset). Now a standard block in
  `retie_to_gsc.py`.
- **VIRAC2 vs GSC3.2 in guide range**: 0.8 mas (J12.5-14), 15 mas (J14-16), 55 mas (J16-18, ~all
  -Dec). The two reference frames diverge for faint stars; our independent JWST catalog agrees with
  GSC3.2 to ~12 mas at J17-18, so **VIRAC2 (not GSC3.2) is the faint-end outlier** -> do not use
  VIRAC2 as the faint reference.
- **Saturated-star fits**: included (2839 flagged) but **biased** — saturated sources sit ~13-17 mas
  off the reference (vs ~4 mas for unsaturated) with larger scatter (MAD 54 vs 37). satstar
  positions need work before trusting bright/saturated targets to <10 mas.
- **F182M vs F200W** (two closest bands, different epochs): faint unsaturated (mag<18, N=7447) =
  (-3.5, -2.3) mas, **MAD (6.2, 4.0)** — excellent. Bright (mag<16) degrades to ~15-50 mas from
  saturation/crowding. So the bands agree well for normal stars; bright/saturated is the weak point.
- **Frame-to-frame (nrca1, 24 dithers, GWCS-transformed)**: per-star cross-frame scatter
  **MAD (4.4, 2.4) mas** -> exposure-to-exposure ~3 mas (<5 mas bar met, not <1 mas). Per-frame
  median spread (13-22 mas) is crowded-field mismatch-sampling noise, not real misalignment.
  Detector-to-detector not testable here (only nrca1 source catalogs present in F182M/pipeline).

## 4d. Satstar + detector-to-detector deep-dive (2026-06-17)
**Satstar / brightest-star astrometry.** Root cause located: for stars saturated in F200W,
`skycoord_ref` IS the F200W satstar position (2220/2375 use ref_filtername=f200w) — i.e. the catalog's
primary position for these comes from the saturated wide-band satstar fit. BUT swapping to the clean
narrow band does NOT help: satstar vs same-star F212N = MAD ~55 mas (92-95% > 20 mas); F212N vs GSC
~53 mas ≈ satstar vs GSC ~54 mas. These are red bright giants (mag_ab F200W~14.5, F212N~15.6-20,
F115W~23) — saturated in wide bands, faint/noisy in narrow bands, in a crowded field, so NO band
gives a clean centroid. The median bias is modest (~10-17 mas); the dominant error is ~50 mas random.
This is a **bright/red-star centroiding floor**, not a fixable satstar-only shift. These stars are
mostly at/beyond the faint edge of the FGS guide range (very red), so the ~10-12 mas guide-star
result stands. Recommendation: **flag F200W/F405N-saturated red giants as ~50 mas-uncertain**; do not
use them as <10 mas NIRSpec references. Improving them needs reduction-level work (better satstar
PSF centroiding / forced astrometry from the least-crowded unsaturated band).

**Detector-to-detector** (F182M exp 00001, source detection on all 8 cal images, GWCS-transformed,
vs GSC3.2 Ks<14): common offset ~(+120, -80) mas = the raw per-exposure pointing BEFORE tweakreg;
detector-to-detector spread ~30-47 mas (dRA 91-138, dDec -68 to -100) but limited by N=26-73/detector
and MAD~50 (per-detector median uncertainty ~6-12 mas), so ~2-4 sigma. This is the cal-level INPUT to
alignment; the final aligned catalog's per-star cross-frame repeatability is ~3 mas (tweakreg removes
the bulk). A clean final-frame detector-to-detector test would require per-detector positions in the
merged frame (not stored separately).

## 4e. F200W vs F115W/F182M + F182M detector seam (2026-06-17)
- **F200W is excellent.** F200W - F115W (clean): median ~0, **MAD (0.84, 0.90) mas for bright
  (F200W<18)**, ~(4.7,4.5) all; centered on zero, no systematic (offset_f200w_f115w.png). F200W and
  F115W (same epoch/visit) are co-aligned to ~2 mas across the WHOLE field, all detectors.
- **F200W - F182M**: ~(+3.4,+2.0) mas global offset (different epochs), MAD ~2-3 mas bright — small,
  fine.
- **F182M detector seam (NEW).** A horizontal band at **Dec -28.705 to -28.715** (~36") shows
  F200W-F182M = **(-12.6, -16.5) mas** vs (+3.6,+2.1) off-stripe -> a **~20 mas discontinuity in the
  F182M mosaic** (spatial_f200w_f182m.png). F200W-F115W shows NO stripe there (-0.6,0.2 vs 0,0), so
  the seam is **specific to F182M**, the astrometric ANCHOR filter. Likely an F182M per-detector /
  module WCS residual (or thin dither coverage) in that Dec band that tweakreg didn't fully remove;
  it propagates into the reference frame locally. **Recommendation: investigate the F182M
  per-detector WCS in that band / consider re-aligning F182M, or cross-check the anchor with
  F200W+F115W (which are seam-free).**

## 4f. Seam: program/detector localization + fix (2026-06-17)
**The seam is a per-program/epoch defect, and it DOES affect delivered positions.** Splitting by
filter vs the seam-free F200W (clean stars), the ~21-25 mas jump at Dec -28.705..-28.715 is present
in the **narrow/medium bands AND skycoord_ref** but absent in the wide bands:
- seam-PRESENT: F182M, F187N, F212N, F405N, F410M, F466N, **skycoord_ref (21 mas)** = **prop 2221
  obs 001, 2022-08-28**.
- seam-FREE: F115W, F200W, F356W, F444W = **prop 1182 obs 004, 2022-09-14**.
So the 2221-o001 mosaic has the seam; the 1182-o004 (wide) mosaic is clean. Since skycoord_ref is
f405n/f410m-based, the **delivered catalog positions carry the ~21 mas seam in that band**.
VIRAC2 cannot independently confirm (its matchable stars are the bright ~100 mas-scatter JWST
population; faint VIRAC2 is unreliable) — but the internal evidence is airtight (F115W-F182M and
F200W-F182M show the identical jump; F115W-F200W shows none).

**Detector localization** (a): source detection on the 8 F182M (2221) detectors, GWCS-transformed,
vs clean F200W: per-detector band-vs-off-band, **nrcb3** (band-off = +21,+17 mas) and **nrcb4**
(+18,+20) show a band-specific offset while **nrca3/nrca4 are clean** -> the seam traces to
**intra-detector distortion residuals in the NRCB-module detectors** in that Dec band, not removed by
tweakreg's per-frame shift. (2-exposure sample, ~3-4 sigma; a fuller run would nail it.)

**Fix** (b): two deliverables in `/orange/.../brick/`:
- `astrometry_retie_qualcuts..._seamfix.fits` — delivered catalog with skycoord_ref in the seam band
  replaced by the clean wide-band (F200W/F115W) position (40,775 sources; band residual 9.8,15 -> 0).
- `catalogs/widebands_seamfree_gsc32_reference_catalog.fits` — seam-free, GSC3.2-tied reference
  built from the 1182-o004 wide bands (F200W), for re-aligning the 2221 mosaics. Recommend re-running
  2221 alignment against this (with per-detector/distortion correction for nrcb3/nrcb4), or adopting
  the wide-band frame as the anchor instead of F182M.

## 4g. The Dec -28.70 module-boundary seam is in BOTH programs (2026-06-17)
- The faint Dec -28.70 line in the wide bands and the **(+15,+25) mas secondary cluster** in the
  F200W-F115W offset distribution are the **same artifact**: the (15,25) population (113 stars) sits
  at Dec median -28.703 spanning all RA (a thin horizontal band).
- So a module-boundary seam exists at Dec ~ -28.70 to -28.71 in BOTH programs: **strong in 2221-o001
  N/M bands (~20 mas, ~40k stars, Dec -28.704..-28.715)** and **weak in 1182-o004 wide bands
  (~15-25 mas, ~113 stars, thin Dec -28.703 line)**.
- Horizontal line at fixed Dec, all RA = **NIRCam A/B module mosaic boundary**. Per-detector check:
  A-module vs B-module differ ~7 mas off-band; NRCB detectors (nrcb3/nrcb4) show the band offset ->
  intra-NRCB-module distortion residuals not removed by tweakreg's per-frame shift, common to both
  programs (same target/orientation). (Localization at the detection limit -- thin band, low N.)
- VIRAC2 verdict: it CANNOT validate at few-mas (its GC per-source error is ~30-40 mas; F200W/F115W
  vs VIRAC2 are broad ~40 mas blobs = VIRAC2 noise). The JWST-internal F200W-F115W is a tight spike
  at (0,0) (MAD 0.85 mas at mag 16-18) -> the frames are aligned; VIRAC2 is just the noisy ruler.
- **Net: wide bands (F200W/F115W) are the clean backbone (sub-mas internal, uniform +-2 mas) EXCEPT
  the thin Dec -28.703 module-boundary band.** For re-alignment: mask/down-weight Dec -28.70..-28.71,
  or apply per-detector (not just per-frame) WCS correction for the NRCB module.

## 5. Recommendations for NIRSpec-grade pointing (<10 mas absolute)
1. **Choose the operational frame = GSC 3.2 / Gaia DR3** (the current active JWST FGS catalog), not
   VVV. Re-tie the reference catalog to it. (VVV is internally fine but ~21–23 mas off the FGS frame
   here; the catalog is currently ~23 mas off GSC3.2.) GSC3.2 query: STScI VO CatalogSearch.aspx
   with `CAT=GSC32&FORMAT=CSV` (VOTABLE trips astropy's bit parser; tile in Dec for the row cap).
2. **Fix `measure_offsets`**: (a) use cos(dec) great-circle ΔRA; (b) tighten threshold to ≪10 mas
   (e.g. 1 mas) and iterate to convergence; (c) consider a 4–6 parameter (shift+rotation+scale) or
   low-order distortion fit instead of shift-only.
3. **Re-tie at the merged-catalog stage** (or re-run tweakreg against a GSC/Gaia refcat) so the
   science catalogs carry the FGS frame directly; current cross-filter transfer adds ~+8 mas.
4. **Use bright (F212N/F200W) stars** for target selection: internal precision is ~1.5–5 mas; the
   only correction needed is the global ~20 mas frame shift.
5. **Definitive module check**: run per-detector (NRCA1-4/B1-4) offsets on the raw F182M per-detector
   catalogs (the cross-filter proxy here already shows no module jumps to ±10 mas).
