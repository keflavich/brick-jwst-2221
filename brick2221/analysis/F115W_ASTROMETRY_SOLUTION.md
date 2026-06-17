# F115W astrometric frame solution (brick, obs 1182-o004)

Author: automated analysis, 2026-06-16. Epoch of F115W observation: **2022.70** (MJD 59836.66).
Scripts in `/orange/adamginsburg/jwst/brick/astrometry_diag/`. Goal: an accurate, documented
F115W frame for NIRSpec pointing, tied to the Gaia / JWST-GSC frame.

## 0. REFERENCE HIERARCHY (decision, 2026-06-17)

**Favor the Gaia-tied frames; deprecate GSC 3.2's GC-NIR positions.**

PRIMARY (Gaia frame, mutually consistent to ~5 mas):
- **Gaia DR3** — absolute standard (sparse/bright in the GC).
- **VIRAC2** (II/387) — dense NIR, tied to Gaia DR3 (~5–7 mas); the working reference (full
  contiguous footprint coverage). Adopted frame for all filters.
- **GNS DR2** (Nogueras-Lara et al. 2025) — Gaia-tied (~5 mas), deep central GC. **Catalog not yet
  released** (Dec 2025 methodology paper; full catalogue forthcoming; images on ESO archive). ADD
  when released for deeper central coverage / a third Gaia-tied check.

DEPRECATED for absolute astrometry (pre-Gaia NIR frames, do not anchor to these):
- GSC 3.2 faint GC positions: posSource=25 = VVV-photometric (2MASS-tied, ~31 mas off Gaia, verified
  MAD 1.3 mas vs VVV-DR4 = same survey) and posSource=27 = GNS **DR1** (VVV-tied, ~50 mas off Gaia;
  DR2 postdates GSC 3.2). Use GSC 3.2 only for its Gaia-backed bright tier (posSource=15).
- VVV DR4 (II/376) — 2MASS-tied, ~24 mas off Gaia.

Combined Gaia-tied refcat built: `catalogs/gaia_virac2_refcat_epoch2022.70.fits` (Gaia DR3 + VIRAC2
fill, epoch 2022.70). Use as the tweakreg abs_refcat for all filters.

## 1. Reference frames — which to trust

All references cross-matched in their native frame and propagated to epoch 2022.70 with their
own proper motions. Offsets are median(catalog − Gaia DR3), mas.

| reference | Vizier | tie to Gaia | N | dRA | dDec | per-source MAD | use |
|---|---|---|--:|--:|--:|--:|---|
| **Gaia DR3** | I/355 | (definition) | — | 0 | 0 | ~0.1 | absolute truth, but sparse/bright-only in GC |
| **GSC 2.4.2** | I/353 | **0.2, 1.3** | 1710 | 0.2 | 1.3 | **1.8** | JWST guide-star frame proxy (GSC 3.1 = Gaia DR3-based, operational since 2024) |
| **VIRAC2** | II/387 | **1.3, 5.4** | 1525 | 1.3 | 5.4 | 26 | **dense NIR working reference on the Gaia frame** |
| VVV DR4 | II/376 | **−22.4, 8.6** | 1102 | −22.4 | 8.6 | 25 | **DEPRECATED for astrometry — 2MASS-tied, ~24 mas off Gaia** |

**Key conclusion (answers the VVV–Gaia question):** the ~28 mas VVV–Gaia offset seen earlier is
real and *expected* — the VVV DR4 photometric catalogue (II/376/vvv4) is astrometrically tied to
**2MASS**, which is offset ~24 mas from Gaia in this field. It is **not** a problem with our data.
VIRAC2 (Smith+ 2025, II/387) re-derives VVV astrometry on the **Gaia DR3** frame and agrees with
Gaia to ~5 mas; GSC 2.4.2 agrees to <2 mas. **VIRAC2/Gaia/GSC are the same frame; VVV DR4 is not.**
VIRAC2 reference epoch is 2016.0 (same as Gaia DR3); "RAJ2000" denotes the coordinate system only.

## 2. Current state of the F115W (and F182M/F200W) catalogs

Merged first-iteration DAO catalogs (`*_merged_indivexp_merged_dao_basic.fits`) vs the Gaia frame
(via dense VIRAC2). The pipeline currently anchors all filters to an **F182M → VVV-DR4 (2MASS)**
reference catalog, so they inherit the 2MASS frame and disagree with each other:

| filter | vs VIRAC2 dRA,dDec | \|vec\| | vs Gaia \|vec\| | spatial structure |
|---|--:|--:|--:|---|
| F115W | 26, 38 | 46 | 34 | near-rigid offset, small local scatter |
| F182M | 33, 13 | 35 | 52 | near-rigid offset |
| F200W | 43, 71 | 83 | 92 | **large coherent distortion field (70–140 mas)** |

- The three filters disagree by up to **~58 mas (Dec)** even though F115W & F200W are the *same
  observation/epoch/field* — a pure reduction/alignment inconsistency, not proper motion.
- **F200W is the worst**: its residual vs the Gaia frame is a spatially-coherent distortion field
  (see `residual_maps_vs_virac2.png`), not a simple offset. F200W alignment needs rework before it
  can be used to validate F115W.

## 3. F115W internal vs absolute accuracy

- **Internal frame-to-frame repeatability is already excellent: ~1.5 mas** (`std_ra`/`std_dec` of
  sources with ≥3 exposures in the merged catalog). The merge applies precise per-frame rigid
  shifts; relative astrometry within the mosaic is ~1.5 mas.
- **Per-frame alignment to VIRAC2** (dense, Gaia frame; median 687 matches/frame): per-frame shift
  precision (SEM) **1.8 mas**; raw frame-to-frame offset spread ~11–13 mas (removed by the shift).
- **Absolute tie to Gaia after per-frame VIRAC2 alignment:**
  - faint DAO sources: **20 mas** residual vs Gaia (frame-spread ~5 mas)
  - this 20 mas is dominated by VIRAC2's own tie to Gaia (~5 mas) + crowding + the bright end of
    the Gaia-matchable sample. External catalogs cannot validate below ~20 mas here.

## 4. Saturated stars as anchors — assessed

The bright stars Gaia sees are saturated in F115W. Saturated-star PSF fits (`*_satstar_catalog`,
`skycoord_fit`):

- Raw frame WCS: **112 mas** from Gaia (and the merged catalog's saturated/`replaced_saturated`
  sources show the same — they bypass the merge's per-frame alignment).
- After applying the faint-star per-frame VIRAC2 shift: improves to **30 mas** vs Gaia, but
  **~25–30 mas residual remains even vs VIRAC2**, with large frame-to-frame scatter (25–33 mas).

**Verdict:** saturated-star fits are **not good enough as primary anchors** (~25–30 mas centroid
bias relative to the faint-star/Gaia frame). They help (112→30 mas with per-frame alignment) and
are useful as a cross-check, but the dense **faint** VIRAC2 network is the better anchor. Any
NIRSpec target that is itself a saturated star in F115W carries this ~25–30 mas position bias —
flag accordingly.

## 5. Do we resolve binaries / blends?

In the brick the field is so dense that ~85% of VIRAC2 sources have ≥4 F115W detections within the
VISTA resolution element (0.5″) — JWST vastly out-resolves VVV/VIRAC2/2MASS. However, blended vs
isolated VIRAC2 sources show the *same* match scatter (~48 vs ~50 mas), so **blending is not the
dominant driver** of the reference scatter; VIRAC2's intrinsic ~26–47 mas per-source astrometric
error is. Deblending therefore won't reconcile VVV-vs-Gaia; the systematic is the 2MASS frame.

## 6. Recommended F115W frame solution (for NIRSpec)

1. **Stop anchoring to VVV-DR4 (2MASS).** Build the absolute reference catalog from **Gaia DR3 /
   VIRAC2 (Gaia frame)**, propagated to epoch 2022.70. (`build_gaia_refcat.py` already exists;
   point the abs_refcat at Gaia/VIRAC2 instead of the F182M→VVV-DR4 product.)
2. **Per-frame align each F115W exposure to VIRAC2** (dense, NIR, Gaia frame). Solution saved:
   `f115w_virac2_perframe_shifts.ecsv` (per-frame correction, SEM ~1.8 mas/frame). This maps onto
   `merge_catalogs.shift_individual_catalog`'s offsets-table (Visit/Exposure/Module/Filter/dra/ddec)
   — confirm the RAOFFSET sign/unit convention, then re-run the merge to produce a Gaia-frame
   catalog non-destructively.
3. **Verify against Gaia DR3 and GSC 2.4.2 / GSC 3.1.** GSC 3.1 (operational JWST guide-star
   catalog, Gaia DR3-based) is the ultimate reference; GSC 2.4.2 confirms it is the Gaia frame to
   <2 mas. Pull GSC 3.1 from MAST/STScI for the final sign-off.
4. **Do not use saturated-star fits as primary anchors** (§4); use them only as a bright cross-check.
5. **Rework F200W** before cross-validating (§2).

## 7. Accuracy budget (F115W, achievable)

| term | value |
|---|--:|
| internal relative (frame-to-frame, within mosaic) | **~1.5 mas** |
| per-frame absolute shift precision (to VIRAC2/Gaia) | ~1.8 mas |
| absolute tie to Gaia/GSC (faint network, ref-limited) | **~20 mas** |
| saturated/bright-star centroid bias (if target is bright) | ~25–30 mas |

Relative pointing between F115W targets is few-mas; the absolute Gaia/GSC zero-point is limited to
~20 mas by the available reference stars in this crowded, reddened field. To beat 20 mas would
require a dedicated bright-but-unsaturated star sample tied directly to GSC 3.1 / Gaia at the JWST
epoch.

## 8. DELIVERED (2026-06-17): Gaia-frame refcat + re-anchored F115W catalog

**GSC 3.2 confirmed = Gaia frame.** Pulled the operational JWST guide-star catalog GSC 3.2 from
MAST (`gsss.stsci.edu/webservices/vo/CatalogSearch.aspx?CATALOG=GSC32`, 559,085 sources over the
FOV, cached `refcache/gsc32.fits`). Its Gaia-backed sources are *identical* to Gaia DR3
(GSC3.2 − Gaia = 0.0, 0.0 mas, MAD 0.0). NB GSC3.2 is a hybrid — faint entries use 2MASS/VISTA/PS1
astrometry, so only the Gaia-backed (gaiaGmag-finite) subset is the clean absolute tie.

**Reference catalog built:** `catalogs/gaia_virac2_refcat_epoch2022.70.fits` (+ .ecsv) — 115,024
sources on the Gaia DR3 frame at epoch 2022.70 (1766 Gaia DR3 + 113,258 VIRAC2 fill where Gaia is
incomplete), pipeline-compatible (RA/DEC/skycoord + VERSION/EPOCH meta). Use this as the tweakreg
abs_refcat instead of the F182M→VVV-DR4 product.

**Re-anchored F115W catalog:** `catalogs/f115w_merged_indivexp_merged_dao_basic_GAIAFRAME.fits`
(598,647 sources). Built by applying the per-frame VIRAC2 corrections
(`f115w_virac2_perframe_shifts.ecsv`) to each exposure's `skycoord_centroid` and re-merging onto
the existing source grid. *(Done standalone, NOT via the production merge offsets-table: that
table's `dra`/`RAOFFSET` are in arcsec with values to −17.5″ and the per-frame meta RAOFFSET is not
the shift actually applied to skycoord_centroid — the convention is ambiguous and unsafe to feed
blindly. Confirm it before a production re-merge.)*

Verification (median offset / per-source MAD, mas):

| reference | original dao_basic | **Gaia-frame re-merge** |
|---|--:|--:|
| Gaia DR3 | 34 / MAD 48 | **18 / MAD 10** (sem 0.5) |
| GSC 3.2 (Gaia subset) | 25 / MAD 55 | **18 / MAD 10** |
| VIRAC2 | 64 / MAD 50 | **10 / MAD 37** |

Internal frame-to-frame repeatability preserved (~3 mas median, ≥3 exposures).

**Final zero-point (anchored to Gaia DR3, not VIRAC2).** The per-frame VIRAC2 step removes
frame-to-frame *structure* but its bulk inherits VIRAC2's ~5 mas tie to Gaia + estimator/sampling
differences (median-of-per-frame-medians ≠ median-of-pooled-merged; different match radius/sample).
A single final rigid shift to Gaia DR3 (`final_zeropoint.py`, shift −9.98, −15.80 mas) zeroes it:

| vs | bulk offset (mas) | per-source MAD (mas) |
|---|--:|--:|
| **Gaia DR3** | **(0.06, 0.13)** | 7.5, 6.5 |
| **GSC 3.2, posSource=15 (= Gaia DR3)** | **(0.06, 0.87)** | 7.2, 5.6 |
| VIRAC2 | (−3.9, −5.5) | 17.8, 17.7 |

The residual −5 mas vs VIRAC2 IS VIRAC2's own offset from Gaia — confirming the catalog is on the
Gaia frame. Final F115W budget: **bulk ~0.1 mas absolute (Gaia frame), ~3 mas internal
frame-to-frame, ~7 mas per-source** (Gaia-counterpart + JWST floor). Anchor the zero-point to
Gaia, never to VIRAC2.

### 8a. GSC 3.2 is a federation (per-source `posSource`) — reconciled
GSC 3.2 is NOT single-frame: each source's position comes from the best available survey, flagged
by `posSource` (authoritative table: STScI "Source Codes", MAST Data Holdings). Relevant codes:
9=2MASS, 15=GAIA, 16=PS1, 25=VISTA-VVV, 26=VISTA-VIRAC, 27=VLT-GNS (GALACTICNUCLEUS). Breakdown over
the brick FOV (559,085 sources):

| posSource | N | fraction | offset vs Gaia DR3 | identity |
|--:|--:|--:|--|--|
| 15 | 4,485 | 0.8% | (0.0, 0.0) mas | **GAIA DR3 (exact)** — the bright/guide-star tier |
| 27 | 522,916 | **93.5%** | (−37, +53) mas | **VLT-GNS** (GALACTICNUCLEUS HAWK-I); VVV/2MASS-tied frame, epoch ~2016, no PM |
| 25 | 27,914 | 5.0% | (21, +58) mas | VISTA-VVV (2MASS-tied) |

`pmSource=26` (VIRAC) supplies proper motions for the ~7% of GNS entries that have them.

**CRITICAL: the JWST guide-star frame in the GC is VVV/GNS, NOT Gaia.** FGS guide stars are
12.5 ≤ J ≤ 18 (2MASS J). In the brick, GSC 3.2 positions in that range come from:
J 12.5–15 → VVV 74% / GNS 21% / Gaia 2%; J 15–16.5 → GNS 58% / VVV 39%; J 16.5–18 → GNS 90% / VVV 8%.
So Gaia supplies ≤2% of guide-star-magnitude positions — JWST guides (and therefore sets the
operational pointing/aperture frame) on **VVV/GNS positions, which are offset from Gaia** in the GC.

**Offset magnitudes (common epoch 2016, clean isolated bright matches — earlier propagated numbers
were inflated by an epoch mismatch):** VVV (posSource=25) **~29 mas** off Gaia (consistent with
independent VVV-DR4 vs Gaia = 24 mas); GNS (posSource=27) **~50–63 mas** (small clean sample, N~25).
The bulk apparent motion that inflated the first estimate is the solar reflex (~−6 mas/yr in l),
NOT intrinsic streaming (random bulge orbits cancel in the median).

**This is real and literature-documented, NOT reconciled in GSC 3.2:**
- GNS DR1 (Nogueras-Lara et al. 2019, A&A 631 A20): astrometry calibrated to **VVV** (not Gaia),
  stated absolute accuracy **≲0.05″ (≲50 mas)** — exactly the offset we measure.
- GNS DR2 (Nogueras-Lara et al. 2025, arXiv:2510.11966): improved 5× to **~5 mas, now Gaia-DR3
  tied** — but this postdates GSC 3.2.
- GSC 3.x is a **labeled federation**: each source keeps its origin survey's astrometry (flagged by
  `posSource`); only the bright tier (posSource=15) is Gaia. The faint VVV/GNS tiers retain their
  native ~29/~50 mas frame offsets — no re-registration to Gaia, hence no reconciliation note.
- GNS (27) ≠ VIRAC (26): different surveys/frames. VIRAC (26) is used by GSC only for proper motions
  here; VIRAC2 (II/387) itself is Gaia-tied to ~5–7 mas.

**Consequence for NIRSpec — choose ONE frame, deliberately:**
- The telescope blind-points in the GSC (VVV/GNS) frame, ~30–50 mas off Gaia. MSATA/WATA then
  centroids observer-supplied reference stars and corrects, so after TA the targets land per the
  *internal* consistency of the observer's catalog — but the absolute frame and the guide↔target
  frame offset matter for blind pointing, TA capture, and any non-TA observation.
- **Do not mix frames across filters** (verified: F115W on Gaia vs F200W on VIRAC2 → 24 mas Dec
  shift between them). Put ALL filters/targets on a single frame.

## 9. ADOPTED FRAME (decision): VIRAC2 (DR2, II/387) for ALL filters
GNS dominates GSC 3.2 by *number* (deep HAWK-I) but is a bounded mosaic; **VVV/VIRAC2 covers the full
pointing footprint contiguously**, so VIRAC2 is the single uniform NIR frame across all pointings.
VIRAC2 is tied to Gaia DR3 (~5–7 mas) — the Gaia frame realized densely; NOT the 2MASS-tied frame of
the GSC guide-star positions.

Final F115W product: `catalogs/f115w_merged_indivexp_merged_dao_basic_VIRAC2FRAME.fits` (598,647 src).
Verification: vs VIRAC2 (−0.5,−0.5) mas (zeroed); vs Gaia (3.8,5.6) mas (= VIRAC2's own Gaia offset);
internal ~3 mas. `_GAIAFRAME.fits` retained as the pure-Gaia alternative (one −3.8,−5.6 mas shift
apart). Apply the SAME VIRAC2 frame to every filter (F182M, F200W to follow).


- `absolute_frame.py` — per-filter absolute offsets vs all references
- `residual_maps_vs_virac2.png` — spatial residual maps (rigid vs distortion)
- `f115w_perframe.py`, `per_frame_offsets.ecsv`, `per_frame_offset_by_detector.png` — per-frame
- `f115w_gaia_tie.py` — per-frame VIRAC2 alignment + saturated-star Gaia tie
- `f115w_virac2_perframe_shifts.ecsv` — the per-frame alignment solution
- `fetch_refs.py`, `refcache/{virac2,gsc242}.fits` — cached Gaia-frame references
