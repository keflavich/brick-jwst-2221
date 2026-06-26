# MIRI cataloging TODO (back-burner until the imaging NaN corruption is fixed)

Created 2026-06-25.  BLOCKED on the MIRI imaging re-reduction (brick F2550W +
w51 F2100W have 32-38% interior NaN; all fields except sickle are 16-21% vs
sickle's clean 5.8%).  Do NOT re-tune cataloging on corrupted images -- re-image
first, then revisit this list.

User evaluation of the 2026-06-24 re-cataloged products (post fakes/satstar/oversub fixes):

## Per-field cataloging issues to fix (after re-imaging)

1. **brick F2550W** -- bad oversubtraction at 17:46:06.55 -28:43:23.0.
   (BLOCKED: image quality ~zero, most of image NaN -> re-image first.)

2. **cloudc 2526 F770W** -- UNIVERSAL under-subtraction.
   Examples: `f770w_undersubtracted_20260626.reg` (in /orange/.../sickle/regions_/
   or /orange/.../cloudc/regions_/ -- confirm path).
   Likely amplitude/PSF too small across the board for this field.

3. **cloudc F2550W** -- oversubtraction, NOT universal (some stars OK).
   Examples: `f2550w_oversubtracted_20260626.reg`.
   ALSO: the brightest MIRI stars need ~2x BIGGER PSFs (fov_pixels) -- the current
   fovp1024 large PSF is too small to capture their wings at 25um.

4. **sgrb2 F2550W** -- MILD oversubtraction, mostly good.
   Examples: `f2550w_oversub_20260626.reg`.  Most oversubs are in LOW-BACKGROUND
   regions.
   - Central saturated region is NOT a point source -> EXTENDED saturated regions
     in MIRI should be IGNORED (do not fit as satstars).  [general rule]
   - Most/all bright stars have SQUARE ARTIFACTS around them -- likely a
     core-saturation artifact, probably not easily solvable.  NOTE FOR LATER.

5. **W51 (all MIRI)** -- so few sources that we will NOT debug cataloging now.

6. **#4/#5 from the 2526 diagnosis (lower harm):**
   - SAT-faint (17:46:18.25 -28:37:20.5): genuinely faint saturated star, uses
     large PSF, mild WING under-subtraction.
   - DQ-completeness edge case: a genuine NaN-variance core with NO DQ-SATURATED
     flag in some frames (e.g. SAT-faint frame 3: sat_area=0 but 121-px NaN core)
     -> seed saturated stars from genuine NaN cores even when DQ-SAT is absent.

## Cross-cutting cataloging rules to add (general, not field-specific)
- EXTENDED saturated regions (large, non-point-source) must be ignored as satstars.
- Per-field PSF size: brightest stars at long MIRI wavelengths (F2550W) need a
  ~2x larger fov_pixels grid than the current fovp1024.
- Universal under/over-subtraction per field suggests a per-field amplitude
  calibration (zeropoint / PSF-peak normalization) is needed.

## Status of the 3 fixes already landed (PR #4, branch miri-joint-satstar)
- #1 fake stars: DONE (554ba98) -- coadd-first + small-radius prominence + NaN-core predicate
- #2 unidentified saturated: DONE (dcc4c96) -- preserve genuine NaN cores in large components
- #3 oversub pockmarks: DONE (f175430) -- hard-cap model peak to data peak
These are validated + regression-tested; the NEW issues above are on top of them.

## Added 2026-06-25 (after imaging fix)
7. **Cataloging data_i2d resample worse than the science reduction.**
   The cataloging-built `_data_i2d` (mosaic_cutout_input_data -> _resample_to_i2d,
   ResampleStep defaults) has MORE NaN than the image3 reduction i2d on the same
   crf (brick F2550W: data_i2d 23% vs reduction 11%; w51 F2100W: 35% vs 26%).
   The SCIENCE reduction is fine; this is a visualization-product discrepancy.
   FIX OPTIONS: (a) match _resample_to_i2d settings (pixfrac/weight/good_bits) to
   image3, or (b) point the CARTA `*_m2_diag` snippets' "data" layer at the
   reduction i2d instead of the cataloging data_i2d.
8. **All w51 MIRI re-reduced 2026-06-25** with current outlier params (NaN now
   F770W 17%, F1280W 19%, F2100W 26%, full coverage).  crf renamed old
   `_align_o002_crf` -> new `_o002_crf`; stale May-2025 crf archived to
   `_stale_may2025_crf/`.  Cataloging launchers updated to `--each-suffix=o002_crf`.
