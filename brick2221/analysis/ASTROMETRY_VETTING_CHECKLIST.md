# Astrometry vetting checklist for JWST GC fields (for other agents)

Goal: for every target, confirm internal precision (frame-to-frame, edge, seam) and a correct
absolute tie, and cross-check overlapping programs. Reference implementations live in
`brick2221/analysis/`: `seam_edge_analysis.py`, `filter_vs_virac2_fluxmatched.py`,
`build_f200w_f182m_reference.py`. Read `BRICK_ASTROMETRY_REPORT.md` for a worked example.

## Ground rules (DON'T skip)
- [ ] **daophot BASIC only.** Crowdsource catalogs are deprecated (`*crowdsource*`, `*qualcuts*`).
- [ ] **Use the CURRENT COMPLETE per-filter catalog, not the legacy base.** The no-suffix
      `<filt>_merged_indivexp_merged_dao_basic.fits` is the STALE legacy build and is INCOMPLETE
      (missing frames/detectors -> coverage holes; it produced false F200W-nrca4 blanks and a false
      "F115W 75 mas off" result). The current complete products are the manual passes
      `<filt>_merged_indivexp_merged_<m>_dao_basic.fits` (192 frames SW / 48 LW). Resolve via
      `catalog_paths.best_dao_basic(filt)`. **Always check frame completeness** (FN meta count vs
      expected 192/48; per-detector coverage) before trusting a catalog. Brick's stale base merges
      were moved to `catalogs/stale_20260618/`.
- [ ] **Reference-frame policy** (see `feedback_reference_frame_policy`):
      GC fields → VIRAC2 (II/387) positions PM-propagated **per-star** from VIRAC2 ref epoch **2014.0**
      to the obs epoch; non-GC → Gaia DR3 per-star from **2016.0**. VIRAC2 epoch is **2014.0** (NOT 2016.0).
      Inner/dense GC (Sgra/Arches/Quintuplet/Gc2211/Sickle) currently stay on **GNS** by decision.
- [ ] **Dense-refcat trap:** never measure a bulk offset by nearest-neighbour median against a dense
      catalog (it is biased to ~0). Use **offset-histogram** stacking (peak of the 2D pair-offset
      histogram), refined with iterative median in shrinking windows. (`feedback_dense_refcat_astrometry`)
- [ ] **Gaia in the GC is foreground** (large individual PMs). Only use Gaia as a check if you
      propagate the Gaia stars **per-star to the obs epoch**; an unpropagated Gaia comparison to a
      2014 VIRAC2 frame is dominated by foreground motion and is NOT a frame error.

## Per-target checks
- [ ] **Obs epoch:** read `DATE-OBS` from a cal header → jyear. Record it (per program/obs).
- [ ] **Field center + radius:** from `TARG_RA/TARG_DEC` (and mosaic extent); use radius that covers
      the full mosaic (≥0.1°) when building the refcat.
- [ ] **Frame-to-frame repeatability:** in each per-filter combined catalog, check `std_ra`/`std_dec`
      (cross-frame scatter). Expect ~0.5–1 mas for well-measured bright stars. Elevated std in a
      region flags a misaligned exposure/detector.
- [ ] **Edge effects:** an edge ramp would inflate same-detector cross-filter MAD. If a same-program
      SW–SW pair (e.g. F182M–F212N) has MAD ≲1 mas across the field, edges are fine. If suspect,
      drop to per-exposure catalogs and plot residual vs detector (x,y).
- [ ] **Internal seam (per program):** run `seam_edge_analysis.py` for **same-program** pairs:
      one SW–SW and one SW–LW (e.g. 2221: F182M–F212N, F182M–F410M; 1182: F200W–F356W, F200W–F444W).
      Map dRA/dDec + the vs-Dec profile. A real internal seam shows a bulk jump at a sky boundary;
      both Brick programs were internally flat (<1 mas jump) — the old "25 mas seam" was a crowdsource
      artifact. Flag any band that disagrees with its same-program siblings (e.g. Brick F115W sat on a
      different raw zero-point).
- [ ] **Absolute tie:** build the policy refcat (`build_gaia_virac2_refcat_byquery.py` for GC),
      offset-histogram tie each filter, confirm residual vs the refcat ≈0 (≲1–2 mas bulk).
- [ ] **Flux-matched VIRAC2 check (the clean per-band test):** run `filter_vs_virac2_fluxmatched.py`
      for the JHK-ish bands (F115W~J, F182M~H, F200W~Ks, F212N~Ks). Down-select matches by the
      flux/mag relation (3-sigma core) to kill crowded false matches; inspect the offset cloud,
      flux-match relation, and dRA/dDec sky map. Bright cores should land within a few mas of VIRAC2.
- [ ] **Gaia cross-check:** Gaia DR3 PM-propagated per-star to the obs epoch; expect ~1–8 mas
      (residual = VIRAC2–Gaia tie ~5 mas + epoch). A uniform per-band offset is the frame tie, not an
      error; a band that deviates from the others is the problem.

## Cross-checks between overlapping frames (e.g. 1182 ⨯ 2221)
When two programs (or two epochs/visits) cover the same sky:
- [ ] **Direct cross-program match** (refcat-independent): match a band from program A to a band from
      program B (e.g. F200W↔F182M), map dRA/dDec vs sky position. This is the most sensitive probe —
      a bulk offset = relative frame difference; a localized jump = a boundary/seam.
- [ ] **Localize any seam:** confirm with the vs-Dec (or vs-RA) profile and a 2D map. Check whether it
      is present in BOTH programs internally (then it's a shared distortion) or only cross-program
      (then it's a relative tweakreg/WCS divergence — Brick's thin Dec −28.709 band is this case).
- [ ] **Decide which program to trust per region:** if one program is internally clean across the
      boundary and the other isn't, prefer the clean one there when building a joint reference.
- [ ] **After both are tied to the same absolute frame, re-check** the cross-program residual. It
      should collapse to the tie noise (~few mas) except at known seams.
- [ ] **Saturation/coverage caveats:** verify the cross-match isn't dominated by saturated stars or a
      thin-coverage detector edge (Brick's F200W "L-hole" was a crowdsource *merge* artifact at the
      nrca4 visit-002 edge — image + per-exposure + per-filter combined were all fine).

## Record / deliver
- [ ] Write per-target shifts, MADs, seam amplitudes, and the Gaia cross-check to a short report.
- [ ] Save diagnostic PNGs to `astrometry_seam_dao_<date>/` (filter–filter `seam_*.png`,
      flux-matched `viracmatch_*.png`, `reference_build_diagnostics.png`).
- [ ] Note any band on a discrepant raw zero-point and any archive/cap caveats (e.g. Gaia 2000-row
      sync cap → use async/uncapped).
