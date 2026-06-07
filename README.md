This repository contains notes & scripts for reduction and analysis of the JWST
Cycle 1 project 2221 GO that observed the Brick and Cloud C in NIRCAM
narrow-band filters and the MIRI F2550W filter.

While some aspects of this repository are well-organized, the notebooks are
not: they often include asides and experiments that lead to dead ends or just
got dropped.  There are a lot of 'notes in the margins'.  Most of them should
work if you have the directory structure set up right and the right files in
place, but I haven't put in the time needed to make clear what that structure
should be.

## Package split

As of 2026, the generic JWST photometry pipeline that was originally developed
in this repository has been extracted into a separate package,
[`jwst-gc-pipeline`](https://github.com/keflavich/jwst-gc-pipeline),
so that other Galactic Center JWST programs can share the same reduction and
crowded-field photometry code.

- **`brick2221`** (this repository) keeps the Brick / Cloud C science:
  ice analyses (`make_icecolumn_fig9*`, `icecolumn_*`, `make_ccd_with_icemodels`,
  `co_*`, etc.), foreground extinction modeling, paper-figure generators,
  selections, distance analysis, and `analysis_setup` / `paths` configuration.
- **`jwst-gc-pipeline`** holds the field-agnostic pipeline:
  `PipelineRerun*`, `destreak`, `align_to_catalogs`, `saturated_star_finding`,
  `filtering`, `make_merged_psf`, `merge_a_plus_b`, `crowdsource_catalogs_*`,
  `merge_catalogs`, `make_reftable`, generic `plot_tools`, and `isochrones`.

The old import paths inside `brick2221.reduction.*` and
`brick2221.analysis.*` continue to work as backward-compatibility shims —
they re-export the implementation from `jwst_gc_pipeline`. New code should
import directly from `jwst_gc_pipeline`.

## Iter1 / Iter2 / Iter3 / Iter4 cataloging cycle

After the JWST pipeline (`PipelineRerunNIRCAM-LONG.py` etc.) produces
per-frame `*_destreak_o<NNN>_crf.fits` images, photometry is built up
across four iterative passes.  Every pass runs the same script
(`jwst_gc_pipeline.photometry.crowdsource_catalogs_long` /
`brick2221.analysis.crowdsource_catalogs_long`) on a per-frame basis,
but with different inputs and goals.

 1. **iter1 — basic per-frame DAO seed.** `crowdsource_catalogs_long.py
    --each-exposure --daophot --skip-crowdsource` (no `--iteration-label`).
    Finds peaks above noise on each per-frame destreak/crf image and runs
    one DAOPHOT PSF-fit pass.  Output is the seed catalog used downstream.
 2. **iter2 — residual-aware refit, per filter.** Same script with
    `--iteration-label=iter2 --postprocess-residuals`.  Uses the iter1
    catalog as a seed, fits the residual for missed sources, and writes a
    cleaned per-frame catalog + residual.  `merge_catalogs.py
    --merge-singlefields --indiv-merge-methods=daoiterative
    --iteration-label=iter2` then merges per-frame catalogs within each
    filter (cross-exposure merge) — the per-filter daoiterative output is
    what feeds the iter3 cross-band union seed.
 3. **iter3 — cross-band union-seeded fit.**
    `build_union_seed_catalog.py` unions the iter2 per-filter merged
    catalogs into a single `seed_union_iter3_<target>.fits`, then
    `crowdsource_catalogs_long.py --iteration-label=iter3
    --seed-catalog=<union> --postprocess-residuals` is run per frame.
    This is the canonical "all filters, all frames" simultaneous fit and
    produces the final per-frame iter3 photometry + residual mosaics.
    `merge_catalogs.py --iteration-label=iter3` produces the merged
    iter3 multiwavelength catalog.  Per-frame "forced photometry on
    satstar residuals" (`forced_photometry_residuals.py`) is gated on
    iter3 to recover sources undetected in individual filters.
 4. **iter4 — residual-background refit (optional refinement).**
    `run_iter4resbgrefit.sh` builds median-smoothed background images
    from the iter3 residuals, then refits the iter3 catalog as seeds
    with tight `xy_bounds` on the residual-bg-subtracted data.  Output
    `--iteration-label=iter4resbgrefit` is purely additive — iter1/2/3
    are not overwritten.  A parallel `iter2residbg` / `iter3residbg`
    cascade exists in `run_residbg_cataloging.sh` for early-stage
    residbg-driven refitting.

The cross-iteration dependency tree is

```
iter1 (per-filter, per-frame DAO)
  ↓ (per-filter merge)
iter2 (per-filter, per-frame, with residual postprocess)
  ↓ (per-filter merge → daoiterative)
seed_union_iter3 (cross-band union)
  ↓
iter3 (per-frame, all sources fit jointly, with residuals)
  ↓ (per-target merge)
iter4resbgrefit (residual-bg refinement, optional)
```

### Job runner coverage matrix

Each shell script in `brick2221/shellscripts/` implements one or more
iterations.  These are the supported entry points:

| Script                          | iter1 | iter2 | iter3 | iter4 | Pipeline | Refcat | Notes |
|---------------------------------|:-----:|:-----:|:-----:|:-----:|:--------:|:------:|-------|
| `submit_full_chain.sh`          |  ✓   |  ✓   |  ✓   |       |          |        | iter1 → iter2 → merge → iter3 launch |
| `run_full_pipeline_common.sh`   |  ✓   |       |       |       |    ✓    |   ✓   | First-pass + refcat + second-pass + iter1 + merge |
| `run_full_pipeline_<target>.sh` |  ✓   |       |       |       |    ✓    |   ✓   | Thin wrappers over `_common.sh` |
| `run_iter3_cataloging.sh`       |       |       |  ✓   |       |          |        | Builds union seed + per-frame iter3 + merge |
| `run_iter4resbgrefit.sh`        |       |       |       |  ✓   |          |        | iter3 residual bg refit (purely additive) |
| `run_residbg_cataloging.sh`     |       | iter2residbg | iter3residbg |  |  |  | residual-bg cascade |

### Target / observation coverage

Each target's full per-frame set must be reduced *across all observations
that belong to it*.  Missing an observation here means a permanent hole
in the merged catalog.  Coverage as of 2026-06-07:

| Target / "name"   | Proposal | Obs (fields)         | Filters                                                            |
|-------------------|---------:|----------------------|--------------------------------------------------------------------|
| brick (2221 narrowband) | 2221 | 001                  | F182M F187N F212N F405N F410M F466N                                |
| brick-1182 (broadband)  | 1182 | 004                  | F115W F200W F356W F444W                                            |
| cloudc            |     2221 | 002                  | F182M F187N F212N F405N F410M F466N                                |
| sickle            |     3958 | 007                  | F187N F210M F335M F470N F480M                                      |
| sgrb2             |     5365 | 001                  | F150W F182M F187N F210M F212N F300M F360M F405N F410M F466N F480M  |
| sgra              |     1939 | 001                  | F115W F212N F405N                                                  |
| cloudef           |     2092 | **002 + 005**        | F162M F210M F360M F480M (Cloud E = 002, Cloud F = 005, merged)     |
| arches            |     2045 | 001                  | F212N F323N                                                        |
| quintuplet        |     2045 | 003                  | F212N F323N                                                        |
| sgrc              |     4147 | 012                  | F115W F162M F182M F212N F360M F405N F470N F480M                    |
| gc2211            |     2211 | **023, 028, 046, 049, 050** | obs 028: F150W,F277W; others: F200W,F277W                  |

Multi-obs targets (cloudef, gc2211) require the runner to loop over
every obs id; missing one silently drops part of the field.  Brick is
covered as two separate launcher targets (`brick` for proposal 2221 +
`brick-1182` for the broadband proposal 1182) because the per-frame
`destreak_o<NNN>_crf` suffix is obs-specific.

## Reduction process

 1. Pipeline files are in `reduction/`.
  * `PipelineRerunNIRCAM-LONG.py` run the JWST pipeline with modifications to the tweakwcs stage
  * `destreak.py` runs "Massimo's Destriper", which is a simple percentile-subtraction across the X-axis of each horizontal quadrant of each detector (this is run by the pipeline)
  * `align_to_catalogs.py` includes some post-facto re-alignment tools.  One function runs only on the final processed data & matches it to VVV. (this is run by the pipeline)
  * `saturated_star_finding.py` performs PSF fitting on saturated stars and removes them.  (this is run by the pipeline)
  * `crowdsource_catalogs_long.py` runs the crowsource extraction algorithm.  This must be run on the long-wavelength channels before running the short-wavelength pipeline to provide the reference catalog we use for the shortwave data.
  * `merge_catalogs.py` merges the multiwavelength catalogs.
  * `make_reftable.py` makes the reference table from the long-wavelength (F410M) data to be used on the short-wavelength data (this is run by merge_catalogs) - note that this is in analysis/
  * `PipelineRerunNIRCAM-SHORT.py` run the JWST pipeline with modifications to the tweakwcs stage including a reference catalog generated from F410M
  * `crowdsource_catalogs_long.py` on the short data (it has _long in the name, but I merged both into this one; the _short version is deprecated)  - note that this is in analysis/
  * `merge_catalogs.py` again to finally merge wavelengths - note that this is in analysis/

 2. Notebooks.  There are a lot of these.
  * `BrA_separation_nrc{a,b}.ipynb` should be run after the pipeline.  They subtract BrA from F410M, then subtract the rescaled continuum from the BrA image.
  * `F466_separation_nrc{a,b}.ipynb` does the same thing, but for the F466 filter.  It doesn't work as well because of the wavelength difference.
  * `StarKILLER_nrc{a,b}.ipynb` manually remove stars from the images, only for display purposes!
  * `Stitch_A_to_B.ipynb` pulls the two modules together.  It makes the "star-free" three-color merged image
  * `nrca_contsub_colorimages.ipynb` and `BrickImages.ipynb` make 3-color images.

  Most of the rest are test / WIP things.


## Analysis

The analysis has been done haphazardly in notebooks, but then I tried to reconcile different notebooks and merge them.

 * `analysis_setup.py` - loads the latest versions of 3-color images and catalogs.  Meant to be called before running plotting things.
 * `co_fundamental_modeling.py` - extracts from the CO2_PhoenixModel.ipynb, CO2_Phoenix_Example.ipynb, and COFundamentalModeling_and_IceModeling.ipynb notebooks.  This script produces a wider range of models than those notebooks
 * `plot_tools.py` - contains many plot templates for color-color, color-magnitude, etc.
 * `selections.py` - downselects from the full merged catalog of NIRCAM data of the brick, in all 6 filters, to the subset(s) that are reliable
 * `ColorVsCOIceAnalysis.ipynb` - makes several color-color plots



### Setup for Reduction

There are a few basic setup requirements before reduction of a new target:

 * The target data root must exist at `/orange/adamginsburg/jwst/{target}/`.
 * A `crds/` directory must exist under that target root.
 * A `reduction/fwhm_table.ecsv` file must exist under that target root.
 * A `catalogs/` directory should exist under that target root (it will be created automatically by helper scripts if missing).
 * If you want VVV-based alignment for that target, provide a field-of-view region file under `regions_/` and wire it into `fov_regname` in `brick2221/reduction/PipelineRerunNIRCAM-LONG.py`.

Validated mapping locations in the codebase (these must be coherent for any new target):

 * `brick2221/reduction/PipelineRerunNIRCAM-LONG.py`
   * proposal/field -> target mapping in `field_to_reg_mapping`
   * proposal/field -> reference catalog mapping in `REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD`
   * optional target -> fov region mapping in `fov_regname`
 * `brick2221/analysis/crowdsource_catalogs_long.py`
   * proposal/field/target mapping in `field_to_reg_mapping`
   * visit-count policy in `nvisits`
 * `brick2221/analysis/merge_catalogs.py`
   * filter inventory per target/proposal in `obs_filters`
   * target/proposal -> field mapping in `project_obsnum`

Current first-pass/second-pass behavior:

 * If the configured reference catalog does not exist yet, `PipelineRerunNIRCAM-LONG.py` now runs a first pass and skips reference-catalog realignment steps.
 * After building a reference catalog, rerun with `--skip_step1and2` to reuse existing `_cal` products and perform the aligned second pass.

### Full Pipeline For cloudef/sgrc/sgrb2/arches/quintuplet/sgra

Use:

 * `brick2221/shellscripts/run_full_pipeline_cloudef_sgrc.sh`

This script submits an end-to-end dependency tree for each target:

 1. First-pass pipeline jobs for all configured filters.
 2. Reference-catalog build via `make_reference_from_pipeline_catalogs.py`.
 3. Second-pass pipeline jobs with `--skip_step1and2`.
 4. Per-exposure DAO cataloging arrays via `crowdsource_catalogs_long.py`.
 5. Cross-filter merge via `merge_catalogs.py --merge-singlefields`.

Defaults:

 * cloudef: proposal `2092`, fields `002,005` (Cloud E + Cloud F), filters `F162M,F210M,F360M,F480M`, reference filter `F210M`
 * sgrc: proposal `4147`, field `012`, filters `F115W,F162M,F182M,F212N,F360M,F405N,F470N,F480M`, reference filter `F212N`
 * sgrb2: proposal `5365`, field `001`, filters `F150W,F182M,F187N,F210M,F212N,F300M,F360M,F405N,F410M,F466N,F480M`, reference filter `F210M`
 * arches: proposal `2045`, field `001`, filters `F212N,F323N`, reference filter `F212N`
 * quintuplet: proposal `2045`, field `003`, filters `F212N,F323N`, reference filter `F212N`
 * sgra: proposal `1939`, field `001`, filters `F115W,F212N,F405N`, reference filter `F212N`

You can override these with environment variables before launching, for example:

 * `CLOUDEF_FIELD`, `CLOUDEF_FIELDS` (CSV for multi-obs), `CLOUDEF_FILTERS`, `CLOUDEF_REF_FILTER`
 * `SGRC_FIELD`, `SGRC_FILTERS`, `SGRC_REF_FILTER`
 * `SGRB2_FIELD`, `SGRB2_FILTERS`, `SGRB2_REF_FILTER`
 * `ARCHES_FIELD`, `ARCHES_FILTERS`, `ARCHES_REF_FILTER`
 * `QUINTUPLET_FIELD`, `QUINTUPLET_FILTERS`, `QUINTUPLET_REF_FILTER`
 * `SGRA_FIELD`, `SGRA_FILTERS`, `SGRA_REF_FILTER`
 * `MODULES` (default `merged`)
 * `ARRAY_RANGE` (default `0-23`)