This repository contains notes & scripts for reduction and analysis of the JWST
Cycle 1 project 2221 GO that observed the Brick and Cloud C in NIRCAM
narrow-band filters and the MIRI F2550W filter.

While some aspects of this repository are well-organized, the notebooks are
not: they often include asides and experiments that lead to dead ends or just
got dropped.  There are a lot of 'notes in the margins'.  Most of them should
work if you have the directory structure set up right and the right files in
place, but I haven't put in the time needed to make clear what that structure
should be.

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

 * cloudef: proposal `2092`, field `005`, filters `F162M,F210M,F360M,F480M`, reference filter `F210M`
 * sgrc: proposal `4147`, field `012`, filters `F115W,F162M,F182M,F212N,F360M,F405N,F470N,F480M`, reference filter `F212N`
 * sgrb2: proposal `5365`, field `001`, filters `F150W,F182M,F187N,F210M,F212N,F300M,F360M,F405N,F410M,F466N,F480M`, reference filter `F210M`
 * arches: proposal `2045`, field `001`, filters `F212N,F323N`, reference filter `F212N`
 * quintuplet: proposal `2045`, field `003`, filters `F212N,F323N`, reference filter `F212N`
 * sgra: proposal `1939`, field `001`, filters `F115W,F212N,F405N`, reference filter `F212N`

You can override these with environment variables before launching, for example:

 * `CLOUDEF_FIELD`, `CLOUDEF_FILTERS`, `CLOUDEF_REF_FILTER`
 * `SGRC_FIELD`, `SGRC_FILTERS`, `SGRC_REF_FILTER`
 * `SGRB2_FIELD`, `SGRB2_FILTERS`, `SGRB2_REF_FILTER`
 * `ARCHES_FIELD`, `ARCHES_FILTERS`, `ARCHES_REF_FILTER`
 * `QUINTUPLET_FIELD`, `QUINTUPLET_FILTERS`, `QUINTUPLET_REF_FILTER`
 * `SGRA_FIELD`, `SGRA_FILTERS`, `SGRA_REF_FILTER`
 * `MODULES` (default `merged`)
 * `ARRAY_RANGE` (default `0-23`)