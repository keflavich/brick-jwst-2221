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
