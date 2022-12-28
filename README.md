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
  * `PipelineRerunNIRCAM-{LONG,SHORT}.py` run the JWST pipeline with modifications to the tweakwcs stage
  * `destreak.py` runs "Massimo's Destriper", which is a simple percentile-subtraction across the X-axis of each horizontal quadrant of each detector (this is run by the pipeline)
  * `align_to_catalogs.py` includes some post-facto re-alignment tools.  One function runs only on the final processed data & matches it to VVV. (this is run by the pipeline)
  * `saturated_star_finding.py` performs PSF fitting on saturated stars and removes them.  (this is run by the pipeline)
 2. Notebooks.  There are a lot of these.
  * `BrA_separation_nrc{a,b}.ipynb` should be run after the pipeline.  They subtract BrA from F410M, then subtract the rescaled continuum from the BrA image.
  * `F466_separation_nrc{a,b}.ipynb` does the same thing, but for the F466 filter.  It doesn't work as well because of the wavelength difference.
  * `StarKILLER_nrc{a,b}.ipynb` manually remove stars from the images, only for display purposes!
  * `Stitch_A_to_B.ipynb` pulls the two modules together.
  * `nrca_contsub_colorimages.ipynb` and `BrickImages.ipynb` make 3-color images.

  Most of the rest are test / WIP things.


## Analysis


Photometry:
  * `crowdsource_catalogs_long.py` and `crowdsource_catalogs_short.py` run the crowsource extraction algorithm
