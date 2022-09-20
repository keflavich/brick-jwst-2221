Reduction process:

 1. Pipeline files are in `reduction/`.
  * `PipelineRerunNIRCAM-{LONG,SHORT}.py` run the JWST pipeline with modifications to the tweakwcs stage
  * `destreak.py` runs "Massimo's Destriper", which is a simple percentile-subtraction across the X-axis of each horizontal quadrant of each detector
  * `align_to_catalogs.py` includes some post-facto re-alignment tools.  One function runs only on the final processed data & matches it to VVV.
 2. Notebooks.  There are a lot of these.
  * `BrA_separation_nrc{a,b}.ipynb` should be run after the pipeline.  They subtract BrA from F410M, then subtract the rescaled continuum from the BrA image.
  * `F466_separation_nrc{a,b}.ipynb` does the same thing, but for the F466 filter.  It doesn't work as well because of the wavelength difference.
  * `StarKILLER_nrc{a,b}.ipynb` manually remove stars from the images, only for display purposes!
  * `Stitch_A_to_B.ipynb` pulls the two modules together.
  * `nrca_contsub_colorimages.ipynb` and `BrickImages.ipynb` make 3-color images.

  Most of the rest are test / WIP things.
