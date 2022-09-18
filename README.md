Reduction process:

1. Pipeline files are in `reduction/`.
  i. `PipelineRerunNIRCAM-{LONG,SHORT}.py` run the JWST pipeline with modifications to the tweakwcs stage
  ii. `destreak.py` runs "Massimo's Destriper", which is a simple percentile-subtraction across the X-axis of each horizontal quadrant of each detector
  iii. `align_to_catalogs.py` includes some post-facto re-alignment tools.  One function runs only on the final processed data & matches it to VVV.
2. Notebooks.  There are a lot of these.
  i. `BrA_separation_nrc{a,b}.ipynb` should be run after the pipeline.  They subtract BrA from F410M, then subtract the rescaled continuum from the BrA image.
  ii