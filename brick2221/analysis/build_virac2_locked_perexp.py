"""MOVED to jwst-gc-pipeline (2026-07-17).

The VIRAC2 per-exposure offsets-table builder is now a general, pipeline-package tool:

    jwst_gc_pipeline.reduction.build_virac2_offsets

which builds the coarse absolute tie with the sanctioned, density-immune, window-swept,
guarded ``photometry.astrometry_offsets.measure_offset`` (no bespoke coarse histogram, no
nearest-neighbour fallback -- see jwst-gc-pipeline PR #120 and brick-jwst-2221 PR #39 review).

Run:
    python -m jwst_gc_pipeline.reduction.build_virac2_offsets --region <key> [filt ...]
"""
raise SystemExit(__doc__)
