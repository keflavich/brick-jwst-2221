#!/usr/bin/env python
"""Backward-compat stub. Implementation moved to ``jwst_gc_pipeline.photometry.make_reference_from_pipeline_catalogs``.

Run as a script or import as a module — both modes work.
"""
import os as _os
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    import jwst_gc_pipeline.photometry as _p
    _target = _os.path.join(_os.path.dirname(_p.__file__), "make_reference_from_pipeline_catalogs.py")
    _runpy.run_path(_target, run_name="__main__")
else:
    import jwst_gc_pipeline.photometry.make_reference_from_pipeline_catalogs as _impl
    _sys.modules[__name__] = _impl
