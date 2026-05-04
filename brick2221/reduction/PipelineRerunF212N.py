#!/usr/bin/env python
"""Backward-compat stub. Implementation moved to ``jwst_gc_pipeline.reduction.PipelineRerunF212N``."""
import os as _os
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    import jwst_gc_pipeline.reduction as _r
    _target = _os.path.join(_os.path.dirname(_r.__file__), "PipelineRerunF212N.py")
    _runpy.run_path(_target, run_name="__main__")
else:
    import jwst_gc_pipeline.reduction.PipelineRerunF212N as _impl
    _sys.modules[__name__] = _impl
