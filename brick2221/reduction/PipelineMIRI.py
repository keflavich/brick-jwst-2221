#!/usr/bin/env python
"""Backward-compat stub. Implementation moved to ``jwst_gc_pipeline.reduction.PipelineMIRI``.

When invoked as a script, this re-runs the moved file via ``runpy.run_path`` so
existing sbatch / cron commands that point at this path keep working.
When imported as a module, attribute access falls through to
``jwst_gc_pipeline.reduction.PipelineMIRI``.
"""
import os as _os
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    import jwst_gc_pipeline.reduction as _r
    _target = _os.path.join(_os.path.dirname(_r.__file__), "PipelineMIRI.py")
    _runpy.run_path(_target, run_name="__main__")
else:
    import jwst_gc_pipeline.reduction.PipelineMIRI as _impl
    _sys.modules[__name__] = _impl
