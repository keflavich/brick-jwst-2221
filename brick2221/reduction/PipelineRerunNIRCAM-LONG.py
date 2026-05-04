#!/usr/bin/env python
"""Backward-compat stub. Implementation moved to ``jwst-gc-pipeline``.

This file's name contains a hyphen so it can only be invoked as a script. The
module-importable form lives at ``jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py``
and is run here via ``runpy.run_path``.
"""
import os
import runpy
import jwst_gc_pipeline.reduction as _r

_target = os.path.join(os.path.dirname(_r.__file__), "PipelineRerunNIRCAM-LONG.py")

if __name__ == "__main__":
    runpy.run_path(_target, run_name="__main__")
