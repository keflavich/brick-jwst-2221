#!/usr/bin/env python
"""Backward-compat stub. Implementation moved to ``jwst-gc-pipeline``.

Filename has a hyphen, so this can only be invoked as a script.
"""
import os
import runpy
import jwst_gc_pipeline.reduction as _r

_target = os.path.join(os.path.dirname(_r.__file__), "PipelineRerunNIRCAM-SHORT.py")

if __name__ == "__main__":
    runpy.run_path(_target, run_name="__main__")
