"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.run_notebook``."""
import sys as _sys
import jwst_gc_pipeline.reduction.run_notebook as _impl
_sys.modules[__name__] = _impl
