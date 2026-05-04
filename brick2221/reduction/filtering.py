"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.filtering``."""
import sys as _sys
import jwst_gc_pipeline.reduction.filtering as _impl
_sys.modules[__name__] = _impl
