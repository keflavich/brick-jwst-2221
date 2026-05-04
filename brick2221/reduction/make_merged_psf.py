"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.make_merged_psf``."""
import sys as _sys
import jwst_gc_pipeline.reduction.make_merged_psf as _impl
_sys.modules[__name__] = _impl
