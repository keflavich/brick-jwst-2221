"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.photometry.isochrones``."""
import sys as _sys
import jwst_gc_pipeline.photometry.isochrones as _impl
_sys.modules[__name__] = _impl
