"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.photometry.merge_catalogs``."""
import sys as _sys
import jwst_gc_pipeline.photometry.merge_catalogs as _impl
_sys.modules[__name__] = _impl
