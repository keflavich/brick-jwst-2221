"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.photometry.crowdsource_catalogs_long``."""
import sys as _sys
import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as _impl
_sys.modules[__name__] = _impl
