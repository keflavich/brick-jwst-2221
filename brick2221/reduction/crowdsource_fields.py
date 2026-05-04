"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.crowdsource_fields``."""
import sys as _sys
import jwst_gc_pipeline.reduction.crowdsource_fields as _impl
_sys.modules[__name__] = _impl
