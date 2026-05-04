"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.saturated_star_finding``."""
import sys as _sys
import jwst_gc_pipeline.reduction.saturated_star_finding as _impl
_sys.modules[__name__] = _impl
