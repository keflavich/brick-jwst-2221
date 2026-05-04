"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.jwebbinar_tools``."""
import sys as _sys
import jwst_gc_pipeline.reduction.jwebbinar_tools as _impl
_sys.modules[__name__] = _impl
