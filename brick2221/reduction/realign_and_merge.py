"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.realign_and_merge``."""
import sys as _sys
import jwst_gc_pipeline.reduction.realign_and_merge as _impl
_sys.modules[__name__] = _impl
