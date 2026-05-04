"""Backward-compat shim. Implementation moved to ``jwst_gc_pipeline.reduction.merge_a_plus_b``."""
import sys as _sys
import jwst_gc_pipeline.reduction.merge_a_plus_b as _impl
_sys.modules[__name__] = _impl
