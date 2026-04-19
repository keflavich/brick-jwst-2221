#!/usr/bin/env python
"""Compatibility wrapper for the renamed reference-catalog builder.

This script preserves the old entrypoint name and forwards to
make_reference_from_pipeline_catalogs.py, which now supports multiple targets
and defaults to Sickle.
"""

try:
    from brick2221.analysis.make_reference_from_pipeline_catalogs import main
except ImportError:
    from make_reference_from_pipeline_catalogs import main


if __name__ == "__main__":
    main()
