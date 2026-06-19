"""
Canonical daophot-BASIC per-filter combined catalog selector for the Brick.

The legacy base `<filt>_merged_indivexp_merged_dao_basic.fits` (Jun-7) is STALE/INCOMPLETE
(e.g. F200W had only 154/192 frames; nrca4 nearly absent -> coverage holes in diagnostics).
The current complete products are the manual-pipeline passes `<filt>_merged_indivexp_merged_<m>_dao_basic.fits`
(192 frames for SW, 48 for LW). Astrometric positions are stable across passes (F200W m2 vs m3 MAD
0.12 mas; F182M m2 vs m7 ≤1.8 mas), so the choice is astrometrically irrelevant -- pick the latest
COMPLETE, deduped pass per filter. All crowdsource catalogs are deprecated.

`best_dao_basic(filt)` returns the canonical complete catalog path. Override CANON if the project
designates a different final pass.
"""
import os

CATDIR = '/orange/adamginsburg/jwst/brick/catalogs'

# latest complete (192 SW / 48 LW) deduped manual pass per filter, from the 2026-06-18 audit.
CANON = {
    'f115w': '_m3', 'f182m': '_m3', 'f187n': '_m3', 'f200w': '_m3', 'f212n': '_m3',
    'f356w': '_m2', 'f405n': '_m3', 'f410m': '_m3', 'f444w': '_m2', 'f466n': '_m3',
}


# EXACT per-exposure module-lock (relock_exposures.py): undo the recorded per-detector offset ->
# SIAF -> one VIRAC2-tied shift per exposure. This is exact (not VIRAC2-per-detector-noise-limited),
# so it does NOT degrade the already-consistent 2221 bands (F182M cross-frame scatter 2.3 mas) and
# puts ALL filters on one common VIRAC2-referenced per-exposure solution. Use LOCKED for every
# relocked band. (The earlier note about 2221 degrading applied to the VIRAC2-DEVIATION method,
# now superseded by the exact relock.)
LOCK_FILTERS = {'f115w', 'f200w', 'f356w', 'f444w', 'f182m', 'f187n', 'f212n', 'f405n', 'f410m', 'f466n'}


def best_dao_basic(filt):
    filt = filt.lower()
    # prefer the module-LOCKED combined catalog (lock_exposures.py) for 1182 bands only
    locked = f'{CATDIR}/{filt}_merged_indivexp_LOCKED_dao_basic.fits'
    if filt in LOCK_FILTERS and os.path.exists(locked):
        return locked
    tag = CANON.get(filt, '_m3')
    path = f'{CATDIR}/{filt}_merged_indivexp_merged{tag}_dao_basic.fits'
    if not os.path.exists(path):
        # fall back to base only if the chosen pass is missing
        alt = f'{CATDIR}/{filt}_merged_indivexp_merged_dao_basic.fits'
        if os.path.exists(alt):
            return alt
    return path
