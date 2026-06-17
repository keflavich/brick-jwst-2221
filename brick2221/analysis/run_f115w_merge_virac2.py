#!/usr/bin/env python
"""Production F115W indivexp merge on the VIRAC2 frame, via the pipeline's own
merge_individual_frames + the validated per-frame offsets table.

Drop-in replacement for catalogs/f115w_merged_indivexp_merged_dao_basic.fits
(backed up under catalogs/backup_preVIRAC2_20260617/). dao/basic, indivexp, merged.
"""
import sys
sys.path.insert(0, '/orange/adamginsburg/repos/jwst-gc-pipeline')
import numpy as np
from astropy.table import Table
from jwst_gc_pipeline.photometry.merge_catalogs import merge_individual_frames

BASE = '/orange/adamginsburg/jwst/brick/'
OFFSETS = f'{BASE}/offsets/Offsets_JWST_Brick1182_F115W_VIRAC2frame.csv'

ot = Table.read(OFFSETS)
print(f"offsets table: {len(ot)} F115W frames; cols {ot.colnames}")

merge_individual_frames(
    module='merged', desat=False, filtername='f115w', progid='1182',
    bgsub=False, epsf=False, fitpsf=False, blur=False,
    suffix='_basic', method='dao', target='brick',
    exposure_numbers=np.arange(1, 25),
    offsets_table=ot,
    iteration_label=None, resbgsub=False,
    basepath=BASE,
)
print("DONE merge_individual_frames F115W dao basic (VIRAC2 frame)")
