"""cloudc F2550W (prop 2221 obs 001) crf are PRODUCT-named
(jw02221-o001_t001_miri_f2550w_<N>_o001_crf.fits) because the asn product name
is set before image3.  Cataloging globs PER-EXPOSURE crf
(jw02221001001_*_mirimage_o001_crf.fits).  Map product->per-exposure by EXPSTART
(1:1) against the existing per-exposure *_mirimage_cal.fits, and copy the product
crf into the per-exposure name so get_filenames finds them.  Idempotent: skips a
per-exposure crf that already exists.

Usage: python rename_cloudc_f2550w_crf_to_perexposure.py
"""
import glob, os, shutil
from astropy.io import fits

PDIR = '/orange/adamginsburg/jwst/cloudc/F2550W/pipeline'

def expstart(fn):
    h = fits.getheader(fn)
    es = h.get('EXPSTART')
    if es is None and len(fits.open(fn)) > 1:
        es = fits.getheader(fn, 1).get('EXPSTART')
    return round(float(es), 6)

# per-exposure frame bases keyed by EXPSTART (from the cal files)
cal = sorted(glob.glob(f'{PDIR}/jw02221001001_*_mirimage_cal.fits'))
base_by_es = {}
for c in cal:
    base = os.path.basename(c)[:-len('_cal.fits')]  # jw02221001001_03201_00001_mirimage
    base_by_es[expstart(c)] = base

prod = sorted(glob.glob(f'{PDIR}/jw02221-o001_t001_miri_f2550w_*_o001_crf.fits'))
print(f"{len(prod)} product crf, {len(base_by_es)} per-exposure bases")
made = miss = skip = 0
for p in prod:
    es = expstart(p)
    base = base_by_es.get(es)
    if base is None:
        print(f"  NO MATCH for {os.path.basename(p)} (EXPSTART={es})"); miss += 1; continue
    dst = f'{PDIR}/{base}_o001_crf.fits'
    if os.path.exists(dst):
        skip += 1; continue
    shutil.copy2(p, dst); made += 1
print(f"made {made}, skipped {skip}, unmatched {miss}")
print("per-exposure crf now:", len(glob.glob(f'{PDIR}/jw02221001001_*_mirimage_o001_crf.fits')))
