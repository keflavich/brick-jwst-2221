#!/usr/bin/env python
"""
Register sickle MIRI o001/o002 (POS1/POS2) products to the NIRCam refcat frame
(2026-06-13).

o003 (brick-bg) was WCS-corrected earlier (apply_measured_miri_wcs_offsets.py)
but o001/o002 were not -- they carry the legacy hard-coded MIRI shift, leaving
them offset from the F480M/NIRCam refcat frame (o001 F770W measured +2.63",
-1.74" via offset-histogram).  This corrects the FITS WCS (CRVAL) of every
o001/o002 product -- the per-frame _cal/_align/_crf files used by cataloging AND
the i2d mosaics -- for all three MIRI filters, using the per-obs offset measured
on F770W (same pointing/shift for all filters of an obs).  Idempotent via MIRIDRA.

OFFSETS (filled from the offset-histogram measurement; obs -> (dRA,dDec) arcsec,
F770W-minus-refcat; correction subtracts these):
"""
import os
import glob
import sys
import numpy as np
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# obs -> (dRA, dDec) in arcsec, measured F770W image-minus-refcat
OFFSETS = {
    '001': (float(sys.argv[1]), float(sys.argv[2])),
    '002': (float(sys.argv[3]), float(sys.argv[4])),
}
FILTERS = ['F770W', 'F1130W', 'F1500W']
BASE = '/orange/adamginsburg/jwst/sickle'


def correct(fn, dra, ddec):
    # safety guard: never apply an implausible offset (sentinel / typo).  MIRI
    # registration shifts are a few arcsec; anything >30" is garbage.
    if not (np.isfinite(dra) and np.isfinite(ddec) and abs(dra) < 30 and abs(ddec) < 30):
        return 'skip'
    fh = fits.open(fn)
    # find the SCI (or only-image) HDU index
    hidx = None
    for i, h in enumerate(fh):
        if h.name == 'SCI' or (h.data is not None and getattr(h, 'data', None) is not None and h.data.ndim == 2):
            hidx = i
            if h.name == 'SCI':
                break
    if hidx is None:
        fh.close(); return False
    hd = fh[hidx].header
    if 'CRVAL1' not in hd:
        fh.close(); return False
    if 'MIRIDRA' in hd:
        fh.close(); return 'skip'
    cosd = np.cos(np.deg2rad(hd['CRVAL2']))
    hd['OLCRVAL1'] = hd['CRVAL1']
    hd['OLCRVAL2'] = hd['CRVAL2']
    hd['CRVAL1'] = hd['CRVAL1'] - dra / 3600.0 / cosd
    hd['CRVAL2'] = hd['CRVAL2'] - ddec / 3600.0
    hd['MIRIDRA'] = (-dra, 'applied RA correction [arcsec] 2026-06-13')
    hd['MIRIDDE'] = (-ddec, 'applied Dec correction [arcsec] 2026-06-13')
    hd['MIRIWCSN'] = ('FITS WCS corrected; ASDF gwcs NOT corrected',
                      'offset-histogram registration to NIRCam refcat')
    fh.writeto(fn, overwrite=True)
    return True


for obs, (dra, ddec) in OFFSETS.items():
    for filt in FILTERS:
        pdir = f'{BASE}/{filt}/pipeline'
        pats = [f'{pdir}/jw03958{obs}001_*_mirimage_cal.fits',
                f'{pdir}/jw03958{obs}001_*_mirimage_align.fits',
                f'{pdir}/jw03958-o{obs}*_mirimage_o{obs}_crf.fits',
                f'{pdir}/jw03958{obs}001_*_mirimage_o{obs}_crf.fits',
                f'{pdir}/jw03958-o{obs}_t00?_miri_*{filt.lower()}*i2d.fits',
                f'{pdir}/jw03958-o{obs}_t00?_miri_*{filt.lower()}*data_i2d.fits']
        files = sorted(set(f for p in pats for f in glob.glob(p)))
        n_ok = n_skip = 0
        for fn in files:
            r = correct(fn, dra, ddec)
            if r == 'skip':
                n_skip += 1
            elif r:
                n_ok += 1
        print(f'o{obs} {filt}: corrected {n_ok}, already-done {n_skip} '
              f'(offset {dra:+.3f},{ddec:+.3f}")', flush=True)
print('ALL DONE')
