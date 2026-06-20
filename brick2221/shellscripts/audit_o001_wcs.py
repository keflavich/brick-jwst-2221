"""Audit ALL sickle F770W o001 image products for the 37deg-rotation bad WCS.
Correct CD-matrix PA ~ -79.78 (matches PA_APER 280.215). Broken ~ -116.9.
Reports every file with a celestial WCS, flags PA more than 5deg off -79.78."""
import glob, os, sys, numpy as np, time
from astropy.io import fits
from astropy.wcs import WCS
import warnings
warnings.filterwarnings('ignore')

PIPE = '/orange/adamginsburg/jwst/sickle/F770W/pipeline'
GOOD_PA = -79.78
TOL = 5.0

# every F770W o001 product that could carry a celestial WCS
pats = [
    'jw03958001001_*_mirimage_o001_crf.fits',
    'jw03958001001_*_mirimage_cal.fits',
    'jw03958001001_*_mirimage_align.fits',
    'jw03958001001_*_mirimage_rate.fits',
    'jw03958-o001_t001_miri_f770w*i2d.fits',
    'jw03958-o001_t001_miri_clear-f770w-mirimage*i2d.fits',
    'jw03958-o001_t001_miri_f770w_?_o001_crf.fits',
    'jw03958-o001-002_t001_miri_clear-f770w-mirimage*i2d.fits',  # joint (mixed o001/o002)
]
seen = set(); rows = []
for p in pats:
    for fn in sorted(glob.glob(os.path.join(PIPE, p))):
        if fn in seen: continue
        seen.add(fn)
        try:
            hl = fits.open(fn)
        except Exception as e:
            rows.append((fn, None, f'OPEN-ERR {e}')); continue
        pa = None; note = ''
        for hdu in hl:
            if hdu.data is not None and getattr(hdu.data, 'ndim', 0) == 2 and 'CRVAL1' in hdu.header:
                try:
                    w = WCS(hdu.header); cd = w.pixel_scale_matrix
                    pa = np.degrees(np.arctan2(cd[0, 1], cd[1, 1]))
                except Exception as e:
                    note = f'WCS-ERR {e}'
                break
        else:
            note = 'no-CRVAL-WCS (GWCS-only?)'
        rows.append((fn, pa, note))

print(f"{'PA':>9}  {'status':6}  {'mtime':12}  file")
nbad = 0
for fn, pa, note in rows:
    mt = time.strftime('%m-%d_%H:%M', time.localtime(os.path.getmtime(fn)))
    if pa is None:
        status = '  ?  '
    elif abs(((pa - GOOD_PA + 180) % 360) - 180) > TOL:
        status = 'BROKEN'; nbad += 1
    else:
        status = ' ok '
    pas = f'{pa:9.2f}' if pa is not None else '    --   '
    print(f"{pas}  {status:6}  {mt}  {os.path.basename(fn)}  {note}")
print(f"\n{nbad} BROKEN (PA >{TOL}deg from {GOOD_PA})")
