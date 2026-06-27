#!/usr/bin/env python
"""Register the GWCS of each sickle o002 F770W crf frame directly to the NIRCam
F480M reference catalog (2026-06-13).

WHY: the crf FITS-header WCS was already corrected (per-frame, MIRIDRA), so the
PSF-photometry catalog is registered.  But the embedded ASDF *gwcs* was NEVER
corrected, and the resample step (image3 i2d, and the cataloging detection i2d)
builds mosaics from the gwcs -- so every displayed o002 F770W mosaic is ~3.35"
off F480M while the catalog underneath is fine.  The user requires astrometry-
corrected FRAMES, so we fix the gwcs at the source: any resample downstream then
comes out registered.

Method (offset-histogram, dense-field-safe -- NOT nearest-neighbor): for each
crf, detect bright sources, project with the CURRENT gwcs, histogram the
(image - F480M) offsets, take the peak, and adjust_wcs() the gwcs by the negative
of it (mirroring PipelineMIRI.fix_alignment).  Both the saved ASDF gwcs and the
FITS header are updated; MIRIGWCS guards against double-application.  Re-measures
after to confirm the residual collapses to ~0.

Run with the o002 cataloging job STOPPED (it reads these files).  Then re-run
cataloging so the detection i2d is rebuilt from the corrected gwcs.
"""
import os
import copy
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import mad_std
import astropy.units as u
from photutils.detection import DAOStarFinder
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault("CRDS_PATH", "/orange/adamginsburg/jwst/brick/crds/")
os.environ.setdefault("CRDS_SERVER_URL", "https://jwst-crds.stsci.edu")
from jwst.datamodels import ImageModel
from jwst.tweakreg.utils import adjust_wcs

import sys
# argv[1] = frame glob (default o002 F770W); argv[2] = FWHM in px (band-dependent)
FRAMEGLOB = sys.argv[1] if len(sys.argv) > 1 else \
    '/orange/adamginsburg/jwst/sickle/F770W/pipeline/jw03958002001_*_mirimage_o002_crf.fits'
REFCAT = '/orange/adamginsburg/jwst/sickle/catalogs/f480m_nrcb_indivexp_merged_resbgsub_m6_dao_basic.fits'
FWHM_PIX = float(sys.argv[2]) if len(sys.argv) > 2 else 2.445
MFSIZE = 31
THR = 5
SEARCH = 6.0
MIN_MATCH = 30    # min refine-box peak count to trust the offset (sparse-field guard)
SHIFT_CAP = 8.0   # max |total applied| arcsec; real MIRI pointing error is < this


def load_ref():
    t = Table.read(REFCAT)
    if 'skycoord' in t.colnames:
        return SkyCoord(t['skycoord'])
    return SkyCoord(t['skycoord_ra'], t['skycoord_dec'], unit='deg')


def detect(model):
    d = model.data
    mfd = median_filter(np.nan_to_num(d, nan=np.nanmedian(d)), size=MFSIZE)
    hp = np.nan_to_num(d) - mfd
    s = DAOStarFinder(threshold=THR * mad_std(hp), fwhm=FWHM_PIX)(hp)
    if s is None or len(s) == 0:
        return None, None
    xcol = 'xcentroid' if 'xcentroid' in s.colnames else 'x_centroid'
    ycol = 'ycentroid' if 'ycentroid' in s.colnames else 'y_centroid'
    return np.asarray(s[xcol]), np.asarray(s[ycol])


def measure(wcsobj, x, y, refsc):
    """Offset-histogram (image - ref) arcsec using the gwcs."""
    sc = wcsobj.pixel_to_world(x, y)
    good = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)  # off-frame -> NaN
    sc = sc[good]
    if len(sc) < 5:
        return None, None, len(sc)
    i_b, i_a, sep, _ = search_around_sky(refsc, sc, SEARCH * u.arcsec)
    if len(i_a) < 5:
        return None, None, len(i_a)
    dra = ((sc.ra[i_a] - refsc.ra[i_b]) * np.cos(sc.dec[i_a].rad)).to(u.arcsec).value
    ddec = (sc.dec[i_a] - refsc.dec[i_b]).to(u.arcsec).value
    bins = np.arange(-SEARCH, SEARCH + 0.1, 0.1)
    H, xe, ye = np.histogram2d(dra, ddec, bins=[bins, bins])
    pk = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = 0.5 * (xe[pk[0]] + xe[pk[0] + 1]), 0.5 * (ye[pk[1]] + ye[pk[1] + 1])
    m = (np.abs(dra - cx) < 0.5) & (np.abs(ddec - cy) < 0.5)
    return float(np.median(dra[m])), float(np.median(ddec[m])), int(m.sum())


def main():
    refsc = load_ref()
    print(f'reference {len(refsc)} F480M sources', flush=True)
    for fn in sorted(glob.glob(FRAMEGLOB)):
        tag = os.path.basename(fn).split('_mirimage')[0]
        # Idempotent: always converge the CURRENT gwcs residual to ~0 (so a
        # re-run after a partial/cos-biased prior correction finishes the job).
        fa = ImageModel(fn)
        wcsobj = fa.meta.wcs
        x, y = detect(fa)
        if x is None:
            print(f'{tag}: no detections; SKIP', flush=True)
            fa.close()
            continue
        dra0, ddec0, n = measure(wcsobj, x, y, refsc)
        if dra0 is None:
            print(f'{tag}: insufficient matches (n={n}); SKIP', flush=True)
            fa.close()
            continue
        # GUARD 1: too-sparse refine-box peak is unreliable (o001 brick-bg
        # diverged at n~15-135).  Require a solid peak before touching the WCS.
        if n < MIN_MATCH:
            print(f'{tag}: peak only n={n} (< {MIN_MATCH}); too sparse to trust; SKIP',
                  flush=True)
            fa.close()
            continue
        fa.meta.oldwcs = copy.copy(wcsobj)
        # adjust_wcs(delta_ra=) shifts CRVAL1 in RA-coordinate, but the measured
        # offset is TRUE-ANGULAR (includes cos dec).  Divide delta_ra by cos(dec)
        # and iterate so the cos/linearization residual converges to ~0.
        tot_ra = tot_de = 0.0
        ww = wcsobj
        rdra, rddec = dra0, ddec0
        best_resid = np.hypot(dra0, ddec0)
        diverged = False
        for _ in range(5):
            cosd = np.cos(np.deg2rad(ww.to_fits()[0]['CRVAL2']))
            ww_try = adjust_wcs(ww, delta_ra=(-rdra / cosd) * u.arcsec,
                                delta_dec=(-rddec) * u.arcsec)
            ntot_ra, ntot_de = tot_ra - rdra, tot_de - rddec
            r2, rd2, rn = measure(ww_try, x, y, refsc)
            cur = np.hypot(r2, rd2) if r2 is not None else 1e9
            # GUARD 2: total shift cap -- a real MIRI pointing error is < 8".
            # GUARD 3: residual must improve; if it grows, the peak jumped to a
            # spurious mode (divergence) -- stop and keep the last good ww.
            if abs(ntot_ra) > SHIFT_CAP or abs(ntot_de) > SHIFT_CAP or cur > best_resid + 0.1:
                diverged = True
                print(f'{tag}: DIVERGENCE guard tripped (total=({ntot_ra:+.1f},{ntot_de:+.1f})" '
                      f'resid {best_resid:.2f}->{cur:.2f}"); keeping last good', flush=True)
                break
            ww, tot_ra, tot_de = ww_try, ntot_ra, ntot_de
            rdra, rddec, best_resid = r2, rd2, cur
            if abs(rdra) < 0.05 and abs(rddec) < 0.05:
                break
        # GUARD 4: if we never applied a safe step (diverged on first try) leave
        # the file UNTOUCHED rather than write a bad WCS.
        if diverged and tot_ra == 0.0 and tot_de == 0.0:
            print(f'{tag}: no safe correction found; LEAVING UNCHANGED', flush=True)
            fa.close()
            continue
        fa.meta.wcs = ww
        fa.save(fn, overwrite=True)
        # FITS header sync
        ah = fits.open(fn)
        if 'OLCRVA1G' not in ah[1].header:
            ah[1].header['OLCRVA1G'] = ah[1].header['CRVAL1']
            ah[1].header['OLCRVA2G'] = ah[1].header['CRVAL2']
        ah[1].header.update(ww.to_fits()[0])
        prev = ah[1].header.get('MIRIGWCS', '')
        ah[1].header['MIRIGWCS'] = (f'{tot_ra:+.3f},{tot_de:+.3f}',
                                    'gwcs+FITS registered to F480M [arcsec] 2026-06-13')
        ah.writeto(fn, overwrite=True)
        fa.close()
        print(f'{tag}: init offset (image-ref)=({dra0:+.2f},{ddec0:+.2f})" n={n} '
              f'-> total applied ({tot_ra:+.2f},{tot_de:+.2f})"  '
              f'final residual=({rdra:+.2f},{rddec:+.2f})"', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
