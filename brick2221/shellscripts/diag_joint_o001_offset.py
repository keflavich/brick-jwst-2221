"""Diagnose whether joint o001-002 mosaic's o001 half is rotated/shifted vs the
individual o001 i2d (the 'correct' reduction). Multi-star cross-match across the
o001 tile, NOT a single near-pivot star."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

P = '/orange/adamginsburg/jwst/sickle/F770W/pipeline/'
indiv = P + 'jw03958-o001_t001_miri_clear-f770w-mirimage_data_i2d.fits'
joint = P + 'jw03958-o001-002_t001_miri_clear-f770w-mirimage_data_i2d.fits'

def load(fn):
    h = fits.open(fn)
    # SCI ext
    for i, hdu in enumerate(h):
        if hdu.data is not None and hdu.data.ndim == 2:
            return hdu.data, WCS(hdu.header), h[i].header
    raise RuntimeError(fn)

di, wi, hi = load(indiv)
dj, wj, hj = load(joint)

def pa(w):
    cd = w.pixel_scale_matrix
    return np.degrees(np.arctan2(cd[0,1], cd[1,1]))

print(f"indiv o001 i2d shape={di.shape} PA={pa(wi):.2f}")
print(f"joint     i2d shape={dj.shape} PA={pa(wj):.2f}")

# detect bright stars in individual o001 i2d
mean, med, std = sigma_clipped_stats(di, sigma=3.0)
finder = DAOStarFinder(fwhm=3.0, threshold=30*std)
src = finder(di - med)
src.sort('flux')
src.reverse()
src = src[:40]
print(f"\n{len(src)} bright stars in indiv o001 i2d (top by flux)")

# their sky coords
sky = wi.pixel_to_world(src['xcentroid'], src['ycentroid'])

# for each, find nearest peak in joint mosaic within a search box
def refine(data, w, sc, box=8):
    x, y = w.world_to_pixel(sc)
    x, y = float(x), float(y)
    if not (box < x < data.shape[1]-box and box < y < data.shape[0]-box):
        return None
    sub = data[int(y-box):int(y+box+1), int(x-box):int(x+box+1)]
    if not np.isfinite(sub).any():
        return None
    iy, ix = np.unravel_index(np.nanargmax(sub), sub.shape)
    peak = sub[iy, ix]
    gx = int(x-box)+ix
    gy = int(y-box)+iy
    return w.pixel_to_world(gx, gy), peak

offs = []
print(f"\n{'#':>2} {'RA':>11} {'Dec':>11} {'offset\"':>9} {'jpeak':>9}")
for k in range(len(src)):
    sc = sky[k]
    r = refine(dj, wj, sc, box=10)
    if r is None:
        continue
    scj, peak = r
    off = sc.separation(scj).arcsec
    offs.append(off)
    if k < 25:
        print(f"{k:>2} {sc.ra.deg:11.5f} {sc.dec.deg:11.5f} {off:9.3f} {peak:9.1f}")

offs = np.array(offs)
print(f"\nN matched={len(offs)}  median offset={np.median(offs):.3f}\"  "
      f"mean={np.mean(offs):.3f}\"  max={np.max(offs):.3f}\"")
print(f"offsets >1\": {np.sum(offs>1)}/{len(offs)}   >3\": {np.sum(offs>3)}/{len(offs)}")
