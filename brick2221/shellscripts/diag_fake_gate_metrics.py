"""Measure the satstar seed/post-fit gate metrics at each FAKE bright-star
location (f770w_fake_bright_stars_20260622.reg) and, for contrast, at the A/B
real saturated stars.  Uses the SAME gate functions and the SAME seed_gate_image
(joint o001-002 data_i2d) the pipeline uses, so the numbers are exactly what the
gate sees.  Also reports the joint satstar MODEL peak at each location (does a
fake model actually exist there?) and a local-peak test on the coadd."""
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from scipy import ndimage
import sys
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint')
from jwst_gc_pipeline.reduction.saturated_star_finding import _seed_prominence, _seed_concentration

P = '/orange/adamginsburg/jwst/sickle/F770W/pipeline/'
BASE = 'jw03958-o001-002_t001_miri_clear-f770w-mirimage'
M6 = '_resbgsub_group_m6_daophot_basic_mergedcat'

def load(fn):
    with fits.open(fn) as h:
        ext = 'SCI' if 'SCI' in [x.name for x in h] else 0
        return h[ext].data.astype(float), wcs.WCS(h[ext].header)

gate, gw = load(P + BASE + '_data_i2d.fits')
model, mw = load(P + BASE + M6 + '_model_i2d.fits')

def read_reg(fn):
    out = []
    for line in open(fn):
        line = line.strip()
        if line.startswith('point('):
            ra, dec = line.split('(')[1].split(')')[0].split(',')[:2]
            out.append(SkyCoord(float(ra), float(dec), unit='deg'))
    return out

fakes = read_reg('/orange/adamginsburg/jwst/sickle/regions_/f770w_fake_bright_stars_20260622.reg')
real = [('A', SkyCoord(266.57431, -28.80958, unit='deg')),
        ('B', SkyCoord(266.57297, -28.80342, unit='deg'))]

def localpeak(img, x, y, rad=6):
    """Is (x,y) within `rad` px of being the local max in a 2*rad+1 box?"""
    xi, yi = int(round(float(x))), int(round(float(y)))
    sub = img[yi-rad:yi+rad+1, xi-rad:xi+rad+1]
    fin = np.isfinite(sub)
    if fin.sum() < 5:
        return np.nan, np.nan, np.nan
    val = img[yi, xi] if np.isfinite(img[yi, xi]) else np.nan
    pk = np.nanmax(sub)
    # distance from center to the peak pixel
    iy, ix = np.unravel_index(np.nanargmax(np.where(fin, sub, -np.inf)), sub.shape)
    dpk = np.hypot(ix-rad, iy-rad)
    return val, pk, dpk

def measure(label, c):
    x, y = gw.world_to_pixel(c)
    if not (np.isfinite(x) and np.isfinite(y)):
        print(f"{label:>6}: off gate grid"); return
    # estimate sat_area from DQ? we don't have per-frame DQ here; use 0 (r_sat=2)
    # The gate caps sat_area at 1600; without it use a moderate area to mimic.
    for sa_label, sa in [('sa=0', 0), ('sa=400', 400)]:
        prom, core = _seed_prominence(gate, (float(y), float(x)), sa)
        conc = _seed_concentration(gate, (float(y), float(x)), sa)
        val, pk, dpk = localpeak(gate, x, y, 6)
        # model peak in a small box
        mx, my = mw.world_to_pixel(c)
        mxi, myi = int(round(float(mx))), int(round(float(my)))
        msub = model[myi-3:myi+4, mxi-3:mxi+4]
        mpk = np.nanmax(msub) if np.isfinite(msub).any() else np.nan
        print(f"{label:>6} {sa_label:>7}: prom={prom:6.1f} core={core:8.0f} "
              f"conc={conc:5.1f} | coadd val={val:7.0f} pk={pk:7.0f} dpk={dpk:.1f}px "
              f"| MODELpk={mpk:8.0f}")

undersub = read_reg('/orange/adamginsburg/jwst/sickle/regions_/f770w_undersubtracted_saturated_stars_20260621.reg')

print("=== REAL saturated stars (must KEEP) ===")
for nm, c in real:
    measure(nm, c)
print("\n=== REAL under-subtracted saturated stars (must KEEP) ===")
for i, c in enumerate(undersub):
    measure(f"U{i+1}", c)
print("\n=== FAKE bright stars (must REJECT) ===")
for i, c in enumerate(fakes):
    measure(f"F{i+1}", c)
