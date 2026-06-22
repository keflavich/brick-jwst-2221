#!/usr/bin/env python
"""F115W astrometry re-check after the per-exposure-locked re-reduction.

Reports (to astrometry_diag/f115w_recheck/):
  - bulk + per-source residuals vs VIRAC2 (2014.0) and Gaia DR3 (bright subset)
  - native internal precision (std_ra/std_dec across exposures)
  - CROSS-FILTER consistency: F115W vs F200W/F356W/F444W bulk offset (all should be on
    one VIRAC2 frame after the per-exposure relock -> ~few mas)
Pairs with f115w_perdetector_vs_joint.py (per-detector / intra-detector structure).
"""
import glob, os, warnings
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
warnings.simplefilter('ignore')

BASE = '/orange/adamginsburg/jwst/brick'
OUTD = f'{BASE}/astrometry_diag/f115w_recheck'
os.makedirs(OUTD, exist_ok=True)
EPOCH = 2022.703
L = []


def log(s):
    print(s, flush=True); L.append(s)


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def prop(ra, dec, pr, pd, dt):
    pr = np.where(np.isfinite(pr), pr, 0.); pd = np.where(np.isfinite(pd), pd, 0.)
    return ra + (pr * dt / 3.6e6) / np.cos(np.radians(dec)), dec + pd * dt / 3.6e6


def newest_merged(filt):
    fs = sorted(glob.glob(f'{BASE}/catalogs/{filt}_merged_indivexp_merged_*dao_basic.fits'),
                key=os.path.getmtime)
    fs = [f for f in fs if 'allcols' not in f and 'i2dseed' not in f]
    return fs[-1] if fs else None


def load(filt):
    p = newest_merged(filt)
    if p is None:
        return None, None, None
    t = Table.read(p)
    col = 'skycoord' if 'skycoord' in t.colnames else 'skycoord_centroid'
    sc = SkyCoord(t[col]).icrs
    fl = farr(t['flux_fit']) if 'flux_fit' in t.colnames else farr(t['flux'])
    ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
    return sc[ok], fl[ok], (t, ok, p)


def robust(dr, dd):
    return (np.median(dr), np.median(dd), stats.mad_std(dr), stats.mad_std(dd), len(dr))


# --- references ---
v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
vra, vdec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2014.0)
virac = SkyCoord(vra * u.deg, vdec * u.deg)
gaia = None
try:
    g = Table.read(f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits')
    gra, gdec = prop(farr(g['RA_ICRS']), farr(g['DE_ICRS']), farr(g['pmRA']), farr(g['pmDE']), EPOCH - 2016.0)
    gf = np.isfinite(gra) & np.isfinite(gdec); gaia = SkyCoord(gra[gf] * u.deg, gdec[gf] * u.deg)
except Exception as e:
    log(f"(gaia unavailable: {e})")

sc, fl, meta = load('f115w')
if sc is None:
    log("NO F115W merged catalog found"); open(f'{OUTD}/recheck.txt', 'w').write("\n".join(L)); raise SystemExit
t, ok, path = meta
log(f"=== F115W re-check ===\njoint catalog: {os.path.basename(path)} ({len(sc)} sources)")
mag = -2.5 * np.log10(fl); br = mag < np.nanpercentile(mag, 20)

# native internal precision
for c in ('std_ra', 'std_dec'):
    if c in t.colnames:
        s = farr(t[c])[ok]
        log(f"native {c}: median {np.nanmedian(s)*3.6e6:.2f} mas (bright {np.nanmedian(s[br])*3.6e6:.2f})")


def xmatch(name, ref, bright=True):
    s = sc[br] if bright else sc
    i, sep, _ = s.match_to_catalog_sky(ref); k = sep < 0.2 * u.arcsec
    dr = (s[k].ra.deg - ref[i[k]].ra.deg) * np.cos(np.radians(s[k].dec.deg)) * 3.6e6
    dd = (s[k].dec.deg - ref[i[k]].dec.deg) * 3.6e6
    m = np.hypot(dr - np.median(dr), dd - np.median(dd)) < 120; dr, dd = dr[m], dd[m]
    r = robust(dr, dd)
    log(f"vs {name}: N={r[4]} BULK ({r[0]:+.2f},{r[1]:+.2f}) mas SEM({r[2]/np.sqrt(r[4]):.2f}/{r[3]/np.sqrt(r[4]):.2f}) per-source MAD({r[2]:.1f}/{r[3]:.1f})")


log("\n[absolute reference residuals, bright 20%]")
xmatch("VIRAC2(2014->2022.7)", virac)
if gaia is not None:
    xmatch("Gaia DR3", gaia)

# cross-filter consistency (all should be on one VIRAC2 frame)
log("\n[cross-filter bulk offset: F115W - <other> (should be ~few mas)]")
for of in ('f200w', 'f356w', 'f444w'):
    osc, ofl, ometa = load(of)
    if osc is None:
        log(f"  {of}: no merged catalog"); continue
    i, sep, _ = sc[br].match_to_catalog_sky(osc); k = sep < 0.1 * u.arcsec
    dr = (sc[br][k].ra.deg - osc[i[k]].ra.deg) * np.cos(np.radians(sc[br][k].dec.deg)) * 3.6e6
    dd = (sc[br][k].dec.deg - osc[i[k]].dec.deg) * 3.6e6
    mm = np.hypot(dr - np.median(dr), dd - np.median(dd)) < 100; dr, dd = dr[mm], dd[mm]
    log(f"  F115W - {of.upper()}: ({np.median(dr):+.2f},{np.median(dd):+.2f}) mas MAD({stats.mad_std(dr):.1f}/{stats.mad_std(dd):.1f}) N={len(dr)}  [{os.path.basename(ometa[2])}]")

open(f'{OUTD}/recheck.txt', 'w').write("\n".join(L) + "\n")
print(f"\nWrote {OUTD}/recheck.txt")
