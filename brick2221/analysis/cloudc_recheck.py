#!/usr/bin/env python
"""Cloud C (2221/002) astrometry re-check after per-exposure VIRAC2 re-anchoring.

Reports (to cloudc/astrometry_diag/cloudc_recheck/recheck.txt):
  - per filter: bulk + per-source residual vs VIRAC2 (cloudc cache, PM-propagated to 2023.30)
    and vs Gaia DR3 (the Gaia subset of the seed, already at the obs epoch); native internal
    precision (std_ra/std_dec).
  - cross-filter bulk offsets vs F405N (all should be ~few mas: one VIRAC2 frame).
Was ~+22 mas RA / ~+90 mas Dec off Gaia before the re-anchor.
"""
import glob, os, warnings
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
warnings.simplefilter('ignore')

BASE = '/orange/adamginsburg/jwst/cloudc'
OUTD = f'{BASE}/astrometry_diag/cloudc_recheck'
os.makedirs(OUTD, exist_ok=True)
EPOCH = 2023.30
FILTS = ['f405n', 'f410m', 'f466n', 'f212n', 'f182m', 'f187n']
REFFILT = 'f405n'
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
        return None
    t = Table.read(p)
    col = 'skycoord' if 'skycoord' in t.colnames else 'skycoord_centroid'
    sc = SkyCoord(t[col]).icrs
    fl = farr(t['flux_fit']) if 'flux_fit' in t.colnames else farr(t['flux'])
    ok = np.isfinite(sc.ra.deg) & np.isfinite(fl) & (fl > 0)
    return dict(sc=sc[ok], fl=fl[ok], t=t, ok=ok, path=p)


# references
v = Table.read(f'{BASE}/astrometry_diag/refcache/virac2.fits')
vra, vdec = prop(farr(v['RAJ2000']), farr(v['DEJ2000']), farr(v['pmRA']), farr(v['pmDE']), EPOCH - 2014.0)
virac = SkyCoord(vra * u.deg, vdec * u.deg)
seed = Table.read(f'{BASE}/catalogs/gaia_virac2_refcat_epoch2023.30.fits')
gm = np.array([str(s) == 'GaiaDR3' for s in seed['source']])
gaia = SkyCoord(seed['RA'][gm] * u.deg, seed['DEC'][gm] * u.deg)   # already at obs epoch
log(f"refs: VIRAC2 {len(virac)} (2014->{EPOCH}); Gaia(seed) {gm.sum()}")


def xmatch(sc, ref, ref_is_sparse=False):
    i, sep, _ = sc.match_to_catalog_sky(ref); k = sep < 0.2 * u.arcsec
    if k.sum() < 10:
        return None
    dr = (sc[k].ra.deg - ref[i[k]].ra.deg) * np.cos(np.radians(sc[k].dec.deg)) * 3.6e6
    dd = (sc[k].dec.deg - ref[i[k]].dec.deg) * 3.6e6
    m = np.hypot(dr - np.median(dr), dd - np.median(dd)) < 120; dr, dd = dr[m], dd[m]
    n = len(dr)
    return (np.median(dr), np.median(dd), stats.mad_std(dr), stats.mad_std(dd),
            stats.mad_std(dr) / np.sqrt(n), stats.mad_std(dd) / np.sqrt(n), n)


cats = {}
for filt in FILTS:
    d = load(filt)
    if d is None:
        log(f"\n{filt.upper()}: no merged catalog"); continue
    cats[filt] = d
    mag = -2.5 * np.log10(d['fl']); br = mag < np.nanpercentile(mag, 30)
    log(f"\n=== {filt.upper()} === {os.path.basename(d['path'])} ({len(d['sc'])} src)")
    for c in ('std_ra', 'std_dec'):
        if c in d['t'].colnames:
            s = farr(d['t'][c])[d['ok']]
            log(f"  native {c}: {np.nanmedian(s)*3.6e6:.2f} mas (bright {np.nanmedian(s[br])*3.6e6:.2f})")
    for name, ref in (('VIRAC2', virac), ('Gaia', gaia)):
        r = xmatch(d['sc'][br], ref)
        if r:
            log(f"  vs {name}: N={r[6]} BULK ({r[0]:+.2f},{r[1]:+.2f}) SEM({r[4]:.2f}/{r[5]:.2f}) MAD({r[2]:.1f}/{r[3]:.1f}) mas")

# cross-filter vs F405N
if REFFILT in cats:
    log(f"\n[cross-filter bulk offset vs {REFFILT.upper()} (should be ~few mas)]")
    ref = cats[REFFILT]['sc']
    for filt in FILTS:
        if filt == REFFILT or filt not in cats:
            continue
        sc = cats[filt]['sc']
        i, sep, _ = sc.match_to_catalog_sky(ref); k = sep < 0.1 * u.arcsec
        dr = (sc[k].ra.deg - ref[i[k]].ra.deg) * np.cos(np.radians(sc[k].dec.deg)) * 3.6e6
        dd = (sc[k].dec.deg - ref[i[k]].dec.deg) * 3.6e6
        m = np.hypot(dr - np.median(dr), dd - np.median(dd)) < 100; dr, dd = dr[m], dd[m]
        log(f"  {filt.upper()} - {REFFILT.upper()}: ({np.median(dr):+.2f},{np.median(dd):+.2f}) MAD({stats.mad_std(dr):.1f}/{stats.mad_std(dd):.1f}) N={len(dr)}")

open(f'{OUTD}/recheck.txt', 'w').write("\n".join(L) + "\n")
print(f"\nWrote {OUTD}/recheck.txt")
