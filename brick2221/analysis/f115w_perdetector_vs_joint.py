#!/usr/bin/env python
"""Per-detector residuals of F115W against the JOINT internal reference catalog.

Question (final F115W astrometry QA): once each detector is tied per-detector, are
there STRUCTURED residuals -- a per-detector systematic shift, or an intra-detector
spatial (distortion-residual) pattern -- when individual detections are compared to
the all-exposure JOINT catalog (self-reference, no external-catalog noise)?

Joint reference = catalogs/f115w_merged_indivexp_merged_dao_basic.fits (skycoord =
exposure-averaged position).  Per-frame detections = F115W/f115w_<det>_visit*_exp*_
daophot_basic.fits (skycoord_centroid + detector pixel x_fit/y_fit).

Outputs to astrometry_diag/f115w_perdetector_vs_joint/:
  - quiver_8SWdet.png : 8-panel dRA/dDec vs detector (x,y) -> distortion structure
  - perdetector_means.png : per-detector mean offset + scatter
  - hist.png : overall residual histograms (should be zero-centered, Gaussian)
  - summary.txt : numbers
"""
import glob, re, os, warnings
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.simplefilter('ignore')
BASE = '/orange/adamginsburg/jwst/brick'
OUTD = f'{BASE}/astrometry_diag/f115w_perdetector_vs_joint'
os.makedirs(OUTD, exist_ok=True)
# Joint reference = the all-exposure merged catalog.  Prefer the canonical no-suffix
# file; fall back to the newest iteration-merged catalog produced by the recat.
def _pick_joint():
    cand = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic.fits'
    if os.path.exists(cand):
        return cand
    its = sorted(glob.glob(f'{BASE}/catalogs/f115w_merged_indivexp_merged_*dao_basic.fits'),
                 key=os.path.getmtime)
    its = [f for f in its if 'allcols' not in f and 'i2dseed' not in f]
    if not its:
        raise FileNotFoundError("no merged F115W catalog found for joint reference")
    return its[-1]
JOINT = _pick_joint()
DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
MATCH = 0.15 * u.arcsec


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


j = Table.read(JOINT)
jsc = SkyCoord(j['skycoord']).icrs

frames = sorted(glob.glob(f'{BASE}/F115W/f115w_*_visit*_exp*_daophot_basic.fits'))
frames = [f for f in frames if not any(t in f for t in ('_m1', '_m2', '_m3', '_m4', '_m5',
                                                        '_m6', 'iter', 'resbgsub', '_group'))]

# accumulate per detector: x, y, dra, ddec (mas)
acc = {d: {'x': [], 'y': [], 'dra': [], 'ddec': []} for d in DETS}
for fn in frames:
    mr = re.search(r'(nrc[ab]\d)_visit(\d+)', fn)
    if not mr:
        continue
    det = mr.group(1)
    if det not in acc:
        continue
    t = Table.read(fn)
    if 'x_fit' in t.colnames:
        x = farr(t['x_fit']); y = farr(t['y_fit'])
    else:
        x = farr(t['xcentroid']); y = farr(t['ycentroid'])
    sc = SkyCoord(t['skycoord_centroid']).icrs
    ok = np.isfinite(sc.ra.deg) & np.isfinite(x) & np.isfinite(y)
    sc, x, y = sc[ok], x[ok], y[ok]
    if len(sc) < 10:
        continue
    i, sep, _ = sc.match_to_catalog_sky(jsc)
    k = sep < MATCH
    dra = (sc[k].ra.deg - jsc[i[k]].ra.deg) * np.cos(np.radians(sc[k].dec.deg)) * 3.6e6
    dd = (sc[k].dec.deg - jsc[i[k]].dec.deg) * 3.6e6
    # clip gross mismatches
    cl = np.hypot(dra, dd) < 100
    acc[det]['x'].append(x[k][cl]); acc[det]['y'].append(y[k][cl])
    acc[det]['dra'].append(dra[cl]); acc[det]['ddec'].append(dd[cl])

for d in DETS:
    for key in acc[d]:
        acc[d][key] = np.concatenate(acc[d][key]) if acc[d][key] else np.array([])

# --- summary numbers ---
L = ["F115W per-detector residuals vs JOINT internal catalog (self-reference)"]
L.append(f"joint ref: {JOINT} ({len(j)} sources); match<{MATCH}")
means = {}
for d in DETS:
    a = acc[d]
    if len(a['dra']) < 20:
        L.append(f"  {d}: too few ({len(a['dra'])})"); continue
    mra, mdd = np.median(a['dra']), np.median(a['ddec'])
    sra, sdd = stats.mad_std(a['dra']), stats.mad_std(a['ddec'])
    means[d] = (mra, mdd)
    L.append(f"  {d}: N={len(a['dra'])}  mean dRA={mra:+.2f} dDec={mdd:+.2f} mas  scatter {sra:.1f}/{sdd:.1f}")
if means:
    ar = np.array([m[0] for m in means.values()]); ad = np.array([m[1] for m in means.values()])
    L.append(f"-> per-detector mean spread: dRA std={np.std(ar):.2f} (range {ar.max()-ar.min():.1f}), "
             f"dDec std={np.std(ad):.2f} (range {ad.max()-ad.min():.1f}) mas")
    L.append("   (structured if spread >> ~2 mas internal scatter / sqrt(N); else detectors consistent => DONE)")
open(f'{OUTD}/summary.txt', 'w').write("\n".join(L) + "\n")
print("\n".join(L))

# --- quiver per detector (binned) ---
fig, axs = plt.subplots(2, 4, figsize=(18, 9))
for ax, d in zip(axs.ravel(), DETS):
    a = acc[d]
    if len(a['dra']) < 20:
        ax.set_title(f"{d}: n/a"); continue
    nb = 8; edges = np.linspace(0, 2048, nb + 1); cx = 0.5 * (edges[:-1] + edges[1:])
    gx, gy, gu, gv = [], [], [], []
    for ix in range(nb):
        for iy in range(nb):
            m = ((a['x'] >= edges[ix]) & (a['x'] < edges[ix + 1]) &
                 (a['y'] >= edges[iy]) & (a['y'] < edges[iy + 1]))
            if m.sum() >= 5:
                gx.append(cx[ix]); gy.append(cx[iy])
                gu.append(np.median(a['dra'][m])); gv.append(np.median(a['ddec'][m]))
    ax.quiver(gx, gy, gu, gv, angles='xy', scale_units='xy', scale=0.05, width=0.004)
    ax.set_title(f"{d}: mean({np.median(a['dra']):+.1f},{np.median(a['ddec']):+.1f}) mas, N={len(a['dra'])}")
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
    ax.set_xlabel('det x'); ax.set_ylabel('det y')
fig.suptitle('F115W per-detector residual vs joint catalog (binned medians; scale: 1px arrow=0.05mas... see scale)')
plt.tight_layout(); plt.savefig(f'{OUTD}/quiver_8SWdet.png', dpi=110); plt.close()

# --- per-detector means ---
fig, ax = plt.subplots(figsize=(8, 6))
for d in DETS:
    if d in means:
        ax.scatter(means[d][0], means[d][1], s=80); ax.annotate(d, means[d])
ax.axhline(0, c='r', lw=.5); ax.axvline(0, c='r', lw=.5)
ax.set_xlabel('mean dRA (mas)'); ax.set_ylabel('mean dDec (mas)')
ax.set_title('F115W per-detector mean offset vs joint catalog'); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig(f'{OUTD}/perdetector_means.png', dpi=120); plt.close()

# --- overall hist ---
allra = np.concatenate([acc[d]['dra'] for d in DETS if len(acc[d]['dra'])])
alldd = np.concatenate([acc[d]['ddec'] for d in DETS if len(acc[d]['ddec'])])
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(allra, bins=100, range=(-50, 50), alpha=.6, label=f'dRA (MAD {stats.mad_std(allra):.1f})')
ax.hist(alldd, bins=100, range=(-50, 50), alpha=.6, label=f'dDec (MAD {stats.mad_std(alldd):.1f})')
ax.legend(); ax.set_xlabel('residual vs joint (mas)')
ax.set_title('F115W all-detector residual vs joint catalog')
plt.tight_layout(); plt.savefig(f'{OUTD}/hist.png', dpi=120); plt.close()
print(f"\nWrote plots to {OUTD}/")
