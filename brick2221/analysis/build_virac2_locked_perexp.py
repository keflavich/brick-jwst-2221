#!/usr/bin/env python
"""Per-EXPOSURE module-locked VIRAC2 offsets (relative-internal + bulk-absolute).

Motivation (measured 2026-06-20, F115W):
  Per-VISIT locking (one shift/visit) leaves real per-exposure jitter -- most exposures
  ~1.5 mas, but individual exposures up to ~7 mas (e.g. 1182 visit001 exp11/12 at
  dDec ~-8 vs visit -1.8).  That blurs those exposures in the mosaic.  Naive per-exposure
  VIRAC2 solving instead injects ~2.4 mas/exposure VIRAC2 noise (only ~215 matches/exp).

Solution -- decouple the two so we get BOTH:
  1. Per-exposure RELATIVE shift vs the dense INTERNAL per-visit consensus (thousands of
     stars, sub-mas) -> removes jitter precisely, no VIRAC2 noise.
  2. ONE per-VISIT BULK absolute tie consensus->VIRAC2 (all visit stars pooled, ~0.7 mas)
     -> sets the zero point and absorbs the per-visit guide-star pointing error (~17"/~2").
  shift[visit,exp] = bulk[visit] + relative[visit,exp]
  Applied to the pristine assign_wcs (SIAF) cal frame this lands every exposure on VIRAC2
  with jitter removed; module-locked (one shift for all 8 detectors -> SIAF lock intact).

Like relock_exposures.py, SIAF positions are recovered by UNDOing the recorded per-detector
RAOFFSET/DEOFFSET in each per-frame catalog meta.

Output: offsets/Offsets_JWST_Brick<prop>_VIRAC2locked.csv, keyed (Visit, Exposure, Filter)
with Exposure as INT (matches fix_alignment's per-exposure lookup).  Existing rows for
other filters are preserved.
"""
import sys, glob, os, re
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import warnings
warnings.filterwarnings('ignore')

BASE = '/orange/adamginsburg/jwst/brick'
OFFDIR = f'{BASE}/offsets'
VIRAC2CACHE = f'{BASE}/astrometry_diag/refcache/virac2.fits'
V2EP = 2014.0
SEARCH = 0.3 * u.arcsec
CLIP_MAS = 60.0
CLUSTER_MAS = 50.0

FILT_CFG = {  # filt: (subdir, vgroup, obs-epoch, mtag, proposal)
    'f115w': ('F115W', '02101', 2022.703, '_m3', '1182'),
    'f200w': ('F200W', '04101', 2022.703, '_m3', '1182'),
}
DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def virac2(epoch):
    v = Table.read(VIRAC2CACHE)
    ra = farr(v['RAJ2000']); dec = farr(v['DEJ2000'])
    pr = np.where(np.isfinite(farr(v['pmRA'])), farr(v['pmRA']), 0.)
    pd = np.where(np.isfinite(farr(v['pmDE'])), farr(v['pmDE']), 0.)
    dt = epoch - V2EP
    return SkyCoord((ra + (pr * dt / 3.6e6) / np.cos(np.radians(dec))) * u.deg,
                    (dec + pd * dt / 3.6e6) * u.deg)


def coord_shift(ra, dec, ref):
    """clipped-median Δα/Δδ COORDINATE offset (arcsec, NO cosδ) to ADD to (ra,dec) to land
    on ref -- the convention adjust_wcs(delta_ra/delta_dec) consumes.  Clipping is on-sky
    (isotropic).  Returns (dra_as, ddec_as, sem_ra_mas, sem_dec_mas, n)."""
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    idx, sep, _ = sc.match_to_catalog_sky(ref)
    m = sep < SEARCH
    if m.sum() < 15:
        return None
    a = sc[m]; b = ref[idx[m]]
    dra_c = (b.ra - a.ra).to(u.arcsec).value          # Δα coordinate, arcsec
    ddec_c = (b.dec - a.dec).to(u.arcsec).value
    onsky = np.hypot((dra_c * np.cos(a.dec.radian)), ddec_c) * 1000.  # mas, for clip
    md, mdd = np.median(dra_c), np.median(ddec_c)
    cl = np.hypot((dra_c - md) * np.cos(np.radians(-28.7)), ddec_c - mdd) * 1000. < CLIP_MAS
    n = int(cl.sum())
    return (float(np.median(dra_c[cl])), float(np.median(ddec_c[cl])),
            mad_std(dra_c[cl] * np.cos(np.radians(-28.7))) * 1000.0 / np.sqrt(n),  # arcsec->mas
            mad_std(ddec_c[cl]) * 1000.0 / np.sqrt(n), n)


def build_consensus(frames):
    """Welford incremental combine of frame SIAF positions -> consensus SkyCoord (per-visit)."""
    cosd = np.cos(np.radians(-28.70)); rad = CLUSTER_MAS / 1000. / 3600.
    cap = sum(len(fr[0]) for fr in frames) + 10
    g_ra = np.empty(cap); g_dec = np.empty(cap); g_n = np.zeros(cap, int); ng = 0
    for (fra, fdec) in frames:
        if ng == 0:
            n = len(fra); g_ra[:n] = fra; g_dec[:n] = fdec; g_n[:n] = 1; ng = n; continue
        base = SkyCoord(g_ra[:ng] * u.deg, g_dec[:ng] * u.deg)
        idx, sep, _ = SkyCoord(fra * u.deg, fdec * u.deg).match_to_catalog_sky(base)
        mt = sep.deg < rad; gi = idx[mt]
        g_n[gi] += 1
        g_ra[gi] += (fra[mt] - g_ra[gi]) / g_n[gi]
        g_dec[gi] += (fdec[mt] - g_dec[gi]) / g_n[gi]
        um = ~mt; k = int(um.sum())
        g_ra[ng:ng+k] = fra[um]; g_dec[ng:ng+k] = fdec[um]; g_n[ng:ng+k] = 1; ng += k
    sel = g_n[:ng] >= 2   # consensus = stars seen in >=2 frames (jitter-averaged)
    return SkyCoord(g_ra[:ng][sel] * u.deg, g_dec[:ng][sel] * u.deg)


def load_siaf(f):
    """Recover SIAF (pristine assign_wcs) positions by undoing the recorded per-detector
    RAOFFSET/DEOFFSET (Δα-coordinate arcsec).  Returns (ra_siaf, dec_siaf, ra0, de0)."""
    t = Table.read(f)
    sc = SkyCoord(t['skycoord_centroid'])
    ra0 = float(t.meta.get('RAOFFSET', 0.0)); de0 = float(t.meta.get('DEOFFSET', 0.0))  # arcsec applied
    fl = farr(t['flux_fit']); q = farr(t['qfit']) if 'qfit' in t.colnames else np.zeros(len(t))
    good = np.isfinite(fl) & (fl > 0) & (q < 0.4) & np.isfinite(sc.ra.deg)
    return sc.ra.deg[good] - ra0 / 3600.0, sc.dec.deg[good] - de0 / 3600.0, ra0, de0


def lock_filter(filt):
    sub, vg, ep, mtag, prop = FILT_CFG[filt]
    print(f"=== per-exposure relock {filt} ({prop}, epoch {ep}) ===", flush=True)
    ref = virac2(ep)
    from collections import defaultdict
    # gather SIAF positions per (visit, exp) pooled over detectors; track RAOFFSET for coarse
    byve = defaultdict(lambda: [[], []]); byv = defaultdict(list); coarse = defaultdict(lambda: [[], []])
    for det in DETS:
        for f in glob.glob(f'{BASE}/{sub}/{filt}_{det}_visit*_vgroup{vg}_exp*{mtag}_daophot_basic.fits'):
            b = os.path.basename(f)
            vis = b.split('_visit')[1][:3]; exp = int(re.search(r'_exp(\d+)', b).group(1))
            ra, dec, ra0, de0 = load_siaf(f)
            byve[(vis, exp)][0].append(ra); byve[(vis, exp)][1].append(dec)
            byv[vis].append((ra, dec))
            coarse[vis][0].append(ra0); coarse[vis][1].append(de0)
    rows = []
    for vis in sorted(byv):
        # coarse per-visit shift (Δα-coord arcsec) = median recorded RAOFFSET -> bridges the
        # ~17.5" early-mission guide-star offset so SIAF lands near VIRAC2 for matching
        c_ra = float(np.median(coarse[vis][0])); c_dec = float(np.median(coarse[vis][1]))
        consensus = build_consensus(byv[vis])                      # SIAF frame
        # apply coarse to consensus, then match VIRAC2 for the fine residual (Δα-coord arcsec)
        cc_ra = consensus.ra.deg + c_ra / 3600.0; cc_dec = consensus.dec.deg + c_dec / 3600.0
        res = coord_shift(cc_ra, cc_dec, ref)
        if res is None:
            print(f"  visit{vis}: bulk fine tie failed (coarse {c_ra:.3f},{c_dec:.3f})"); continue
        bulk_ra = c_ra + res[0]; bulk_dec = c_dec + res[1]          # SIAF->VIRAC2 bulk, arcsec coord
        print(f"  visit{vis}: coarse({c_ra:.4f},{c_dec:.4f})\" + fine_resid({res[0]*1000:+.1f},{res[1]*1000:+.1f})mas "
              f"=> BULK ({bulk_ra:.4f},{bulk_dec:.4f})\" SEM {res[2]:.2f}/{res[3]:.2f}mas n={res[4]}; "
              f"consensus={len(consensus)} stars", flush=True)
        exps = sorted(e for (v, e) in byve if v == vis)
        for exp in exps:
            ra = np.concatenate(byve[(vis, exp)][0]); dec = np.concatenate(byve[(vis, exp)][1])
            rel = coord_shift(ra, dec, consensus)   # relative exp->consensus (internal, dense, arcsec coord)
            if rel is None:
                print(f"    exp{exp}: relative failed"); continue
            tot_ra = bulk_ra + rel[0]; tot_dec = bulk_dec + rel[1]   # SIAF->VIRAC2, arcsec coord
            rows.append(dict(Visit=f'jw0{prop}004{vis}' if prop == '1182' else f'jw0{prop}001{vis}',
                             Exposure=int(exp), Filter=filt.upper(),
                             dra=tot_ra, ddec=tot_dec, nmatch=rel[4],
                             rel_ra_mas=rel[0] * 1000, rel_dec_mas=rel[1] * 1000))
            print(f"    exp{exp:>2}: rel({rel[0]*1000:+.2f},{rel[1]*1000:+.2f})mas SEM {rel[2]:.2f}/{rel[3]:.2f} n={rel[4]}"
                  f"  -> total({tot_ra:.4f},{tot_dec:.4f})\"", flush=True)
    return rows, prop


if __name__ == '__main__':
    filts = sys.argv[1:] or ['f115w']
    allrows = {}
    for f in filts:
        rows, prop = lock_filter(f)
        allrows.setdefault(prop, []).extend(rows)
    os.makedirs(OFFDIR, exist_ok=True)
    for prop, rows in allrows.items():
        if not rows:
            continue
        t = Table(rows)
        t['dra (arcsec)'] = t['dra']; t['ddec (arcsec)'] = t['ddec']
        path = f'{OFFDIR}/Offsets_JWST_Brick{prop}_VIRAC2locked.csv'
        newfilts = set(str(x) for x in t['Filter'])
        if os.path.exists(path):
            old = Table.read(path)
            keep = ~np.isin([str(x) for x in old['Filter']], list(newfilts))
            if keep.any():
                # align columns
                for c in t.colnames:
                    if c not in old.colnames:
                        old[c] = np.nan
                for c in old.colnames:
                    if c not in t.colnames:
                        t[c] = np.nan
                t = vstack([old[keep], t])
        t.write(path, overwrite=True)
        print(f"\nwrote {path}: {len(t)} rows ({sorted(newfilts)} replaced per-exposure)", flush=True)
