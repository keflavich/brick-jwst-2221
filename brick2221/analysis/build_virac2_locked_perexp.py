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
  2. ONE per-VISIT BULK absolute tie consensus->VIRAC2 (all visit stars pooled, ~0.5 mas)
     -> sets the zero point and absorbs the per-visit guide-star pointing error (~17"/~2").
  shift[visit,exp] = bulk[visit] + relative[visit,exp]
  Applied to the pristine assign_wcs (SIAF) cal frame this lands every exposure on VIRAC2
  with jitter removed; module-locked (one shift for all detectors -> SIAF lock intact).

SIAF positions are recovered by UNDOing the recorded per-detector RAOFFSET/DEOFFSET in each
per-frame catalog meta.  The coarse per-visit shift (median RAOFFSET) bridges the large
guide-star offset so the consensus matches VIRAC2.

Output: <basepath>/offsets/Offsets_JWST_Brick<prop>_VIRAC2locked.csv, keyed (Visit, Exposure,
Filter) with Exposure INT (matches fix_alignment's per-exposure lookup).  Rows for OTHER
filters AND other (proposal,field) Visit prefixes are PRESERVED (field-safe), so a per-field
run does not clobber another field that shares the per-proposal table.

Usage:  build_virac2_locked_perexp.py [--region 1182|cloudc] [filt ...]
"""
import sys, glob, os, re, argparse
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.stats import mad_std
import warnings
warnings.filterwarnings('ignore')

V2EP = 2014.0
SEARCH = 0.3 * u.arcsec
CLIP_MAS = 60.0
CLUSTER_MAS = 50.0
# Coarse per-visit bulk tie: large-radius crowding-robust search (cross-correlation
# peak), NOT nearest-neighbour. A nearest-neighbour match with radius SEARCH cannot
# recover an offset larger than SEARCH: in a dense field every star has a chance
# neighbour within SEARCH, so the median collapses to ~0 and the visit is left
# UNcorrected (silently). This was the brick-1182 recurrence: the old coarse bridge
# was taken from each catalog's previously-applied RAOFFSET, which was itself ~0, so
# every rebuild re-confirmed ~0. COARSE_MAXSEP must exceed the largest expected raw
# guide-star pointing error (~2" for brick); QA_FAIL_MAS rejects a bad solve.
COARSE_MAXSEP = 5.0 * u.arcsec   # >= largest expected raw pointing error (1182 ~1.94"); headroom is cheap on clean i2d input
COARSE_BIN = 0.08          # arcsec, coarse-histogram bin (refined by the SEARCH fine step)
COARSE_MIN_PEAK_RATIO = 5.0  # i2d xcorr peak/background below this -> tie ambiguous, fail loud


def coarse_xcorr(sc, ref, maxsep=COARSE_MAXSEP, binarc=COARSE_BIN):
    """Robust bulk COORDINATE offset (arcsec Delta-alpha/Delta-delta, NO cosd) to ADD to
    ``sc`` to land on ``ref``, via the peak of the 2-D histogram of ALL pairs within
    ``maxsep``.  Crowding-proof and recovers offsets up to ``maxsep`` (unlike a
    nearest-neighbour match, which cannot see past its own radius) -- PROVIDED the input
    is a clean, high-SNR source list.  Returns (dra, ddec, npairs, peak, bg) or
    (None,)*5 if too few pairs.

    NB: must be fed CLEAN positions (drizzled-mosaic detections), NOT raw per-frame
    SIAF positions: per-detector SIAF residuals smear the per-frame pair-peak below the
    chance-pair background (measured brick 1182: per-frame peak/bg~1.5, i2d peak/bg~55).
    """
    ia, ib, sep, _ = search_around_sky(sc, ref, maxsep)
    if len(ia) < 50:
        return None, None, None, None, None
    dra = (ref[ib].ra - sc[ia].ra).to(u.arcsec).value
    ddec = (ref[ib].dec - sc[ia].dec).to(u.arcsec).value
    m = maxsep.to(u.arcsec).value
    bins = np.arange(-m, m + binarc, binarc)
    H, xe, ye = np.histogram2d(dra, ddec, bins=[bins, bins])
    i, j = np.unravel_index(H.argmax(), H.shape)
    bg = float(np.median(H[H > 0])) if (H > 0).any() else 0.0
    return ((xe[i] + xe[i + 1]) / 2.0, (ye[j] + ye[j + 1]) / 2.0,
            len(ia), float(H.max()), bg)


def detect_i2d_sources(i2d_path, thr=80.0, fwhm=2.5):
    """Bright high-SNR source SkyCoords from a drizzled mosaic (for the coarse tie)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    sci = fits.open(i2d_path)['SCI']
    w = WCS(sci.header)
    d = sci.data.astype('float32')
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=fwhm, threshold=thr * std)(d - med)
    return SkyCoord(w.pixel_to_world(t['xcentroid'], t['ycentroid']))


def coarse_from_i2d(filt, rc, ref):
    """Per-FILTER coarse bulk tie measured on the drizzled mosaic (clean) vs VIRAC2.
    This seeds every visit of the filter; the per-visit/per-exposure fine NN then
    resolves the <SEARCH residual.  Returns (dra, ddec) arcsec to ADD, or None."""
    sub = rc['filts'][filt][0]
    i2d = (f"{rc['basepath']}/{sub}/pipeline/"
           f"jw0{rc['proposal']}-o{rc['field']}_t001_nircam_clear-{filt}-merged_i2d.fits")
    if not os.path.exists(i2d):
        print(f"  [coarse] no i2d for {filt}: {i2d}")
        return None
    sc = detect_i2d_sources(i2d)
    dra, ddec, n, peak, bg = coarse_xcorr(sc, ref)
    if dra is None:
        print(f"  [coarse] {filt}: too few i2d pairs"); return None
    ratio = peak / bg if bg > 0 else np.inf
    if ratio < COARSE_MIN_PEAK_RATIO:
        print(f"  [coarse] {filt}: peak/bg {ratio:.1f} < {COARSE_MIN_PEAK_RATIO} -> ambiguous, refusing")
        return None
    # refine the histogram coarse (bin-limited ~half-bin) to mas precision with a fine NN
    # on the CLEAN i2d detections themselves (now within <SEARCH of VIRAC2, so the NN finds
    # real counterparts, not chance) -- removes the coarse-bin dependence of the final tie.
    shifted = SkyCoord((sc.ra.deg + dra / 3600.0) * u.deg, (sc.dec.deg + ddec / 3600.0) * u.deg)
    fine = coord_shift(shifted.ra.deg, shifted.dec.deg, ref)
    if fine is not None:
        dra += fine[0]; ddec += fine[1]
    print(f"  [coarse] {filt}: i2d tie ADD=({dra:+.4f},{ddec:+.4f})\" "
          f"npairs={n} peak/bg={peak:.0f}/{bg:.0f}={ratio:.1f} "
          f"i2dfine=({(fine[0]*1000 if fine else 0):+.1f},{(fine[1]*1000 if fine else 0):+.1f})mas "
          f"({len(sc)} i2d srcs)", flush=True)
    return dra, ddec

# region -> proposal/field/basepath + {filt: (subdir, obs-epoch, mtag)}
REGION = {
    '1182': dict(proposal='1182', field='004', basepath='/orange/adamginsburg/jwst/brick',
                 filts={'f115w': ('F115W', 2022.703, '_m3'), 'f200w': ('F200W', 2022.703, '_m3'),
                        'f356w': ('F356W', 2022.703, '_m2'), 'f444w': ('F444W', 2022.703, '_m2')}),
    'cloudc': dict(proposal='2221', field='002', basepath='/orange/adamginsburg/jwst/cloudc',
                   filts={'f182m': ('F182M', 2023.30, '_m3'), 'f187n': ('F187N', 2023.30, '_m3'),
                          'f212n': ('F212N', 2023.30, '_m3'), 'f405n': ('F405N', 2023.30, '_m3'),
                          'f410m': ('F410M', 2023.30, '_m3'), 'f466n': ('F466N', 2023.30, '_m3')}),
}
SW = {'f115w', 'f200w', 'f182m', 'f187n', 'f212n'}
SW_DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
LW_DETS = ['nrcalong', 'nrcblong']


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, float)), np.nan), float)


def virac2(epoch, cachepath):
    v = Table.read(cachepath)
    ra = farr(v['RAJ2000']); dec = farr(v['DEJ2000'])
    pr = np.where(np.isfinite(farr(v['pmRA'])), farr(v['pmRA']), 0.)
    pd = np.where(np.isfinite(farr(v['pmDE'])), farr(v['pmDE']), 0.)
    dt = epoch - V2EP
    return SkyCoord((ra + (pr * dt / 3.6e6) / np.cos(np.radians(dec))) * u.deg,
                    (dec + pd * dt / 3.6e6) * u.deg)


def coord_shift(ra, dec, ref):
    """clipped-median Δα/Δδ COORDINATE offset (arcsec, NO cosδ) to ADD to (ra,dec) to land
    on ref -- the convention adjust_wcs(delta_ra/delta_dec) consumes.  Clip is on-sky."""
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    idx, sep, _ = sc.match_to_catalog_sky(ref)
    m = sep < SEARCH
    if m.sum() < 15:
        return None
    a = sc[m]; b = ref[idx[m]]
    dra_c = (b.ra - a.ra).to(u.arcsec).value
    ddec_c = (b.dec - a.dec).to(u.arcsec).value
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
    sel = g_n[:ng] >= 2
    return SkyCoord(g_ra[:ng][sel] * u.deg, g_dec[:ng][sel] * u.deg)


def load_siaf(f):
    """Recover SIAF positions by undoing the recorded RAOFFSET/DEOFFSET. -> (ra,dec,ra0,de0)."""
    t = Table.read(f)
    sc = SkyCoord(t['skycoord_centroid'])
    ra0 = float(t.meta.get('RAOFFSET', 0.0)); de0 = float(t.meta.get('DEOFFSET', 0.0))  # arcsec
    fl = farr(t['flux_fit']); q = farr(t['qfit']) if 'qfit' in t.colnames else np.zeros(len(t))
    good = np.isfinite(fl) & (fl > 0) & (q < 0.4) & np.isfinite(sc.ra.deg)
    return sc.ra.deg[good] - ra0 / 3600.0, sc.dec.deg[good] - de0 / 3600.0, ra0, de0


def module_key(det):
    """fix_alignment (PipelineRerunNIRCAM-LONG.py:1208) matches a 'Module' cell against
    the detector name OR its digit-stripped root.  SW detectors nrca1..4 -> 'nrca',
    nrcb1..4 -> 'nrcb'; LW nrcalong/nrcblong keep their full names (strip('1234') is a
    no-op there).  Grouping by this key gives one tie per PHYSICAL module (A vs B)."""
    return det if det in LW_DETS else det[:4]


def _gather(filt, base, sub, mtag, dets):
    """Collect per-(visit,exp) and per-visit SIAF positions + legacy coarse for a det set."""
    from collections import defaultdict
    byve = defaultdict(lambda: [[], []]); byv = defaultdict(list); coarse = defaultdict(lambda: [[], []])
    for det in dets:
        for f in glob.glob(f'{base}/{sub}/{filt}_{det}_visit*_vgroup*_exp*{mtag}_daophot_basic.fits'):
            b = os.path.basename(f)
            vis = b.split('_visit')[1][:3]; exp = int(re.search(r'_exp(\d+)', b).group(1))
            ra, dec, ra0, de0 = load_siaf(f)
            byve[(vis, exp)][0].append(ra); byve[(vis, exp)][1].append(dec)
            byv[vis].append((ra, dec)); coarse[vis][0].append(ra0); coarse[vis][1].append(de0)
    return byve, byv, coarse


def _solve(byve, byv, coarse, c_ra, c_dec, ref, prop, field, filt, modlabel=None):
    """Per-visit bulk tie (consensus vs VIRAC2, seeded by the merged i2d coarse) + per-exposure
    relative shift vs that consensus.  modlabel=None -> module-LOCKED (one shift/exposure over all
    detectors, no Module column).  modlabel set -> that module's own tie, written with a Module
    cell so fix_alignment applies it per-module (removes a real inter-module A/B offset)."""
    tag = f"[{modlabel}] " if modlabel else ""
    rows = []
    for vis in sorted(byv):
        # legacy coarse (median of previously-applied RAOFFSET) -- diagnostic only.
        c_ra_legacy = float(np.median(coarse[vis][0])); c_dec_legacy = float(np.median(coarse[vis][1]))
        consensus = build_consensus(byv[vis])
        cc_ra = consensus.ra.deg + c_ra / 3600.0; cc_dec = consensus.dec.deg + c_dec / 3600.0
        res = coord_shift(cc_ra, cc_dec, ref)
        if res is None:
            # coarse alone (no per-visit fine refinement available)
            res = (0.0, 0.0, 0.0, 0.0, 0)
            print(f"  visit{vis}: fine tie weak; using i2d coarse alone")
        bulk_ra = c_ra + res[0]; bulk_dec = c_dec + res[1]
        print(f"  {tag}visit{vis}: i2d_coarse({c_ra:+.4f},{c_dec:+.4f})\" [legacy {c_ra_legacy:+.4f},"
              f"{c_dec_legacy:+.4f}] + fine({res[0]*1000:+.1f},{res[1]*1000:+.1f})mas "
              f"=> BULK ({bulk_ra:.4f},{bulk_dec:.4f})\" SEM {res[2]:.2f}/{res[3]:.2f}mas "
              f"n={res[4]}; consensus={len(consensus)}", flush=True)
        for exp in sorted(e for (v, e) in byve if v == vis):
            ra = np.concatenate(byve[(vis, exp)][0]); dec = np.concatenate(byve[(vis, exp)][1])
            rel = coord_shift(ra, dec, consensus)
            if rel is None:
                print(f"    {tag}exp{exp}: relative failed"); continue
            tot_ra = bulk_ra + rel[0]; tot_dec = bulk_dec + rel[1]
            row = dict(Visit=f'jw0{prop}{field}{vis}', Exposure=int(exp), Filter=filt.upper(),
                       dra=tot_ra, ddec=tot_dec, nmatch=rel[4],
                       rel_ra_mas=rel[0] * 1000, rel_dec_mas=rel[1] * 1000)
            if modlabel is not None:
                row['Module'] = modlabel
            rows.append(row)
            print(f"    {tag}exp{exp:>2}: rel({rel[0]*1000:+.2f},{rel[1]*1000:+.2f})mas n={rel[4]}"
                  f"  -> total({tot_ra:.4f},{tot_dec:.4f})\"", flush=True)
    return rows


def lock_filter(filt, rc, per_module=False):
    sub, ep, mtag = rc['filts'][filt]
    prop, field, base = rc['proposal'], rc['field'], rc['basepath']
    cache = f'{base}/astrometry_diag/refcache/virac2.fits'
    print(f"=== per-exposure relock {filt} ({prop}/{field}, epoch {ep}) "
          f"[{'PER-MODULE' if per_module else 'module-locked'}] ===", flush=True)
    ref = virac2(ep, cache)
    dets = SW_DETS if filt in SW else LW_DETS
    # PER-FILTER coarse bulk tie, measured ONCE on the clean drizzled mosaic vs VIRAC2.
    # Seeds every visit; the per-visit/per-exposure fine NN below resolves the residual
    # (including any per-module <SEARCH difference).  FAIL LOUD if the mosaic tie is dirty.
    i2d_coarse = coarse_from_i2d(filt, rc, ref)
    if i2d_coarse is None:
        raise SystemExit(f"[FAIL] {filt}: could not measure a clean i2d coarse tie; "
                         f"refusing to write a lock table (would re-perpetuate ~0).")
    c_ra, c_dec = i2d_coarse
    if not per_module:
        byve, byv, coarse = _gather(filt, base, sub, mtag, dets)
        return _solve(byve, byv, coarse, c_ra, c_dec, ref, prop, field, filt, modlabel=None)
    # PER-MODULE: solve a separate tie for each physical module (A=nrca*, B=nrcb*/LW
    # nrcalong/nrcblong).  A single module-locked shift cannot remove a real A/B offset
    # (the ~20 mas Dec-28.71 seam / NRCB distortion residual); two independent ties do.
    groups = {}
    for det in dets:
        groups.setdefault(module_key(det), []).append(det)
    rows = []
    for modlabel, gdets in sorted(groups.items()):
        print(f"  --- module '{modlabel}': {gdets} ---", flush=True)
        byve, byv, coarse = _gather(filt, base, sub, mtag, gdets)
        rows.extend(_solve(byve, byv, coarse, c_ra, c_dec, ref, prop, field, filt, modlabel=modlabel))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default='1182', choices=list(REGION))
    ap.add_argument('--per-module', action='store_true',
                    help='solve a separate per-module (A/B) tie and emit a Module column '
                         '(removes a real inter-module offset; fix_alignment narrows by Module)')
    ap.add_argument('--out', default=None, help='override output path (for validation before '
                    'overwriting the production table)')
    ap.add_argument('filts', nargs='*', help='filters (default: all of region)')
    args = ap.parse_args()
    rc = REGION[args.region]
    filts = args.filts or list(rc['filts'])
    rows = []
    for f in filts:
        rows.extend(lock_filter(f, rc, per_module=args.per_module))
    if not rows:
        print("no rows produced"); sys.exit(1)
    t = Table(rows)
    t['dra (arcsec)'] = t['dra']; t['ddec (arcsec)'] = t['ddec']
    path = args.out or f"{rc['basepath']}/offsets/Offsets_JWST_Brick{rc['proposal']}_VIRAC2locked.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # FIELD-SAFE merge: replace only rows for the SAME (Filter, proposal+field Visit prefix);
    # preserve every other filter AND every other field that shares this per-proposal table.
    new_visit_prefixes = set(str(v)[:11] for v in t['Visit'])   # jw0<prop><field>
    new_filts = set(str(x) for x in t['Filter'])
    if os.path.exists(path):
        old = Table.read(path)
        keepmask = np.array([not (str(r['Filter']) in new_filts and str(r['Visit'])[:11] in new_visit_prefixes)
                             for r in old])
        if keepmask.any():
            old = old[keepmask]
            for c in t.colnames:
                if c not in old.colnames:
                    old[c] = np.nan
            for c in old.colnames:
                if c not in t.colnames:
                    t[c] = np.nan
            t = vstack([old, t])
    t.write(path, overwrite=True)
    print(f"\nwrote {path}: {len(t)} rows (replaced {sorted(new_filts)} for prefixes {sorted(new_visit_prefixes)})", flush=True)
