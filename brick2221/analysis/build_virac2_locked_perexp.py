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
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/orange/adamginsburg/repos/jwst-gc-pipeline')
# GENERATION LOCK: recompute RA/Dec from stable detector x/y through the live crf WCS.
from jwst_gc_pipeline.astrometry_utils import _resolve_existing_path  # noqa: E402

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
COARSE_MAXSEP = 5.0 * u.arcsec   # per-FILTER i2d seed radius; refined per-visit below
COARSE_BIN = 0.08          # arcsec, coarse-histogram bin (refined by the SEARCH fine step)
COARSE_MIN_PEAK_RATIO = 5.0  # i2d xcorr peak/background below this -> tie ambiguous, fail loud
# PER-VISIT coarse radius.  MUST exceed the largest raw per-visit guide-star pointing
# error, NOT the ~2" once thought: brick-1182 visit001 is ~22" off (-17.5"/+13.5") while
# visit002 is ~2".  The single mosaic-wide COARSE_MAXSEP seed captures only the dominant
# visit; the other visit needs its OWN large-radius histogram xcorr (below) or it silently
# inherits the dominant visit's shift (the 2026-07 brick-1182 visit001 corruption).
COARSE_MAXSEP_VISIT = 25.0 * u.arcsec


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
    ratio = (peak / bg) if (peak is not None and bg and bg > 0) else 0.0
    if dra is None or ratio < COARSE_MIN_PEAK_RATIO:
        # ALL-PAIRS histogram is density-biased: two catalogs tracing the same dense field
        # make a non-uniform wrong-pair background that can bury the true peak (memory
        # histogram-vs-samestar-offset-bias; measured cloudef obs005 F162M all-pairs pk/bg
        # 1.8 while the tie is clean at -80,-30mas).  FALL BACK to a nearest-neighbour coarse
        # (sparse VIRAC -> nearest bright i2d source, ONE pair per ref star -> no chance-pair
        # pileup), safe here because the i2d detections are clean/high-SNR.
        idx, sep, _ = ref.match_to_catalog_sky(sc)
        mw = sep < COARSE_MAXSEP
        if mw.sum() >= 50:
            d_ra = (sc[idx[mw]].ra - ref[mw].ra).to(u.arcsec).value
            d_de = (sc[idx[mw]].dec - ref[mw].dec).to(u.arcsec).value
            mm = COARSE_MAXSEP.to(u.arcsec).value
            bins = np.arange(-mm, mm + COARSE_BIN, COARSE_BIN)
            H, xe, ye = np.histogram2d(d_ra, d_de, bins=[bins, bins])
            i, j = np.unravel_index(H.argmax(), H.shape)
            nbg = float(np.median(H[H > 0])) if (H > 0).any() else 0.0
            nratio = (H.max() / nbg) if nbg > 0 else np.inf
            if nratio >= COARSE_MIN_PEAK_RATIO:
                dra = -(xe[i] + xe[i + 1]) / 2.0; ddec = -(ye[j] + ye[j + 1]) / 2.0
                peak, bg, ratio, n = float(H.max()), nbg, nratio, int(mw.sum())
                print(f"  [coarse] {filt}: all-pairs ambiguous -> NN coarse pk/bg={ratio:.1f}", flush=True)
            else:
                print(f"  [coarse] {filt}: all-pairs AND NN ambiguous (pk/bg {ratio:.1f}/{nratio:.1f}) -> refusing")
                return None
        else:
            print(f"  [coarse] {filt}: peak/bg {ratio:.1f} ambiguous, too few NN pairs -> refusing")
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
    # cloudef (2092): Cloud E (obs 002) + Cloud F (obs 005), separate pointings ->
    # separate region keys; combine their VIRAC2locked tables into one Offsets file after.
    # Per-exposure catalogs are unstaged (f210m_nrcX_visit..._exp..._daophot_basic) -> mtag=''.
    'cloudef2': dict(proposal='2092', field='002', basepath='/orange/adamginsburg/jwst/cloudef',
                     filts={'f162m': ('F162M', 2023.21, '_m3'), 'f210m': ('F210M', 2023.21, '_m3'),
                            'f360m': ('F360M', 2023.21, '_m3'), 'f480m': ('F480M', 2023.21, '_m3')}),
    'cloudef5': dict(proposal='2092', field='005', basepath='/orange/adamginsburg/jwst/cloudef',
                     filts={'f162m': ('F162M', 2023.21, '_m3'), 'f210m': ('F210M', 2023.21, '_m3'),
                            'f360m': ('F360M', 2023.21, '_m3'), 'f480m': ('F480M', 2023.21, '_m3')}),
}
# NIRCam SW (nrca1-4/nrcb1-4) vs LW (nrcalong/nrcblong) split at ~2.4um: F070W..F212N are
# SW, F250M+ are LW.  Membership by filter number so any GC field's bands classify right.
def _is_sw(filt):
    m = re.match(r'f(\d{3})', filt.lower())
    return bool(m) and int(m.group(1)) < 240
SW = {f for f in ('f115w', 'f150w', 'f162m', 'f182m', 'f187n', 'f200w', 'f210m', 'f212n')}
SW_DETS = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
LW_DETS = ['nrcalong', 'nrcblong']

# STAGE-2 JWST<->JWST cross-tie (2026-07-06).  A field tied to VIRAC2 lands with a
# ~15-30 mas residual (2MASS-tie floor + hardcoded per-module shifts in fix_alignment);
# two fields tied INDEPENDENTLY (jw01182 broadbands vs jw02221 narrows -- SAME epoch,
# same brick) therefore disagree ~15 mas, which is unacceptable for JWST<->JWST.  So
# after the VIRAC2 solve we ADD a fixed per-filter shift that lands each secondary-field
# filter on the dense 2221 MASTER (F212N) frame (~1-3 mas).  VIRAC2 stays the absolute
# tie of the master (Gaia too sparse to tie per-visit here).
#
# The shift is a HARDCODED CONSTANT (Δα no-cosδ, Δδ; arcsec), NOT auto-measured each
# build.  Rationale: the 1182<->2221 frame offset is a fixed physical quantity (the
# VIRAC2 tie is deterministic + stable), so a constant is durable and cannot silently
# regress -- whereas re-measuring the LIVE catalog residual each build self-cancels once
# a prior cross-tie has already been drizzled in (catalog already on-frame -> measures 0
# -> writes 0 -> regression).  Re-measure with `--remeasure-crosstie` (flux-vetted; it
# PRINTS suggested constants, does not write) whenever the master 2221 frame moves, then
# paste the new numbers here.  Values below measured flux-vetted vs the m7 F212N catalog
# 2026-07-06 (fluxcorr 0.65-1.00, pk/bg 400-1000, n~22k), validated by simulation to
# bring 1182<->F212N from 11-19 mas to 1-3 mas.  cloudc is single-proposal -> no cross-tie.
CROSSTIE = {
    '1182': dict(master_cat='/orange/adamginsburg/jwst/brick/catalogs/'
                            'f212n_merged_indivexp_merged*_m[0-9]*_dao_basic_vetted.fits',
                 master_name='2221 F212N',
                 shifts={  # per-filter (Δα no-cosδ, Δδ) arcsec to ADD
                     'f115w': (+0.01868, -0.00080),
                     'f200w': (+0.01852, -0.00070),
                     'f356w': (+0.02114, -0.00100),
                     'f444w': (+0.02084, -0.00090),
                 }),
}


def _region_key(rc):
    for rk, rv in REGION.items():
        if rv is rc:
            return rk
    return None


def crosstie_constant(filt, rc):
    """Hardcoded flux-vetted-derived (Δα no-cosδ, Δδ) arcsec to ADD to `filt`'s VIRAC2
    tie so it lands on the master frame.  (0,0) if the region/filter has no cross-tie."""
    cfg = CROSSTIE.get(_region_key(rc))
    if cfg is None:
        return 0.0, 0.0
    return cfg['shifts'].get(filt, (0.0, 0.0))
CROSSTIE_SEED_WIN = 0.5      # arcsec, candidate window for the pair histogram
CROSSTIE_RIDGE_MAS = 60.0    # near-peak radius (mas) used to LEARN the cross-band mag ridge
CROSSTIE_CORE_MAS = 50.0     # true-match core (mas) for the final clipped-median shift
CROSSTIE_MIN_N = 200         # refuse (warn, apply 0) below this many vetted core matches


def _load_cat_fluxpos(path_or_glob):
    """(SkyCoord, instrumental mag) from a merged per-band vetted catalog."""
    g = sorted(glob.glob(path_or_glob), key=os.path.getmtime)
    if not g:
        return None
    t = Table.read(g[-1])
    col = 'skycoord' if 'skycoord' in t.colnames else next(
        (c for c in t.colnames if c.startswith('skycoord')), None)
    if col is None:
        return None
    fx = farr(t['flux']) if 'flux' in t.colnames else np.full(len(t), np.nan)
    return SkyCoord(t[col]), -2.5 * np.log10(np.where(fx > 0, fx, np.nan)), os.path.basename(g[-1])


def _offhist_peak(dra_mas, dde_mas, win_mas=500.0, bin_mas=2.0):
    e = np.arange(-win_mas, win_mas + bin_mas, bin_mas)
    H, xe, ye = np.histogram2d(dra_mas, dde_mas, bins=[e, e])
    i, j = np.unravel_index(H.argmax(), H.shape)
    pk = H.max(); bg = np.median(H[H > 0]) if (H > 0).any() else 1.0
    return (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2, pk / max(bg, 1.0)


def crosstie_offset(filt, rc):
    """Flux-vetted coordinate shift (Δα no-cosδ, Δδ; arcsec) to ADD to this filter's
    VIRAC2-locked tie so it lands on the master (2221 F212N) frame.  Returns (0,0) with a
    loud warning if it cannot be measured cleanly (never silently skips)."""
    region = None
    for rk, rv in REGION.items():
        if rv is rc:
            region = rk; break
    cfg = CROSSTIE.get(region)
    if cfg is None:
        return 0.0, 0.0
    master = _load_cat_fluxpos(cfg['master_cat'])
    src = _load_cat_fluxpos(f"{rc['basepath']}/catalogs/"
                            f"{filt}_merged_indivexp_merged*_m[0-9]*_dao_basic_vetted.fits")
    if master is None or src is None:
        print(f"  [crosstie] {filt}: missing master/src catalog -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0
    (msc, mmag, mnm), (ssc, smag, snm) = master, src
    i2, i1, _, _ = msc.search_around_sky(ssc, CROSSTIE_SEED_WIN * u.arcsec)
    if len(i1) < CROSSTIE_MIN_N:
        print(f"  [crosstie] {filt}: only {len(i1)} candidate pairs -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0
    # on-sky separations (mas) for the seed peak + ridge learning
    dra_gc = (ssc[i2].ra - msc[i1].ra).to(u.arcsec).value * np.cos(msc[i1].dec.rad) * 1000.0
    dde = (ssc[i2].dec - msc[i1].dec).to(u.arcsec).value * 1000.0
    dm = smag[i2] - mmag[i1]
    ra0, de0, _ = _offhist_peak(dra_gc, dde)
    near = (np.hypot(dra_gc - ra0, dde - de0) < CROSSTIE_RIDGE_MAS) & np.isfinite(dm)
    if near.sum() < 20:
        print(f"  [crosstie] {filt}: too few near-peak for ridge -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0
    med = np.median(dm[near]); mad = 1.4826 * np.median(np.abs(dm[near] - med))
    tol = max(3 * mad, 0.5)
    vet = np.isfinite(dm) & (np.abs(dm - med) < tol)          # FLUX VET
    ra1, de1, pr = _offhist_peak(dra_gc[vet], dde[vet])
    core = vet & (np.hypot(dra_gc - ra1, dde - de1) < CROSSTIE_CORE_MAS)
    if core.sum() < CROSSTIE_MIN_N:
        print(f"  [crosstie] {filt}: only {core.sum()} vetted core matches -> APPLYING 0 (WARN)", flush=True)
        return 0.0, 0.0
    # coordinate offset (src - master), NO cosδ on RA (table convention); ADD its NEGATION
    dra_nc = (ssc[i2[core]].ra - msc[i1[core]].ra).to(u.arcsec).value
    dde_c = (ssc[i2[core]].dec - msc[i1[core]].dec).to(u.arcsec).value
    add_ra = -float(np.median(dra_nc)); add_de = -float(np.median(dde_c))
    fcorr = np.corrcoef(mmag[i1[core]], smag[i2[core]])[0, 1]
    print(f"  [crosstie] {filt} vs {cfg['master_name']}: residual "
          f"({np.median(dra_nc) * 1000:+.1f},{np.median(dde_c) * 1000:+.1f})mas -> ADD "
          f"({add_ra * 1000:+.1f},{add_de * 1000:+.1f})mas  n={core.sum()} pk/bg={pr:.0f} "
          f"ridgeΔm={med:+.2f} fluxcorr={fcorr:.2f} [{mnm} <- {snm}]", flush=True)
    return add_ra, add_de


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
    """Recover current-generation SIAF positions. -> (ra,dec,ra0,de0).

    GENERATION LOCK: RA/Dec are recomputed from the STABLE detector x_fit/y_fit
    through the LIVE crf WCS (meta['FILENAME']), not the catalog's cached
    skycoord_centroid (which encodes the WCS at build time and goes stale ~up to
    48 mas across re-drizzle generations).  Then undo the RAOFFSET currently baked
    into that crf to reach SIAF.
    """
    t = Table.read(f)
    if 'x_fit' in t.colnames and 'FILENAME' in t.meta:
        crf = _resolve_existing_path(t.meta['FILENAME'])
        with fits.open(crf) as hl:
            wcs = WCS(hl['SCI'].header)
            ra0 = float(hl['SCI'].header.get('RAOFFSET', t.meta.get('RAOFFSET', 0.0)))
            de0 = float(hl['SCI'].header.get('DEOFFSET', t.meta.get('DEOFFSET', 0.0)))
        sc = SkyCoord(wcs.pixel_to_world(farr(t['x_fit']), farr(t['y_fit'])))
        if 'skycoord_centroid' in t.colnames:
            cached = SkyCoord(t['skycoord_centroid'])
            drift = float(np.nanmedian((sc.ra.deg - cached.ra.deg)
                                       * np.cos(np.radians(cached.dec.deg)) * 3.6e6))
            if abs(drift) > 15:
                print(f"    [genlock] {os.path.basename(f)}: cached skycoord {drift:+.0f} mas "
                      f"stale vs live crf WCS -> reprojected from x/y", flush=True)
    else:   # legacy catalog without x/y or FILENAME: fall back to cached positions
        sc = SkyCoord(t['skycoord_centroid'])
        ra0 = float(t.meta.get('RAOFFSET', 0.0)); de0 = float(t.meta.get('DEOFFSET', 0.0))
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
        # PER-VISIT coarse residual on top of the shared per-filter i2d seed.  REQUIRED:
        # visits can carry very different raw guide-star pointing errors (brick-1182
        # visit001 ~22" vs visit002 ~2").  The single mosaic-wide i2d coarse captures only
        # the dominant visit, so the other visit is left mis-seeded and the <SEARCH fine NN
        # below cannot bridge it -> it silently inherits the dominant visit's shift (the
        # cause of the 2026-07 brick-1182 visit001 corruption: all visits got visit002's
        # +1.9" instead of visit001's true -17.5").  A per-visit large-radius histogram
        # xcorr (crowding-proof) recovers each visit's own bulk before the fine step.
        cv_ra, cv_dec = c_ra, c_dec
        vx = coarse_xcorr(SkyCoord(cc_ra * u.deg, cc_dec * u.deg), ref, maxsep=COARSE_MAXSEP_VISIT)
        if vx[0] is not None:
            vratio = vx[3] / vx[4] if vx[4] > 0 else np.inf
            # only apply a MEANINGFUL per-visit correction (> the fine-NN radius); small
            # residuals are left to the fine step to avoid double counting.
            if vratio >= COARSE_MIN_PEAK_RATIO and np.hypot(vx[0], vx[1]) > SEARCH.to(u.arcsec).value:
                cc_ra = cc_ra + vx[0] / 3600.0; cc_dec = cc_dec + vx[1] / 3600.0
                cv_ra = c_ra + vx[0]; cv_dec = c_dec + vx[1]
                print(f"  {tag}visit{vis}: PER-VISIT coarse ADD ({vx[0]:+.3f},{vx[1]:+.3f})\" "
                      f"peak/bg={vratio:.1f} npairs={vx[2]}  (visit differs from mosaic seed)",
                      flush=True)
        res = coord_shift(cc_ra, cc_dec, ref)
        if res is None:
            # coarse alone (no per-visit fine refinement available)
            res = (0.0, 0.0, 0.0, 0.0, 0)
            print(f"  visit{vis}: fine tie weak; using coarse alone")
        bulk_ra = cv_ra + res[0]; bulk_dec = cv_dec + res[1]
        print(f"  {tag}visit{vis}: i2d_coarse({c_ra:+.4f},{c_dec:+.4f})\" [legacy {c_ra_legacy:+.4f},"
              f"{c_dec_legacy:+.4f}] pervisit({cv_ra:+.4f},{cv_dec:+.4f}) + fine({res[0]*1000:+.1f},{res[1]*1000:+.1f})mas "
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
    dets = SW_DETS if (filt in SW or _is_sw(filt)) else LW_DETS
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
    ap.add_argument('--remeasure-crosstie', action='store_true',
                    help='flux-vetted RE-MEASURE of the JWST<->JWST cross-tie vs the 2221 master; '
                         'PRINTS suggested CROSSTIE constants and EXITS (writes nothing). Run this '
                         'when the master 2221 frame moves, then paste the numbers into CROSSTIE.')
    ap.add_argument('filts', nargs='*', help='filters (default: all of region)')
    args = ap.parse_args()
    rc = REGION[args.region]
    filts = args.filts or list(rc['filts'])

    if args.remeasure_crosstie:
        cfg = CROSSTIE.get(args.region)
        if cfg is None:
            print(f"region {args.region} has no cross-tie master; nothing to measure."); sys.exit(0)
        print(f"# flux-vetted cross-tie vs {cfg['master_name']} -- paste into CROSSTIE['{args.region}']['shifts']:")
        for f in filts:
            ra, de = crosstie_offset(f, rc)
            print(f"    '{f}': ({ra:+.5f}, {de:+.5f}),")
        sys.exit(0)

    rows = []
    for f in filts:
        frows = lock_filter(f, rc, per_module=args.per_module)
        # STAGE-2: JWST<->JWST cross-tie onto the master frame (hardcoded constant; 1182 only).
        ct_ra, ct_de = crosstie_constant(f, rc)
        if ct_ra or ct_de:
            for r in frows:
                r['dra'] += ct_ra; r['ddec'] += ct_de
            print(f"  [crosstie] {f}: applied CONSTANT ({ct_ra*1000:+.1f},{ct_de*1000:+.1f})mas "
                  f"to {len(frows)} rows", flush=True)
        rows.extend(frows)
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
