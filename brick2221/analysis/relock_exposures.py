"""
EXACT per-exposure module-lock for the Brick (root-cause fix for per-detector tweakreg).

Per-detector tweakreg (fix_alignment applying a separate offset per detector) broke the SIAF lock
-> 'quiltwork'. The exact per-detector shift that was applied is recorded in each per-exposure
daophot catalog's meta (RAOFFSET/DEOFFSET, arcsec) = the crf fix_alignment shift. We:

  1. UNDO it exactly to recover the SIAF (assign_wcs) positions:  siac = skycoord_centroid - metaoffset
  2. per EXPOSURE, combine all detectors' SIAF positions, match VIRAC2 (PM-propagated to obs epoch),
     and solve ONE rigid shift (clipped median over thousands of stars across all 8 detectors)
  3. corrected = SIAF + one_shift   (identical shift for every detector -> SIAF lock restored;
     per-detector quiltwork removed to SIAF precision, NOT VIRAC2-per-detector-noise-limited)

Outputs (per filter):
  catalogs/<filt>_merged_indivexp_LOCKED_dao_basic.fits   (combined, compat columns)
  offsets/Offsets_JWST_Brick<prop>_VIRAC2locked.csv        (per visit,exposure,filter shift for
                                                            fix_alignment / re-reduction)

All filters tie to the SAME reference (VIRAC2 PM-propagated), so 115/182/200 land on one frame.
"""
import sys, glob, os
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.stats import mad_std
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/orange/adamginsburg/repos/jwst-gc-pipeline')
# dra_coordinate = ra1-ra2 (what adjust_wcs / the offsets table consume); reprojection
# recomputes RA/Dec from the stable detector x/y through the LIVE crf WCS (generation lock).
from jwst_gc_pipeline.astrometry_utils import _resolve_existing_path  # noqa: E402

CATDIR = '/orange/adamginsburg/jwst/brick/catalogs'
OFFDIR = '/orange/adamginsburg/jwst/brick/offsets'
BASE = '/orange/adamginsburg/jwst/brick'
VIRAC2CACHE = f'{BASE}/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'
V2EP = 2014.0
MATCH_RADIUS_MAS = 50.0

FILT_CFG = {  # filt: (subdir, vgroup, obs-epoch, mtag, proposal)
    'f115w': ('F115W', '02101', 2022.703, '_m3', '1182'),
    'f200w': ('F200W', '04101', 2022.703, '_m3', '1182'),
    'f356w': ('F356W', '02101', 2022.703, '_m2', '1182'),
    'f444w': ('F444W', '04101', 2022.703, '_m2', '1182'),
    'f182m': ('F182M', '07101', 2022.655, '_m3', '2221'),
    'f187n': ('F187N', '03101', 2022.655, '_m3', '2221'),
    'f212n': ('F212N', '05101', 2022.655, '_m3', '2221'),
    'f405n': ('F405N', '03101', 2022.655, '_m3', '2221'),
    'f410m': ('F410M', '07101', 2022.655, '_m3', '2221'),
    'f466n': ('F466N', '05101', 2022.655, '_m3', '2221'),
}
SW = {'f115w', 'f200w', 'f182m', 'f187n', 'f212n'}


def virac2(epoch):
    v = Table.read(VIRAC2CACHE)
    ra = np.asarray(v['RAJ2000'], float); dec = np.asarray(v['DEJ2000'], float)
    pr = np.where(np.isfinite(np.asarray(v['pmRA'], float)), np.asarray(v['pmRA'], float), 0.)
    pd = np.where(np.isfinite(np.asarray(v['pmDE'], float)), np.asarray(v['pmDE'], float), 0.)
    dt = epoch - V2EP
    return SkyCoord((ra + (pr * dt / 3.6e6) / np.cos(np.radians(dec))) * u.deg, (dec + pd * dt / 3.6e6) * u.deg)


def robust_shift(sc, ref, search=0.3 * u.arcsec, clip=60.,
                 coarse_maxsep=25. * u.arcsec, coarse_bin=0.08, min_peak_ratio=5.0):
    """one rigid (dRA_coordinate, dDec) mas to add to sc to land on ref (ref-sc), clipped median.

    Returns the COORDINATE Delta-alpha (ra_ref - ra_sc, NO cos(dec)) -- the value
    fix_alignment feeds to adjust_wcs(delta_ra=...), whose on-sky effect is *cos(dec).
    (Previously returned the on-sky Delta-alpha*cos(dec); fix_alignment then applied
    it as a coordinate rotation -> a ~cos(dec) under-correction, ~19-25 mas at the GC,
    that helped put every Brick band ~69 mas off VIRAC2.)

    FIRST bridge any large per-visit guide-star offset with a crowding-proof
    offset-HISTOGRAM stack (all pairs within ``coarse_maxsep``, take the 2-D peak).
    A plain nearest-neighbour median within ``search`` CANNOT see past its own
    radius: brick-1182 visit1 is ~17.5" off, so a 0.3" NN silently returns
    too-few/~0 and the whole visit is dropped or mis-locked (the 2026-07 visit001
    corruption).  Histogram stacking is crowding-immune and recovers the true
    offset up to ``coarse_maxsep`` before the fine NN refines the residual."""
    from astropy.coordinates import search_around_sky
    cra = cdec = 0.0   # coarse coordinate offset (arcsec, NO cosd) to add to sc
    ia, ib, _, _ = search_around_sky(sc, ref, coarse_maxsep)
    if len(ia) >= 50:
        dra_a = (ref[ib].ra - sc[ia].ra).to(u.arcsec).value
        ddec_a = (ref[ib].dec - sc[ia].dec).to(u.arcsec).value
        mm = coarse_maxsep.to(u.arcsec).value
        bins = np.arange(-mm, mm + coarse_bin, coarse_bin)
        H, xe, ye = np.histogram2d(dra_a, ddec_a, bins=[bins, bins])
        i, j = np.unravel_index(H.argmax(), H.shape)
        bg = float(np.median(H[H > 0])) if (H > 0).any() else 0.0
        if bg > 0 and H.max() / bg >= min_peak_ratio:
            cra = (xe[i] + xe[i + 1]) / 2.0; cdec = (ye[j] + ye[j + 1]) / 2.0
    cosd = np.cos(np.radians(-28.70))
    sc2 = SkyCoord((sc.ra.deg + cra / 3600.0) * u.deg, (sc.dec.deg + cdec / 3600.0) * u.deg)
    idx, sep, _ = sc2.match_to_catalog_sky(ref)
    m = sep < search
    if m.sum() < 30:
        return None, None, int(m.sum())
    a = sc2[m]; b = ref[idx[m]]
    # COORDINATE Delta-alpha = dra_coordinate = ra_ref - ra_sc (NO cos(dec)).
    dra_coord = (b.ra - a.ra).to(u.mas).value
    ddec = (b.dec - a.dec).to(u.mas).value
    cosd_star = np.cos(a.dec.radian)
    md, mdd = np.median(dra_coord), np.median(ddec)
    cl = np.hypot((dra_coord - md) * cosd_star, ddec - mdd) < clip   # clip on ON-SKY distance
    # total COORDINATE shift = coarse bridge (already coordinate, NO cosd) + fine coordinate
    # residual.  cra came from a histogram of (ref-sc) with no cos(dec) -> already coordinate.
    return (float(cra * 1000.0 + np.median(dra_coord[cl])),
            float(cdec * 1000.0 + np.median(ddec[cl])), int(cl.sum()))


def lock_filter(filt):
    sub, vg, ep, mtag, prop = FILT_CFG[filt]
    print(f"=== relock {filt} ({prop}, epoch {ep}) ===", flush=True)
    ref = virac2(ep)
    dets = (['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
            if filt in SW else ['nrcalong', 'nrcblong'])
    from collections import defaultdict
    # group per VISIT (not per exposure): the guide-star pointing error is per-visit (1182 visit1
    # ~-17.5", visit2 ~+1.95"), stable within a visit. SIAF/assign_wcs already carries the correct
    # per-exposure dithers + per-detector geometry, so one shift per visit -- jointly solved over
    # ALL exposures and detectors of the visit -- is robust (~1 mas) and keeps cross-filter
    # consistency. Per-exposure solving injected per-exposure VIRAC2 noise that broke cross-program.
    byvisit = defaultdict(list)
    for det in dets:
        for f in glob.glob(f'{BASE}/{sub}/{filt}_{det}_visit*_vgroup{vg}_exp*{mtag}_daophot_basic.fits'):
            base = os.path.basename(f)
            vis = base.split('_visit')[1][:3]
            byvisit[vis].append((det, f))

    def load_siac(f):
        # GENERATION LOCK: recompute RA/Dec from the STABLE detector x_fit/y_fit through
        # the LIVE crf WCS, NOT the catalog's cached skycoord_centroid.  The cached RA/Dec
        # encode the crf WCS at catalog-build time; a re-drizzle with a different
        # assign_wcs/distortion generation moves the frame (measured up to ~48 mas between
        # Brick runs) while x/y are generation-invariant.  Solving on the cached (stale)
        # RA/Dec is what left the tie a generation behind the crf it corrects.
        t = Table.read(f)
        crf = _resolve_existing_path(t.meta['FILENAME'])
        with fits.open(crf) as hl:
            wcs = WCS(hl['SCI'].header)
            # undo the RAOFFSET currently baked into THIS crf (not the possibly-stale
            # catalog meta), so we recover the current-generation SIAF positions.
            ra0 = float(hl['SCI'].header.get('RAOFFSET', t.meta.get('RAOFFSET', 0.0)))
            de0 = float(hl['SCI'].header.get('DEOFFSET', t.meta.get('DEOFFSET', 0.0)))
        sky = SkyCoord(wcs.pixel_to_world(np.asarray(t['x_fit'], float),
                                          np.asarray(t['y_fit'], float)))
        if 'skycoord_centroid' in t.colnames:   # drift diagnostic vs the cached positions
            cached = SkyCoord(t['skycoord_centroid'])
            drift = float(np.nanmedian((sky.ra.deg - cached.ra.deg)
                                       * np.cos(np.radians(cached.dec.deg)) * 3.6e6))
            if abs(drift) > 15:
                print(f"    [genlock] {os.path.basename(f)}: cached skycoord {drift:+.0f} mas "
                      f"stale vs live crf WCS -> reprojected from x/y", flush=True)
        fl = np.asarray(t['flux_fit'], float)
        q = np.asarray(t['qfit'], float) if 'qfit' in t.colnames else np.zeros(len(t))
        good = np.isfinite(fl) & (fl > 0) & (q < 0.4) & np.isfinite(sky.ra.deg)
        # undo applied per-detector shift -> SIAF positions (raw RA/Dec arcsec)
        return sky.ra.deg[good] - ra0 / 3600.0, sky.dec.deg[good] - de0 / 3600.0, fl[good]

    all_ra = []; all_dec = []; all_flux = []   # one entry PER FRAME (detector,exposure) for combine
    offrows = []
    for vis, files in sorted(byvisit.items()):
        # Pass 1: ONE shift per VISIT from all frames' SIAF positions jointly vs VIRAC2
        frames = [load_siac(f) for (det, f) in files]
        vra = np.concatenate([fr[0] for fr in frames]); vdec = np.concatenate([fr[1] for fr in frames])
        sr, sd, n = robust_shift(SkyCoord(vra * u.deg, vdec * u.deg), ref)
        if sr is None:
            continue
        offrows.append(dict(Visit=f'jw01182004{vis}' if prop == '1182' else f'jw02221001{vis}',
                            Filter=filt.upper(), dra=sr / 1000.0, ddec=sd / 1000.0, nmatch=n))  # arcsec
        # Pass 2: apply the per-visit shift to each FRAME separately (keeps within-visit grouping).
        # sr/sd are COORDINATE mas (dra_coordinate) now, so add straight to RA/Dec coordinates --
        # NO /cos(dec) (that division belonged to the old on-sky convention).
        for (fra, fdec, ffl) in frames:
            all_ra.append(fra + (sr / 1000.0 / 3600.0))
            all_dec.append(fdec + (sd / 1000.0 / 3600.0))
            all_flux.append(ffl)
    print(f"  {len(offrows)} visits locked; {len(all_ra)} frames; {sum(len(a) for a in all_ra)} detections; combining...", flush=True)

    # incremental combine (Welford)
    cosd = np.cos(np.radians(-28.70)); rad = MATCH_RADIUS_MAS / 1000. / 3600.
    cap = 2_500_000
    g_ra = np.empty(cap); g_dec = np.empty(cap); g_flux = np.empty(cap)
    g_n = np.zeros(cap, int); g_m2r = np.zeros(cap); g_m2d = np.zeros(cap); ng = 0
    for fi, (fra, fdec, fflux) in enumerate(zip(all_ra, all_dec, all_flux)):
        if ng == 0:
            n = len(fra); g_ra[:n] = fra; g_dec[:n] = fdec; g_flux[:n] = fflux; g_n[:n] = 1; ng = n; continue
        base_sc = SkyCoord(g_ra[:ng] * u.deg, g_dec[:ng] * u.deg)
        idx, sep, _ = SkyCoord(fra * u.deg, fdec * u.deg).match_to_catalog_sky(base_sc)
        mt = sep.deg < rad; gi = idx[mt]
        dra = (fra[mt] - g_ra[gi]) * cosd; ddec = (fdec[mt] - g_dec[gi])
        g_n[gi] += 1; g_ra[gi] += dra / g_n[gi] / cosd; g_dec[gi] += ddec / g_n[gi]
        g_m2r[gi] += dra * ((fra[mt] - g_ra[gi]) * cosd); g_m2d[gi] += ddec * (fdec[mt] - g_dec[gi])
        um = ~mt; k = int(um.sum())
        g_ra[ng:ng+k] = fra[um]; g_dec[ng:ng+k] = fdec[um]; g_flux[ng:ng+k] = fflux[um]; g_n[ng:ng+k] = 1; ng += k
    g_ra = g_ra[:ng]; g_dec = g_dec[:ng]; g_flux = g_flux[:ng]; g_n = g_n[:ng]
    with np.errstate(invalid='ignore'):
        g_sr = np.where(g_n > 1, np.sqrt(g_m2r[:ng] / np.maximum(g_n - 1, 1)) * 3.6e6, np.nan)
        g_sd = np.where(g_n > 1, np.sqrt(g_m2d[:ng] / np.maximum(g_n - 1, 1)) * 3.6e6, np.nan)
    out = Table()
    out['RA'] = g_ra; out['DEC'] = g_dec; out['skycoord'] = SkyCoord(g_ra * u.deg, g_dec * u.deg)
    out['flux'] = g_flux; out['std_ra_mas'] = g_sr; out['std_dec_mas'] = g_sd; out['nframes'] = g_n
    out['nmatch'] = g_n; out['std_ra'] = g_sr / 3.6e6; out['std_dec'] = g_sd / 3.6e6
    out['qfit'] = np.zeros(len(out)); out['is_saturated'] = np.zeros(len(out), bool)
    out.meta['CONTENT'] = f'{filt} EXACT per-exposure module-locked (SIAF lock restored)'
    out.meta['METHOD'] = 'undo recorded per-detector RAOFFSET -> SIAF -> one VIRAC2-tied shift/exposure'
    out.meta['EPOCH'] = ep
    outpath = f'{CATDIR}/{filt}_merged_indivexp_LOCKED_dao_basic.fits'
    out.write(outpath, overwrite=True)
    mn = g_n >= 2
    print(f"  wrote {outpath}: {len(out)} groups; cross-frame scatter ({np.nanmedian(g_sr[mn]):.2f},{np.nanmedian(g_sd[mn]):.2f}) mas", flush=True)
    return offrows


if __name__ == '__main__':
    filts = sys.argv[1:] or ['f200w']
    allrows = {}
    for f in filts:
        rows = lock_filter(f)
        prop = FILT_CFG[f][4]
        allrows.setdefault(prop, []).extend(rows)
    # write/merge per-exposure offset tables per proposal
    os.makedirs(OFFDIR, exist_ok=True)
    for prop, rows in allrows.items():
        if not rows:
            continue
        t = Table(rows)
        t['dra (arcsec)'] = t['dra']; t['ddec (arcsec)'] = t['ddec']
        path = f'{OFFDIR}/Offsets_JWST_Brick{prop}_VIRAC2locked.csv'
        if os.path.exists(path):
            old = Table.read(path)
            keep = ~np.isin(old['Filter'], list(set(t['Filter'])))
            t = vstack([old[keep], t]) if keep.any() else t
        t.write(path, overwrite=True)
        print(f"wrote per-exposure offset table {path}: {len(t)} rows", flush=True)
