"""
Lock the NIRCam modules within each exposure to a single global shift, then rebuild the per-filter
combined catalog. Fixes the 1182 'quiltwork' where each detector got an independent (over-fit)
tweakreg shift, breaking the SIAF-locked relative detector geometry.

Model: within one exposure the 8 detectors are rigidly fixed by SIAF to <0.01 px, so there is ONE
pointing shift per exposure, not 8. We measure each detector's residual vs the absolute reference
(VIRAC2 PM-propagated to the obs epoch), and the exposure-mean residual over all detectors combined;
we then remove each detector's DEVIATION from the exposure mean:

    corrected = skycoord_centroid - (detector_resid[d] - exposure_resid)

This locks all detectors of an exposure to the exposure-mean tie (removes per-detector quiltwork)
while leaving the exposure-level shift for the global VIRAC2 tie (build_f200w_f182m_reference). The
corrected per-exposure detections are then grouped across frames (incremental KDTree on a tangent
plane) into a combined catalog with mean position + cross-frame scatter.

Output: catalogs/<filt>_merged_indivexp_LOCKED_dao_basic.fits  (RA, DEC, skycoord, flux, std_ra_mas,
std_dec_mas, nframes).

Usage: python lock_exposures.py f200w [f115w f356w f444w f182m ...]
"""
import sys, glob, os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import mad_std
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

CATDIR = '/orange/adamginsburg/jwst/brick/catalogs'
F200WDIR = '/orange/adamginsburg/jwst/brick'
VIRAC2CACHE = '/orange/adamginsburg/jwst/brick/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'
V2EP = 2014.0
MATCH_RADIUS_MAS = 50.0   # cross-frame grouping radius

# per-filter: (subdir, vgroup, obs-epoch, manual-pass tag)
FILT_CFG = {
    'f115w': ('F115W', '02101', 2022.703, '_m3'),
    'f200w': ('F200W', '04101', 2022.703, '_m3'),
    'f356w': ('F356W', '02101', 2022.703, '_m2'),
    'f444w': ('F444W', '04101', 2022.703, '_m2'),
    'f182m': ('F182M', '07101', 2022.655, '_m3'),
    'f187n': ('F187N', '03101', 2022.655, '_m3'),
    'f212n': ('F212N', '05101', 2022.655, '_m3'),
    'f405n': ('F405N', '03101', 2022.655, '_m3'),
    'f410m': ('F410M', '07101', 2022.655, '_m3'),
    'f466n': ('F466N', '05101', 2022.655, '_m3'),
}
SW = {'f115w', 'f200w', 'f182m', 'f187n', 'f212n'}


def virac2(epoch):
    v = Table.read(VIRAC2CACHE)
    ra = np.asarray(v['RAJ2000'], float); dec = np.asarray(v['DEJ2000'], float)
    pr = np.where(np.isfinite(np.asarray(v['pmRA'], float)), np.asarray(v['pmRA'], float), 0.)
    pd = np.where(np.isfinite(np.asarray(v['pmDE'], float)), np.asarray(v['pmDE'], float), 0.)
    dt = epoch - V2EP
    return SkyCoord((ra + (pr * dt / 3.6e6) / np.cos(np.radians(dec))) * u.deg,
                    (dec + pd * dt / 3.6e6) * u.deg)


def robust_offset(sc, ref, search=0.3 * u.arcsec, clip=60.0):
    """median (dRA,dDec) mas of sc relative to ref (ref - sc), clipped."""
    idx, sep, _ = sc.match_to_catalog_sky(ref)
    m = sep < search
    if m.sum() < 25:
        return None, None, int(m.sum())
    a = sc[m]; b = ref[idx[m]]
    dra = ((b.ra - a.ra) * np.cos(a.dec.radian)).to(u.mas).value
    ddec = (b.dec - a.dec).to(u.mas).value
    md, mdd = np.median(dra), np.median(ddec)
    cl = np.hypot(dra - md, ddec - mdd) < clip
    return float(np.median(dra[cl])), float(np.median(ddec[cl])), int(cl.sum())


def load_exposure_detectors(filt):
    """yield (visit, exp, {det: (ra,dec,flux)}) for every exposure of the filter."""
    sub, vg, ep, mtag = FILT_CFG[filt]
    dets = ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'] if filt in SW else ['nrcalong', 'nrcblong']
    # discover exposures present
    from collections import defaultdict
    byexp = defaultdict(dict)
    for det in dets:
        for f in glob.glob(f'{F200WDIR}/{sub}/{filt}_{det}_visit*_vgroup{vg}_exp*{mtag}_daophot_basic.fits'):
            base = os.path.basename(f)
            vis = base.split('_visit')[1][:3]
            exp = base.split('_exp')[1][:5]
            byexp[(vis, exp)][det] = f
    return byexp, ep


def lock_filter(filt):
    print(f"=== locking {filt} ===", flush=True)
    byexp, ep = load_exposure_detectors(filt)
    ref = virac2(ep)
    cosd0 = np.cos(np.radians(-28.70))
    ra0 = 266.53
    # accumulate corrected detections
    all_ra = []; all_dec = []; all_flux = []
    n_exp = 0
    for (vis, exp), detfiles in sorted(byexp.items()):
        # load all detectors, measure per-detector residual + exposure residual
        det_sc = {}; det_flux = {}
        for det, f in detfiles.items():
            t = Table.read(f)
            sc = SkyCoord(t['skycoord_centroid'])
            fl = np.asarray(t['flux_fit'], float)
            q = np.asarray(t['qfit'], float) if 'qfit' in t.colnames else np.zeros(len(t))
            good = np.isfinite(fl) & (fl > 0) & (q < 0.4)
            det_sc[det] = sc[good]; det_flux[det] = fl[good]
        # exposure residual: all detectors combined
        comb = SkyCoord(np.concatenate([s.ra.deg for s in det_sc.values()]) * u.deg,
                        np.concatenate([s.dec.deg for s in det_sc.values()]) * u.deg)
        exp_r, exp_d, ne = robust_offset(comb, ref)
        if exp_r is None:
            continue
        n_exp += 1
        for det in det_sc:
            d_r, d_d, nd = robust_offset(det_sc[det], ref)
            if d_r is None:
                d_r, d_d = exp_r, exp_d   # too few -> assume exposure mean (locked)
            # remove per-detector DEVIATION from the exposure mean (lock); keep exposure shift
            corr_ra = d_r - exp_r   # mas, deviation to subtract from sky.ra*cosd
            corr_dec = d_d - exp_d
            sc = det_sc[det]
            new_ra = sc.ra + (corr_ra * u.mas) / np.cos(sc.dec.radian)
            new_dec = sc.dec + corr_dec * u.mas
            all_ra.append(new_ra.deg); all_dec.append(new_dec.deg); all_flux.append(det_flux[det])
    rad = MATCH_RADIUS_MAS / 1000.0 / 3600.0   # deg
    # incremental combine across frames (Welford running mean + variance), brightest frames first.
    # accumulators (grow as new groups are appended)
    cap = 2_000_000
    g_ra = np.empty(cap); g_dec = np.empty(cap); g_flux = np.empty(cap)
    g_n = np.zeros(cap, int); g_m2ra = np.zeros(cap); g_m2dec = np.zeros(cap)
    ng = 0
    cosd = cosd0
    for fi, (fra, fdec, fflux) in enumerate(zip(all_ra, all_dec, all_flux)):
        if ng == 0:
            n = len(fra)
            g_ra[:n] = fra; g_dec[:n] = fdec; g_flux[:n] = fflux; g_n[:n] = 1; ng = n
            continue
        base = SkyCoord(g_ra[:ng] * u.deg, g_dec[:ng] * u.deg)
        cur = SkyCoord(fra * u.deg, fdec * u.deg)
        idx, sep, _ = cur.match_to_catalog_sky(base)
        sepd = sep.deg
        matched = sepd < rad
        gi = idx[matched]
        # Welford update for matched
        dra = (fra[matched] - g_ra[gi]) * cosd
        ddec = (fdec[matched] - g_dec[gi])
        g_n[gi] += 1
        g_ra[gi] += dra / g_n[gi] / cosd
        g_dec[gi] += ddec / g_n[gi]
        g_m2ra[gi] += dra * ((fra[matched] - g_ra[gi]) * cosd)
        g_m2dec[gi] += ddec * (fdec[matched] - g_dec[gi])
        # append unmatched
        um = ~matched
        k = um.sum()
        if ng + k > cap:
            raise RuntimeError("group capacity exceeded; raise cap")
        g_ra[ng:ng + k] = fra[um]; g_dec[ng:ng + k] = fdec[um]; g_flux[ng:ng + k] = fflux[um]; g_n[ng:ng + k] = 1
        ng += k
        if fi % 24 == 0:
            print(f"    frame {fi}/{len(all_ra)}  groups={ng}", flush=True)
    g_ra = g_ra[:ng]; g_dec = g_dec[:ng]; g_flux = g_flux[:ng]; g_n = g_n[:ng]
    with np.errstate(invalid='ignore'):
        g_sra = np.where(g_n > 1, np.sqrt(g_m2ra[:ng] / np.maximum(g_n - 1, 1)) * 3.6e6, np.nan)
        g_sdec = np.where(g_n > 1, np.sqrt(g_m2dec[:ng] / np.maximum(g_n - 1, 1)) * 3.6e6, np.nan)
    out = Table()
    out['RA'] = g_ra; out['DEC'] = g_dec
    out['skycoord'] = SkyCoord(g_ra * u.deg, g_dec * u.deg)
    out['flux'] = g_flux; out['std_ra_mas'] = g_sra; out['std_dec_mas'] = g_sdec; out['nframes'] = g_n
    # compatibility columns so the existing seam/reference/fluxmatch loaders work unchanged
    out['nmatch'] = g_n
    out['std_ra'] = g_sra / 3.6e6           # deg (loaders multiply by 3600e3 -> mas)
    out['std_dec'] = g_sdec / 3.6e6
    out['qfit'] = np.zeros(len(out))
    out['is_saturated'] = np.zeros(len(out), bool)
    out.meta['CONTENT'] = f'{filt} module-locked (one shift/exposure) combined daophot-basic catalog'
    out.meta['LOCKEDBY'] = 'lock_exposures.py: remove per-detector deviation from exposure-mean VIRAC2 residual'
    out.meta['EPOCH'] = ep
    path = f'{CATDIR}/{filt}_merged_indivexp_LOCKED_dao_basic.fits'
    out.write(path, overwrite=True)
    mn = g_n >= 2
    print(f"  wrote {path}: {len(out)} groups ({mn.sum()} with nframes>=2); "
          f"median cross-frame scatter ({np.nanmedian(g_sra[mn]):.2f},{np.nanmedian(g_sdec[mn]):.2f}) mas")
    return path


if __name__ == '__main__':
    filts = sys.argv[1:] or ['f200w']
    for f in filts:
        lock_filter(f)
