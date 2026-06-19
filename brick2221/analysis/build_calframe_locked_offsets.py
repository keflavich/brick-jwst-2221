"""
Build per-exposure MODULE-LOCKED alignment offsets from the CAL-frame (assign_wcs/SIAF) positions.

The catalog-level undo (skycoord - RAOFFSET) is NOT exact for the m3 manual-pipeline catalogs, so
instead we recompute the true SIAF sky positions directly: take the daophot detector pixel positions
(x_fit, y_fit) and evaluate them through the CAL file's GWCS (the assign_wcs solution, identical
distortion to crf but WITHOUT the fix_alignment shift). All detectors of an exposure share one SIAF
pointing, so per VISIT we combine all detectors+exposures, match VIRAC2 (PM-propagated to obs epoch),
and solve ONE rigid shift. That shift is the module-locked fix_alignment offset; applied to the SIAF
positions it gives a quiltwork-free, cross-program-consistent catalog (all filters -> one VIRAC2 frame).

Outputs:
  catalogs/<filt>_merged_indivexp_CALLOCKED_dao_basic.fits  (SIAF + per-visit VIRAC2 shift, combined)
  offsets/Offsets_JWST_Brick<prop>_VIRAC2locked.csv          (per visit,filter shift for fix_alignment)
"""
import sys, glob, os
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')
from jwst import datamodels

CATDIR = '/orange/adamginsburg/jwst/brick/catalogs'
OFFDIR = '/orange/adamginsburg/jwst/brick/offsets'
BASE = '/orange/adamginsburg/jwst/brick'
VIRAC2CACHE = f'{BASE}/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'
V2EP = 2014.0
MATCH_RADIUS_MAS = 50.0

FILT_CFG = {
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


def robust_shift(sc, ref, search=0.5 * u.arcsec, clip=80.):
    idx, sep, _ = sc.match_to_catalog_sky(ref)
    m = sep < search
    if m.sum() < 30:
        return None, None, int(m.sum())
    a = sc[m]; b = ref[idx[m]]
    dra = ((b.ra - a.ra) * np.cos(a.dec.radian)).to(u.mas).value
    ddec = (b.dec - a.dec).to(u.mas).value
    md, mdd = np.median(dra), np.median(ddec)
    cl = np.hypot(dra - md, ddec - mdd) < clip
    return float(np.median(dra[cl])), float(np.median(ddec[cl])), int(cl.sum())


def calwcs_cache():
    return {}


def siaf_positions(filt, det, vis, exp, vg, sub, _wcscache):
    """daophot x_fit,y_fit (m-pass) evaluated through the CAL GWCS -> SIAF sky."""
    capath = f'{BASE}/{sub}/pipeline/jw0{ "1182004" if FILT_CFG[filt][4]=="1182" else "2221001" }{vis}_{vg}_{exp}_{det}_cal.fits'
    if not os.path.exists(capath):
        return None
    if capath not in _wcscache:
        m = datamodels.open(capath); _wcscache[capath] = m.meta.wcs; m.close()
    wcs = _wcscache[capath]
    # find the matching daophot catalog (m-pass)
    cat = glob.glob(f'{BASE}/{sub}/{filt}_{det}_visit{vis}_vgroup{vg}_exp{exp}{FILT_CFG[filt][3]}_daophot_basic.fits')
    if not cat:
        return None
    t = Table.read(cat[0])
    x = np.asarray(t['x_fit'], float); y = np.asarray(t['y_fit'], float)
    fl = np.asarray(t['flux_fit'], float)
    q = np.asarray(t['qfit'], float) if 'qfit' in t.colnames else np.zeros(len(t))
    good = np.isfinite(x) & np.isfinite(fl) & (fl > 0) & (q < 0.4)
    ra, dec = wcs(x[good], y[good])
    fin = np.isfinite(ra) & np.isfinite(dec)
    return ra[fin], dec[fin], fl[good][fin]


def lock_filter(filt):
    sub, vg, ep, mtag, prop = FILT_CFG[filt]
    print(f"=== CAL-frame relock {filt} ({prop}) ===", flush=True)
    ref = virac2(ep)
    dets = (['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
            if filt in SW else ['nrcalong', 'nrcblong'])
    # discover (visit,exp,det) from the daophot catalogs present
    from collections import defaultdict
    byvisit = defaultdict(list)
    for det in dets:
        for f in glob.glob(f'{BASE}/{sub}/{filt}_{det}_visit*_vgroup{vg}_exp*{mtag}_daophot_basic.fits'):
            b = os.path.basename(f); vis = b.split('_visit')[1][:3]; exp = b.split('_exp')[1][:5]
            byvisit[vis].append((det, exp))

    wcache = calwcs_cache()
    all_ra = []; all_dec = []; all_flux = []; offrows = []
    cosd = np.cos(np.radians(-28.70))
    for vis, items in sorted(byvisit.items()):
        frames = []
        for det, exp in items:
            r = siaf_positions(filt, det, vis, exp, vg, sub, wcache)
            if r is not None and len(r[0]):
                frames.append(r)
        if not frames:
            continue
        vra = np.concatenate([f[0] for f in frames]); vdec = np.concatenate([f[1] for f in frames])
        sr, sd, n = robust_shift(SkyCoord(vra * u.deg, vdec * u.deg), ref)
        if sr is None:
            continue
        offrows.append(dict(Visit=f'jw01182004{vis}' if prop == '1182' else f'jw02221001{vis}',
                            Filter=filt.upper(), dra=sr / 1000., ddec=sd / 1000., nmatch=n))
        for (fra, fdec, ffl) in frames:
            all_ra.append(fra + (sr / 1000. / 3600.) / cosd); all_dec.append(fdec + (sd / 1000. / 3600.)); all_flux.append(ffl)
    print(f"  {len(offrows)} visits; {len(all_ra)} frames; {sum(len(a) for a in all_ra)} detections; combining...", flush=True)

    # incremental combine
    rad = MATCH_RADIUS_MAS / 1000. / 3600.; cap = 3_000_000
    g_ra = np.empty(cap); g_dec = np.empty(cap); g_flux = np.empty(cap)
    g_n = np.zeros(cap, int); g_m2r = np.zeros(cap); g_m2d = np.zeros(cap); ng = 0
    for (fra, fdec, fflux) in zip(all_ra, all_dec, all_flux):
        if ng == 0:
            n = len(fra); g_ra[:n] = fra; g_dec[:n] = fdec; g_flux[:n] = fflux; g_n[:n] = 1; ng = n; continue
        idx, sep, _ = SkyCoord(fra * u.deg, fdec * u.deg).match_to_catalog_sky(SkyCoord(g_ra[:ng] * u.deg, g_dec[:ng] * u.deg))
        mt = sep.deg < rad; gi = idx[mt]
        dra = (fra[mt] - g_ra[gi]) * cosd; ddec = (fdec[mt] - g_dec[gi])
        g_n[gi] += 1; g_ra[gi] += dra / g_n[gi] / cosd; g_dec[gi] += ddec / g_n[gi]
        g_m2r[gi] += dra * ((fra[mt] - g_ra[gi]) * cosd); g_m2d[gi] += ddec * (fdec[mt] - g_dec[gi])
        um = ~mt; k = int(um.sum()); g_ra[ng:ng+k] = fra[um]; g_dec[ng:ng+k] = fdec[um]; g_flux[ng:ng+k] = fflux[um]; g_n[ng:ng+k] = 1; ng += k
    g_ra = g_ra[:ng]; g_dec = g_dec[:ng]; g_flux = g_flux[:ng]; g_n = g_n[:ng]
    with np.errstate(invalid='ignore'):
        g_sr = np.where(g_n > 1, np.sqrt(g_m2r[:ng] / np.maximum(g_n - 1, 1)) * 3.6e6, np.nan)
        g_sd = np.where(g_n > 1, np.sqrt(g_m2d[:ng] / np.maximum(g_n - 1, 1)) * 3.6e6, np.nan)
    out = Table()
    out['RA'] = g_ra; out['DEC'] = g_dec; out['skycoord'] = SkyCoord(g_ra * u.deg, g_dec * u.deg)
    out['flux'] = g_flux; out['std_ra_mas'] = g_sr; out['std_dec_mas'] = g_sd; out['nframes'] = g_n
    out['nmatch'] = g_n; out['std_ra'] = g_sr / 3.6e6; out['std_dec'] = g_sd / 3.6e6
    out['qfit'] = np.zeros(len(out)); out['is_saturated'] = np.zeros(len(out), bool)
    out.meta['METHOD'] = 'CAL-GWCS SIAF positions + per-visit VIRAC2 shift (module-locked, exact SIAF)'
    out.meta['EPOCH'] = ep
    outpath = f'{CATDIR}/{filt}_merged_indivexp_CALLOCKED_dao_basic.fits'
    out.write(outpath, overwrite=True)
    mn = g_n >= 2
    print(f"  wrote {outpath}: {len(out)} groups; cross-frame scatter ({np.nanmedian(g_sr[mn]):.2f},{np.nanmedian(g_sd[mn]):.2f}) mas", flush=True)
    return offrows


if __name__ == '__main__':
    filts = sys.argv[1:] or ['f200w']
    allrows = {}
    for f in filts:
        allrows.setdefault(FILT_CFG[f][4], []).extend(lock_filter(f))
    os.makedirs(OFFDIR, exist_ok=True)
    for prop, rows in allrows.items():
        if not rows:
            continue
        t = Table(rows); t['dra (arcsec)'] = t['dra']; t['ddec (arcsec)'] = t['ddec']
        path = f'{OFFDIR}/Offsets_JWST_Brick{prop}_VIRAC2locked.csv'
        if os.path.exists(path):
            old = Table.read(path); keep = ~np.isin(old['Filter'], list(set(t['Filter'])))
            t = vstack([old[keep], t]) if keep.any() else t
        t.write(path, overwrite=True)
        print(f"wrote {path}: {len(t)} rows", flush=True)
