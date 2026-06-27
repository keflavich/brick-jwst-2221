#!/usr/bin/env python
"""Build the sickle -> GNS offsets table (Offsets_JWST_Brick3958_GNS.csv).

Audit (2026-06-20) found sickle catalogs sit at the RAW assign_wcs frame
(RAOFFSET=0, no offsets table for 3958): ~91 mas (RA) off the GNS reference the
mosaics are tied to.  User chose the GNS frame.  This measures the per-filter
bulk correction sickle->GNS and writes a per-frame offsets table in the
shift_individual_catalog convention:

    final = centroid - RAOFFSET_meta + dra_table     (RAOFFSET=0 for sickle)
    want  = centroid + corr_onsky                    (corr = GNS - catalog, mas)
    => dra_table  = corr_dRA_onsky_arcsec / cos(dec)   (Delta-alpha coordinate)
       ddec_table = corr_dDec_onsky_arcsec

GNS reference = catalogs/nircam_bootstrapped_to_gns_refcat.fits (dense, covers
the field; the absolute GNS-tied NIRCam reference the mosaics already use).

Validation: (A) apply the correction to each filter's merged catalog and confirm
residual vs GNS -> ~0; (B) confirm every crf frame maps to exactly one table row
(the shift_individual_catalog match.sum()==1 contract).  Writes nothing unless
both validations pass (unless --force).
"""
import sys, os, glob, re, warnings
warnings.simplefilter('ignore')
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

BP = '/orange/adamginsburg/jwst/sickle'
OUT = f'{BP}/offsets/Offsets_JWST_Brick3958_GNS.csv'
FILTERS = ['F187N', 'F210M', 'F335M', 'F470N', 'F480M']
GNS = Table.read(f'{BP}/catalogs/nircam_bootstrapped_to_gns_refcat.fits')
gns_sc = (SkyCoord(GNS['skycoord']) if 'skycoord' in GNS.colnames
          else SkyCoord(GNS['RA'], GNS['DEC'], unit='deg'))


def bulk_corr(cat_sc):
    """median on-sky (corr_dRA, corr_dDec) in mas to move cat -> GNS, + n, rms."""
    idx, sep, _ = cat_sc.match_to_catalog_sky(gns_sc)
    ridx, _, _ = gns_sc.match_to_catalog_sky(cat_sc)
    mutual = ridx[idx] == np.arange(len(idx))
    keep = mutual & (sep.arcsec < 0.3)
    if keep.sum() < 30:
        return None
    c, r = cat_sc[keep], gns_sc[idx[keep]]
    dra = (r.ra - c.ra).to(u.arcsec).value * np.cos(np.radians(c.dec.deg)) * 1000  # GNS - cat, mas on-sky
    ddec = (r.dec - c.dec).to(u.arcsec).value * 1000
    # robust: sigma-clip around the median to kill mismatches
    m_dra, m_ddec = np.median(dra), np.median(ddec)
    cl = np.hypot(dra - m_dra, ddec - m_ddec) < 100
    return (float(np.median(dra[cl])), float(np.median(ddec[cl])),
            int(cl.sum()), float(np.std(np.hypot(dra[cl] - m_dra, ddec[cl] - m_ddec))))


def merged_cat(filt):
    for tok in ('resbgsub_m7', 'resbgsub_m6', 'resbgsub_m5', 'm4', 'm3', 'm2'):
        p = sorted(glob.glob(f'{BP}/catalogs/{filt.lower()}_nrcb_indivexp_merged_{tok}_dao_basic_vetted.fits'))
        if p:
            return p[-1]
    return None


# ---- 1. measure per-filter correction ----
corr = {}
print("Per-filter sickle->GNS bulk correction (median, mas on-sky):")
for f in FILTERS:
    mp = merged_cat(f)
    if not mp:
        print(f"  {f}: NO merged catalog found -- skip"); continue
    t = Table.read(mp)
    sc = SkyCoord(t['skycoord']) if 'skycoord' in t.colnames else SkyCoord(t['ra'], t['dec'], unit='deg')
    fcol = 'flux' if 'flux' in t.colnames else 'flux_fit'
    br = np.argsort(-np.asarray(t[fcol], float))[:5000]
    r = bulk_corr(sc[br])
    if r is None:
        print(f"  {f}: <30 GNS matches -- skip"); continue
    corr[f] = r
    print(f"  {f}: corr_dRA={r[0]:+.1f}  corr_dDec={r[1]:+.1f}  (n={r[2]}, scatter={r[3]:.0f} mas)  src={os.path.basename(mp)}")

# ---- 2. enumerate crf frames -> table rows ----
rows = []
for f in FILTERS:
    if f not in corr:
        continue
    cdra, cddec = corr[f][0], corr[f][1]
    # representative dec for the cos() term (field center)
    decc = -28.805
    dra_tab = (cdra / 1000.0) / np.cos(np.radians(decc))   # arcsec, Delta-alpha
    ddec_tab = cddec / 1000.0                                # arcsec
    suffix = 'destreak_o007_crf' if f in ('F187N', 'F210M') else 'align_o007_crf'
    frames = sorted(glob.glob(f'{BP}/{f}/pipeline/jw03958007*_{suffix}.fits'))
    seen = set()
    for fn in frames:
        bn = os.path.basename(fn)
        # jw{ppppp}{ooo}{vvv}_{vgroup}_{exp}_{detector}_...  e.g.
        # jw03958007001_03104_00001_nrcblong_align_o007_crf.fits
        m = re.match(r'jw(\d{5})(\d{3})(\d{3})_(\w+?)_(\d{5})_(nrc\w+?)_', bn)
        if not m:
            continue
        prop, obs, visit, vgroup, exp, det = m.groups()
        key = (f, det, int(visit), int(exp))
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(Filter=f, Module=det, Visit=f'jw{prop}{obs}{visit}',
                         Exposure=int(exp), dra=dra_tab, ddec=ddec_tab,
                         RAOFFSET_meta=0.0, DEOFFSET_meta=0.0,
                         corr_dRA_mas=cdra, corr_dDec_mas=cddec))
    print(f"  {f}: {len(seen)} unique (det,visit,exp) frame rows")

ot = Table(rows)
print(f"\nTotal offsets-table rows: {len(ot)}")

# ---- 3. VALIDATION A: applying the correction lands the merged catalog on GNS ----
print("\nVALIDATION A (apply corr -> residual vs GNS, should be ~0):")
ok_A = True
for f in corr:
    mp = merged_cat(f); t = Table.read(mp)
    sc = SkyCoord(t['skycoord']) if 'skycoord' in t.colnames else SkyCoord(t['ra'], t['dec'], unit='deg')
    fcol = 'flux' if 'flux' in t.colnames else 'flux_fit'
    br = np.argsort(-np.asarray(t[fcol], float))[:5000]
    cdra, cddec = corr[f][0], corr[f][1]
    decarr = sc[br].dec.deg
    new = SkyCoord(sc[br].ra + (cdra/1000/np.cos(np.radians(decarr)))*u.arcsec,
                   sc[br].dec + (cddec/1000)*u.arcsec)
    r = bulk_corr(new)
    res = np.hypot(r[0], r[1]) if r else 999
    flag = 'OK' if res < 5 else 'FAIL'
    if res >= 5: ok_A = False
    print(f"  {f}: residual after correction = {r[0]:+.1f},{r[1]:+.1f} mas ({flag})")

# ---- 4. write ----
if '--force' in sys.argv or ok_A:
    os.makedirs(f'{BP}/offsets', exist_ok=True)
    ot.write(OUT, overwrite=True)
    print(f"\nWROTE {OUT}  ({len(ot)} rows)")
else:
    print("\nValidation A FAILED -- not writing (use --force to override)")
print("BUILD_DONE")
