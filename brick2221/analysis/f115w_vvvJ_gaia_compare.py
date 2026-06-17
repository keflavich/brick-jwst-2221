#!/usr/bin/env python
"""F115W first-iteration catalog vs VVV J-band (flux-matched) vs Gaia DR3.

F115W (1.15um) is the bluest NIRCam band and the cornerstone tying Gaia to the
redder filters.  VVV J (1.25um) is the closest external NIR band, so we flux-match
F115W instrumental magnitudes to VVV J.  Gaia DR3 is propagated from its 2016.0
reference epoch to the JWST F115W epoch (2022.70) using pmRA/pmDE before matching.

Outputs a markdown report + plots under astrometry_diag/f115w_compare/.
"""
import os
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/orange/adamginsburg/jwst/brick'
OUT = f'{BASE}/astrometry_diag/f115w_compare'
os.makedirs(OUT, exist_ok=True)

F115W_EPOCH = 2022.70          # MJD 59836.66 EXPSTART, o004
GAIA_REF_EPOCH = 2016.0        # Gaia DR3
DT = F115W_EPOCH - GAIA_REF_EPOCH
MAX_SEP = 0.4 * u.arcsec
NSIGMA = 3.0

CATFILE = f'{BASE}/catalogs/f115w_merged_indivexp_merged_dao_basic.fits'
VVV_CACHE = f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_vvv.fits'
GAIA_CACHE = f'{BASE}/astrometry_analysis/reference_cache/basic_merged_photometry_tables_merged_gaia.fits'


def farr(x):
    return np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(x, dtype=float)), np.nan), dtype=float)


def mutual_match(cat, ref, max_sep):
    """Mutual nearest-neighbour match. Returns cat_idx, ref_idx, dra_mas, ddec_mas, sep_mas."""
    cat = SkyCoord(cat).icrs
    ref = SkyCoord(ref).icrs
    ci, sep, _ = cat.match_to_catalog_sky(ref)
    ri, _, _ = ref.match_to_catalog_sky(cat)
    mutual = ri[ci] == np.arange(len(ci))
    keep = mutual & np.isfinite(sep.deg) & (sep <= max_sep)
    cidx = np.where(keep)[0]
    ridx = ci[keep]
    mc = cat[cidx]
    mr = ref[ridx]
    # ref -> cat offsets (mas), manual small-angle
    dra = ((mc.ra.deg - mr.ra.deg) * np.cos(np.radians(mc.dec.deg)) * 3.6e6)
    ddec = ((mc.dec.deg - mr.dec.deg) * 3.6e6)
    return cidx, ridx, dra, ddec, mc.separation(mr).to(u.mas).value


def summarize(dra, ddec, sep):
    n = len(dra)
    if n == 0:
        return dict(n=0, mdra=np.nan, mddec=np.nan, vec=np.nan, mad_ra=np.nan,
                    mad_dec=np.nan, med_sep=np.nan, sem_vec=np.nan)
    mdra, mddec = np.nanmedian(dra), np.nanmedian(ddec)
    mra, mdec = stats.mad_std(dra, ignore_nan=True), stats.mad_std(ddec, ignore_nan=True)
    return dict(n=n, mdra=mdra, mddec=mddec, vec=np.hypot(mdra, mddec),
                mad_ra=mra, mad_dec=mdec, med_sep=np.nanmedian(sep),
                sem_vec=np.hypot(mra, mdec) / np.sqrt(n))


def fmt(s):
    return (f"N={s['n']:5d}  dRA={s['mdra']:7.2f}  dDec={s['mddec']:7.2f}  "
            f"|vec|={s['vec']:6.2f}+-{s['sem_vec']:.2f}  MAD=({s['mad_ra']:.1f},{s['mad_dec']:.1f})  "
            f"medsep={s['med_sep']:.1f}  [mas]")


# ---------- load F115W ----------
cat = Table.read(CATFILE)
cat_sc = cat['skycoord']
flux = farr(cat['flux'])
good = np.isfinite(flux) & (flux > 0)
cat = cat[good]; cat_sc = cat_sc[good]; flux = flux[good]
m_inst = -2.5 * np.log10(flux)   # instrumental F115W mag
print(f"F115W catalog: {CATFILE}")
print(f"  {len(cat)} sources with positive flux")

# ---------- VVV J ----------
vvv = Table.read(VVV_CACHE)
vvv_sc = vvv['skycoord']
# J 3rd-aperture, prefer epoch-1 then epoch-2
J = farr(vvv['J1ap3']); Je = farr(vvv['e_J1ap3'])
use2 = ~np.isfinite(J)
J[use2] = farr(vvv['J2ap3'])[use2]; Je[use2] = farr(vvv['e_J2ap3'])[use2]
print(f"\nVVV: {len(vvv)} rows, {np.isfinite(J).sum()} with finite J")

ci, ri, dra, ddec, sep = mutual_match(cat_sc, vvv_sc, MAX_SEP)
s_vvv_raw = summarize(dra, ddec, sep)
print("\nF115W vs VVV  (spatial only):      ", fmt(s_vvv_raw))

# flux-match: instrumental F115W vs VVV J, robust median offset + nsigma clip
mi = m_inst[ci]; Jm = J[ri]; Jer = Je[ri]
fin = np.isfinite(mi) & np.isfinite(Jm) & np.isfinite(Jer) & (Jer > 0)
dmag = mi - Jm
med_dmag = np.nanmedian(dmag[fin])
scatter = stats.mad_std(dmag[fin], ignore_nan=True)
phot_ok = fin & (np.abs(dmag - med_dmag) <= NSIGMA * np.maximum(Jer, scatter))
s_vvv_fm = summarize(dra[phot_ok], ddec[phot_ok], sep[phot_ok])
print(f"F115W vs VVV-J (flux-matched 3sig): ", fmt(s_vvv_fm))
print(f"   ZP(m_inst - J) median = {med_dmag:.3f}, color/scatter = {scatter:.3f} mag; "
      f"kept {phot_ok.sum()}/{fin.sum()}")
# keep matched VVV J positions/info for agreement test
vvv_match = dict(ci=ci[phot_ok], ri=ri[phot_ok], dra=dra[phot_ok], ddec=ddec[phot_ok])

# ---------- Gaia (PM-propagated to F115W epoch) ----------
g = Table.read(GAIA_CACHE)
gra = farr(g['RA_ICRS']); gdec = farr(g['DE_ICRS'])
pmra = farr(g['pmRA']); pmde = farr(g['pmDE'])   # mas/yr, pmRA already *cos(dec)
pmra0 = np.where(np.isfinite(pmra), pmra, 0.0)
pmde0 = np.where(np.isfinite(pmde), pmde, 0.0)
gra_p = gra + (pmra0 * DT / 1000.0 / 3600.0) / np.cos(np.radians(gdec))
gdec_p = gdec + (pmde0 * DT / 1000.0 / 3600.0)
gaia_sc = SkyCoord(gra_p * u.deg, gdec_p * u.deg, frame='icrs')
gaia_sc_noPM = SkyCoord(gra * u.deg, gdec * u.deg, frame='icrs')
Gmag = farr(g['Gmag'])
print(f"\nGaia: {len(g)} rows; propagated {DT:.2f} yr (med |PM|={np.nanmedian(np.hypot(pmra,pmde)):.1f} mas/yr)")

ci_g, ri_g, dra_g, ddec_g, sep_g = mutual_match(cat_sc, gaia_sc, MAX_SEP)
s_gaia = summarize(dra_g, ddec_g, sep_g)
print("\nF115W vs Gaia (PM-corrected):      ", fmt(s_gaia))
cn, rn, drn, ddn, spn = mutual_match(cat_sc, gaia_sc_noPM, MAX_SEP)
print("F115W vs Gaia (NO PM correction):  ", fmt(summarize(drn, ddn, spn)))
gaia_match = dict(ci=ci_g, ri=ri_g, dra=dra_g, ddec=ddec_g)

# ---------- agreement: do VVV and Gaia frames agree? ----------
# (a) systematic offset vectors
print("\n=== AGREEMENT ===")
print(f"VVV-J  systematic offset: dRA={s_vvv_fm['mdra']:.2f} dDec={s_vvv_fm['mddec']:.2f} |{s_vvv_fm['vec']:.2f}| mas")
print(f"Gaia   systematic offset: dRA={s_gaia['mdra']:.2f} dDec={s_gaia['mddec']:.2f} |{s_gaia['vec']:.2f}| mas")
ddra = s_vvv_fm['mdra'] - s_gaia['mdra']
dddec = s_vvv_fm['mddec'] - s_gaia['mddec']
print(f"Difference (VVV - Gaia):  dRA={ddra:.2f} dDec={dddec:.2f} |{np.hypot(ddra,dddec):.2f}| mas")

# (b) F115W sources matched to BOTH refs: compare the two implied positions per-star
common = np.intersect1d(vvv_match['ci'], gaia_match['ci'])
print(f"\nF115W sources matched to BOTH VVV-J and Gaia: {len(common)}")
if len(common) > 0:
    vmap = {c: k for k, c in enumerate(vvv_match['ci'])}
    gmap = {c: k for k, c in enumerate(gaia_match['ci'])}
    dvg_ra = np.array([(vvv_match['dra'][vmap[c]] - gaia_match['dra'][gmap[c]]) for c in common])
    dvg_dec = np.array([(vvv_match['ddec'][vmap[c]] - gaia_match['ddec'][gmap[c]]) for c in common])
    # (F115W-VVV) - (F115W-Gaia) = Gaia - VVV position per common star
    print(f"  per-star (VVV pos - Gaia pos): med dRA={np.median(dvg_ra):.2f} dDec={np.median(dvg_dec):.2f} "
          f"|{np.hypot(np.median(dvg_ra),np.median(dvg_dec)):.2f}| mas, "
          f"scatter=({stats.mad_std(dvg_ra):.1f},{stats.mad_std(dvg_dec):.1f})")

# (c) direct VVV<->Gaia frame match (independent of JWST)
vg_ci, vg_ri, vg_dra, vg_ddec, vg_sep = mutual_match(gaia_sc, vvv_sc, 0.5 * u.arcsec)
print(f"\nDirect Gaia<->VVV match (no JWST): {len(vg_ci)} pairs, "
      f"med (VVV-Gaia) dRA={np.nanmedian(vg_dra):.2f} dDec={np.nanmedian(vg_ddec):.2f} "
      f"|{np.hypot(np.nanmedian(vg_dra),np.nanmedian(vg_ddec)):.2f}| mas")

# ---------- plots ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (dr, dd, lab, s) in zip(axes, [
        (vvv_match['dra'], vvv_match['ddec'], 'F115W - VVV J (flux-matched)', s_vvv_fm),
        (dra_g, ddec_g, 'F115W - Gaia (PM-corrected)', s_gaia)]):
    lim = 300
    ax.scatter(dr, dd, s=4, alpha=0.25, color='k', rasterized=True)
    ax.axhline(0, color='0.6'); ax.axvline(0, color='0.6')
    ax.arrow(0, 0, s['mdra'], s['mddec'], color='red', width=2, length_includes_head=True, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('dRA cos(dec) [mas]'); ax.set_ylabel('dDec [mas]')
    ax.set_title(f"{lab}\nN={s['n']} med=({s['mdra']:.1f},{s['mddec']:.1f}) "
                 f"|vec|={s['vec']:.1f} MAD=({s['mad_ra']:.0f},{s['mad_dec']:.0f}) mas", fontsize=10)
fig.tight_layout()
fig.savefig(f'{OUT}/f115w_vvvJ_vs_gaia_offsets.png', dpi=150, bbox_inches='tight')
print(f"\nWrote {OUT}/f115w_vvvJ_vs_gaia_offsets.png")

# write report
with open(f'{OUT}/REPORT.md', 'w') as fh:
    fh.write("# F115W first-iteration astrometry: VVV J vs Gaia\n\n")
    fh.write(f"Catalog: `{os.path.basename(CATFILE)}` ({len(cat)} sources), F115W epoch {F115W_EPOCH}.\n")
    fh.write(f"Match: mutual NN, max_sep {MAX_SEP}. Gaia PM-propagated {DT:.2f} yr to JWST epoch.\n\n")
    fh.write("| comparison | N | dRA mas | dDec mas | \\|vec\\| mas | sem | MAD_RA | MAD_Dec | med_sep |\n")
    fh.write("|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for lab, s in [("VVV J (spatial only)", s_vvv_raw),
                   ("VVV J (flux-matched)", s_vvv_fm),
                   ("Gaia DR3 (PM-corrected)", s_gaia)]:
        fh.write(f"| {lab} | {s['n']} | {s['mdra']:.2f} | {s['mddec']:.2f} | {s['vec']:.2f} | "
                 f"{s['sem_vec']:.2f} | {s['mad_ra']:.1f} | {s['mad_dec']:.1f} | {s['med_sep']:.1f} |\n")
    fh.write(f"\nVVV-Gaia systematic difference: |{np.hypot(ddra,dddec):.2f}| mas "
             f"(dRA={ddra:.2f}, dDec={dddec:.2f}).\n")
    fh.write(f"Direct Gaia<->VVV frame offset (no JWST): "
             f"|{np.hypot(np.nanmedian(vg_dra),np.nanmedian(vg_ddec)):.2f}| mas over {len(vg_ci)} pairs.\n")
print(f"Wrote {OUT}/REPORT.md")
