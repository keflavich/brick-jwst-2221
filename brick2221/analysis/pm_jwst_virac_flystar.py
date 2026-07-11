#!/usr/bin/env python
"""JWST(F200W, 2022.70) x VIRAC2(2014.0) proper motions via flystar, then an
isolation filter for trustworthy PMs.  Compares the JW-VVV 2-epoch PM to the
VIRAC-catalog (VVV-only) PM and counts how many stars each method delivers.

Frame: VIRAC2/Gaia DR3.  JWST is affine-tied onto the VIRAC frame first
(shift_to_virac_frame), then VIRAC->JWST counterparts are matched with VIRAC's
own PM as a prior, and StarTable.fit_velocities fits the 2-epoch motion.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord, search_around_sky
from astropy import units as u
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-pm')
from jwst_gc_pipeline.astrometry.multiepoch_pm import tangent_xy, shift_to_virac_frame
from flystar.startables import StarTable

BASE = '/orange/adamginsburg/jwst/brick'
CAT = os.environ.get('JCAT', 'f200w')     # 'f200w' (1182 snapshot) or 'f212n' (2221 cross-tied)
OUTDIR = f'{BASE}/astrometry_diag/pm_flystar'; os.makedirs(OUTDIR, exist_ok=True)
EPOCH_V, EPOCH_J = 2014.0, 2022.70
DT = EPOCH_J - EPOCH_V
VVV_BEAM = 0.9          # arcsec: VVV seeing -> blend scale
TAG = CAT

SNAP = f'{BASE}/astrometry_diag/m8_dedup_1182_snapshot_20260708.fits'
F212 = f'{BASE}/catalogs/f212n_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
VIRAC = f'{BASE}/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'

# ---- JWST catalog (F200W deep 1182 pre-cross-tie, or F212N 2221 cross-tied) ---
if CAT == 'f200w':
    s = Table.read(SNAP); scj = s['skycoord_f200w']
    fj = np.asarray(s['flux_jy_f200w'], float); efj = np.asarray(s['eflux_jy_f200w'], float)
    ex = np.asarray(s['std_ra_f200w'], float); ey = np.asarray(s['std_dec_f200w'], float)
    ok = (np.isfinite(scj.ra.deg) & np.isfinite(fj) & (fj > 0) & np.isfinite(efj) & (efj > 0) & (fj/efj > 5))
    ra_j, dec_j = scj.ra.deg[ok], scj.dec.deg[ok]
else:
    s = Table.read(F212); scj = s['skycoord']
    fj = np.asarray(s['flux'], float); efj = np.asarray(s['flux_err'], float)
    ex = np.asarray(s['std_ra'], float); ey = np.asarray(s['std_dec'], float)
    ok = (np.isfinite(scj.ra.deg) & np.isfinite(fj) & (fj > 0) & np.isfinite(efj) & (efj > 0) & (fj/efj > 5))
    ra_j, dec_j = scj.ra.deg[ok], scj.dec.deg[ok]
jflux = fj[ok]
magj_instr = -2.5*np.log10(jflux)          # instrumental; zeropoint absorbed by color cal
jwst = dict(sc=SkyCoord(ra_j*u.deg, dec_j*u.deg),
            ex=np.where(np.isfinite(ex[ok])&(ex[ok]>0), ex[ok], 0.005),
            ey=np.where(np.isfinite(ey[ok])&(ey[ok]>0), ey[ok], 0.005),
            flux=jflux, mag_instr=magj_instr, epoch=EPOCH_J, n=int(ok.sum()))
print(f'JWST {CAT.upper()} SNR>5: {jwst["n"]:,}')

# ---- VIRAC2 ------------------------------------------------------------------
vt = Table.read(VIRAC)
vsc = SkyCoord(np.asarray(vt['RAJ2000'],float)*u.deg, np.asarray(vt['DEJ2000'],float)*u.deg)
virac = dict(sc=vsc, mag=np.asarray(vt['Ksmag'],float),
             ex=np.where(np.isfinite(np.asarray(vt['e_RAJ2000'],float)), np.asarray(vt['e_RAJ2000'],float)/1e3, 0.05),
             ey=np.where(np.isfinite(np.asarray(vt['e_DEJ2000'],float)), np.asarray(vt['e_DEJ2000'],float)/1e3, 0.05),
             pmra=np.asarray(vt['pmRA'],float), pmde=np.asarray(vt['pmDE'],float),
             epmra=np.asarray(vt['e_pmRA'],float), epmde=np.asarray(vt['e_pmDE'],float),
             epoch=EPOCH_V, n=len(vt))
N_VVV_ONLY = int(np.isfinite(virac['pmra']).sum())
print(f'VIRAC2 with catalog PM (VVV-only PMs): {N_VVV_ONLY:,}')

# ---- CONSTANT-shift tie JWST onto VIRAC frame (preserve the PM field) --------
# An affine tie would absorb the real bulk+gradient PM signal; the validated
# approach removes only a constant bulk offset.  VIRAC propagated to JWST epoch:
vra = virac['sc'].ra.deg + (np.nan_to_num(virac['pmra'])*DT/3.6e6)/np.cos(virac['sc'].dec.rad)
vde = virac['sc'].dec.deg + (np.nan_to_num(virac['pmde'])*DT/3.6e6)
vpred = SkyCoord(vra*u.deg, vde*u.deg)
# pass 1: coarse bright match -> constant shift
brightV = virac['mag'] < 14
i0, s0, _ = vpred[brightV].match_to_catalog_sky(jwst['sc'])
c0 = s0 < 0.5*u.arcsec
dra0 = (jwst['sc'].ra.deg[i0][c0] - vpred[brightV].ra.deg[c0])*np.cos(np.radians(vpred[brightV].dec.deg[c0]))
dde0 = (jwst['sc'].dec.deg[i0][c0] - vpred[brightV].dec.deg[c0])
shift_ra, shift_de = np.median(dra0), np.median(dde0)   # deg (on-sky RA, dec)
print(f'constant tie shift: ({shift_ra*3.6e6:+.1f},{shift_de*3.6e6:+.1f}) mas from {c0.sum()} bright')
jwst['sc'] = SkyCoord((jwst['sc'].ra.deg - shift_ra/np.cos(jwst['sc'].dec.rad))*u.deg,
                      (jwst['sc'].dec.deg - shift_de)*u.deg)

# ---- match VIRAC(pm-propagated) -> JWST, mutual NN + FLUX VET -----------------
idx, sep, _ = vpred.match_to_catalog_sky(jwst['sc'])
idxb, _, _ = jwst['sc'].match_to_catalog_sky(vpred)
mutual = idxb[idx] == np.arange(len(vpred))
# flux vet: (JWST instr mag) - Ks forms a tight ridge; auto-calibrate the ridge
# centre from bright tight pairs, keep within +/-0.75 mag -> rejects crowded mispairs
color = jwst['mag_instr'][idx] - virac['mag']
tight = (sep < 0.15*u.arcsec) & mutual & (virac['mag'] < 15) & np.isfinite(color)
CO = np.median(color[tight])
print(f'flux-vet ridge (JWST_instr - Ks) = {CO:.2f} from {tight.sum()} bright tight')
fluxvet = np.abs(color - CO) < 0.75
matched = (sep < 0.3*u.arcsec) & mutual & fluxvet
print(f'VIRAC->JWST matched (mutual NN <0.3" + flux-vet): {matched.sum():,} '
      f'(mutual<0.3" pre-vet: {((sep<0.3*u.arcsec)&mutual).sum():,})')

# ---- 2-epoch StarTable + fit_velocities --------------------------------------
center = SkyCoord(np.median(virac['sc'].ra), np.median(virac['sc'].dec))
vx, vy = tangent_xy(virac['sc'], center)
jx, jy = tangent_xy(jwst['sc'][idx], center)
nref = virac['n']
X = np.full((nref,2), np.nan); Y = np.full((nref,2), np.nan)
XE = np.full((nref,2), np.nan); YE = np.full((nref,2), np.nan); Mg = np.full((nref,2), np.nan)
X[:,0], Y[:,0], XE[:,0], YE[:,0], Mg[:,0] = vx, vy, virac['ex'], virac['ey'], virac['mag']
X[matched,1] = jx[matched]; Y[matched,1] = jy[matched]
XE[matched,1] = jwst['ex'][idx][matched]; YE[matched,1] = jwst['ey'][idx][matched]
Mg[matched,1] = (jwst['mag_instr'][idx][matched] - CO)   # JWST on the Ks scale
sel = matched.copy()
name = np.array([f'v{i}' for i in np.where(sel)[0]])
st = StarTable(name=name, x=X[sel], y=Y[sel], m=Mg[sel], xe=XE[sel], ye=YE[sel],
               LIST_TIMES=[EPOCH_V, EPOCH_J], ref_list=0)
st.fit_velocities(use_scipy=True, show_progress=False, mask_val=np.nan)

pm = Table()
pm['ra0'] = center.ra.deg + (st['x0']/3600.0)/np.cos(center.dec.rad)
pm['dec0'] = center.dec.deg + (st['y0']/3600.0)
pm['pm_ra'] = st['vx']*1e3; pm['pm_dec'] = st['vy']*1e3
pm['pm_tot'] = np.hypot(pm['pm_ra'], pm['pm_dec'])
selidx = np.where(sel)[0]
# flystar 2-epoch velocity errors are degenerate (0 dof) -> analytic instead:
# 2-epoch PM err = hypot(endpoint position errors)/baseline
pm['pm_ra_err'] = np.hypot(virac['ex'][selidx], jwst['ex'][idx][selidx])*1e3/DT
pm['pm_dec_err'] = np.hypot(virac['ey'][selidx], jwst['ey'][idx][selidx])*1e3/DT
pm['virac_pmra'] = virac['pmra'][selidx]; pm['virac_pmde'] = virac['pmde'][selidx]
pm['virac_pmtot'] = np.hypot(pm['virac_pmra'], pm['virac_pmde'])
pm['virac_epmtot'] = np.hypot(virac['epmra'][selidx], virac['epmde'][selidx])
pm['Ks'] = virac['mag'][selidx]

# ---- isolation criteria (trustworthy) ----------------------------------------
# The VVV/VIRAC 2014 centroid is a single dominant star iff no COMPARABLY BRIGHT
# JWST source (flux > primary/DOMRATIO, ~1.5 mag) sits within the VVV beam; faint
# companions do not move the VVV centroid.  Also require no other VIRAC star <1.0".
# A clean VVV/VIRAC 2014 centroid needs the primary to dominate the beam by a wide
# margin: a companion f_c at separation s shifts the VVV centroid by ~s*f_c/f_p, so
# even a 1.5-mag companion (f/4) at 0.5" moves it ~60 mas (=7 mas/yr).  Require any
# in-beam companion >2.5 mag fainter (f/10) AND no comparable star within 0.35".
prim_jidx = idx[selidx]
prim_flux = jwst['flux'][prim_jidx]
vi, ji, d2d, _ = search_around_sky(vpred[sel], jwst['sc'], VVV_BEAM*u.arcsec)
comp_flux = np.zeros(int(sel.sum()))            # brightest companion in beam
comp_flux_close = np.zeros(int(sel.sum()))      # brightest companion within 0.35"
for g, b, dd in zip(vi, ji, d2d.arcsec):
    if b == prim_jidx[g]:
        continue
    if jwst['flux'][b] > comp_flux[g]:
        comp_flux[g] = jwst['flux'][b]
    if dd < 0.35 and jwst['flux'][b] > comp_flux_close[g]:
        comp_flux_close[g] = jwst['flux'][b]
frac = np.where(prim_flux > 0, comp_flux/prim_flux, np.nan)
frac_close = np.where(prim_flux > 0, comp_flux_close/prim_flux, np.nan)
vv_i, vv_j, vv_d, _ = search_around_sky(virac['sc'][sel], virac['sc'], 1.0*u.arcsec)
n_virac_near = np.bincount(vv_i, minlength=sel.sum())
good_err = np.isfinite(pm['pm_ra_err']) & (pm['pm_ra_err'] < 2) & (pm['pm_dec_err'] < 2)
dominant = (frac < 0.10) & (frac_close < 0.25) & (n_virac_near == 1)
trust = dominant & good_err
pm['comp_flux_ratio'] = frac
pm['n_virac_within_1arcsec'] = n_virac_near
pm['dominant'] = dominant
pm['trustworthy'] = trust
pm.meta['epochs'] = [EPOCH_V, EPOCH_J]; pm.meta['frame'] = 'VIRAC2/Gaia DR3'
pm.write(f'{OUTDIR}/pm_jwst_virac_{TAG}.fits', overwrite=True)

# ---- counts + validation -----------------------------------------------------
n_match = int(sel.sum()); n_trust = int(trust.sum())
n_blended = int((frac >= 0.10).sum())
print('\n==== PM YIELD ====')
print(f'VVV-only PMs (VIRAC catalog):          {N_VVV_ONLY:,}')
print(f'JW-VVV PMs (matched, flux-vetted):     {n_match:,}')
print(f'JW-VVV PMs TRUSTWORTHY (isolated):     {n_trust:,}')
print(f'  matched with comparable blend (frac>0.1 in VVV beam): {n_blended:,} '
      f'({100*n_blended/n_match:.0f}%) -> VVV-only PM corrupted for these')

def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    c = np.corrcoef(a[m], b[m])[0, 1]
    slope = np.polyfit(a[m], b[m], 1)[0]
    resid = b[m] - a[m]
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    return c, slope, mad, m.sum()
print('\ntrustworthy JW-VVV vs VIRAC catalog PM:')
for lab, jw, vc in [('pmRA', pm['pm_ra'], pm['virac_pmra']), ('pmDE', pm['pm_dec'], pm['virac_pmde'])]:
    c, sl, mad, nn = stats(np.asarray(vc[trust]), np.asarray(jw[trust]))
    print(f'  {lab}: corr={c:.2f} slope={sl:.2f} resid_mad={mad:.2f} mas/yr (n={nn})')
t = pm[trust]

# ---- plots -------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for a,(x,y,lab) in zip(ax[:2], [(t['virac_pmra'],t['pm_ra'],'pmRA'),(t['virac_pmde'],t['pm_dec'],'pmDE')]):
    a.plot(x,y,'.',ms=2,alpha=0.3)
    lim=[-30,30]; a.plot(lim,lim,'r-',lw=1); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel(f'VIRAC catalog {lab} (mas/yr)'); a.set_ylabel(f'JW-VVV {lab} (mas/yr)')
    a.set_title(f'{lab}  (trustworthy, n={len(t)})'); a.set_aspect('equal')
ax[2].bar(['VVV-only\n(VIRAC cat)','JW-VVV\nmatched','JW-VVV\ntrustworthy'],
          [N_VVV_ONLY, n_match, n_trust], color=['gray','steelblue','seagreen'])
ax[2].set_ylabel('N stars with a PM'); ax[2].set_title('PM yield')
for i,v in enumerate([N_VVV_ONLY,n_match,n_trust]): ax[2].text(i,v,f'{v:,}',ha='center',va='bottom',fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/pm_yield_and_agreement_{TAG}.png', dpi=130, bbox_inches='tight')
print('\nwrote', f'{OUTDIR}/pm_jwst_virac_{TAG}.fits', 'and pm_yield_and_agreement_{TAG}.png')
