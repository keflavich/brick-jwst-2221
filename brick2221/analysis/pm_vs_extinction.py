#!/usr/bin/env python
"""Mean proper-motion vector +/- dispersion vs extinction, using JWST photometry
(F200W-F444W, CT06_MWGC law) for A_V.  Trustworthy JW-VVV PMs only.

Expectation (Brick): low A_V = foreground disk (distinct mean PM, low sigma);
high A_V = embedded CMZ population (bulk CMZ motion, higher sigma).
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from dust_extinction.averages import CT06_MWGC

OUT = '/orange/adamginsburg/jwst/brick/astrometry_diag/pm_flystar'
SNAP = '/blue/adamginsburg/adamginsburg/jwst/brick/astrometry_diag/m8_dedup_1182_snapshot_20260708.fits'

pm = Table.read(f'{OUT}/pm_jwst_virac_f212n.fits')
tr = pm[pm['trustworthy']]
pmc = SkyCoord(np.asarray(tr['ra0'])*u.deg, np.asarray(tr['dec0'])*u.deg)

# ---- JWST extinction: F200W - F444W, CT06_MWGC (cleaned photometry) ----------
s = Table.read(SNAP)
sc = s['skycoord_f200w']
m200 = np.asarray(s['mag_ab_f200w'], float); m444 = np.asarray(s['mag_ab_f444w'], float)
f200 = np.asarray(s['flux_jy_f200w'], float); e200 = np.asarray(s['eflux_jy_f200w'], float)
f444 = np.asarray(s['flux_jy_f444w'], float); e444 = np.asarray(s['eflux_jy_f444w'], float)
ff200 = np.asarray(s['forced_filled_f200w']) if 'forced_filled_f200w' in s.colnames else np.zeros(len(s), bool)
ff444 = np.asarray(s['forced_filled_f444w']) if 'forced_filled_f444w' in s.colnames else np.zeros(len(s), bool)
clean = (np.isfinite(sc.ra.deg) & np.isfinite(m200) & np.isfinite(m444)
         & (f200/e200 > 10) & (f444/e444 > 10) & ~ff200.astype(bool) & ~ff444.astype(bool))
ssc = SkyCoord(sc.ra.deg[clean]*u.deg, sc.dec.deg[clean]*u.deg)
m200c, m444c = m200[clean], m444[clean]
idx, sep, _ = pmc.match_to_catalog_sky(ssc)
law = CT06_MWGC()
coeff = float(law(2.0*u.um) - law(4.44*u.um))     # E(F200W-F444W)/A_V
color = m200c[idx] - m444c[idx]
# foreground zero-point: least-reddened 5% ~ disk stars -> A_V = 0 there
c0 = np.percentile(color[sep < 0.3*u.arcsec], 5)
AV = (color - c0) / coeff
print(f'matched clean broadbands: {(sep<0.3*u.arcsec).sum()}/{len(tr)};  '
      f'CT06 E(F200W-F444W)/A_V={coeff:.3f}; foreground color0={c0:.2f}')

pra = np.asarray(tr['pm_ra'], float); pde = np.asarray(tr['pm_dec'], float)
# clip color outliers (mismatches) to a physical A_V window
sel = (sep < 0.3*u.arcsec) & np.isfinite(AV) & (AV > -5) & (AV < 60) & np.isfinite(pra) & np.isfinite(pde)
AV, pra, pde = AV[sel], pra[sel], pde[sel]
print(f'AV range p5/p50/p95: {np.percentile(AV,[5,50,95]).round(1)}  N={sel.sum()}')

# ---- bin by A_V (equal-N quantile bins) --------------------------------------
nbin = 7
edges = np.quantile(AV, np.linspace(0, 1, nbin+1))
edges[0] -= 1e-6; edges[-1] += 1e-6
ib = np.digitize(AV, edges) - 1
rows = []
for b in range(nbin):
    m = ib == b
    if m.sum() < 20: continue
    n = m.sum()
    mra, mde = np.mean(pra[m]), np.mean(pde[m])
    sra, sde = np.std(pra[m], ddof=1), np.std(pde[m], ddof=1)
    rows.append(dict(av=np.median(AV[m]), n=n, mra=mra, mde=mde, sra=sra, sde=sde,
                     era=sra/np.sqrt(n), ede=sde/np.sqrt(n),
                     stot=np.sqrt(0.5*(sra**2+sde**2))))
B = Table(rows)
print('\n  A_V    N   <pmRA>    <pmDE>    sig_RA sig_DE sig_tot')
for r in B:
    print(f'{r["av"]:6.1f} {r["n"]:4d}  {r["mra"]:+6.2f}   {r["mde"]:+6.2f}    {r["sra"]:5.2f}  {r["sde"]:5.2f}  {r["stot"]:5.2f}')
B.write(f'{OUT}/pm_vs_extinction_bins.fits', overwrite=True)

# ---- figure ------------------------------------------------------------------
fig = plt.figure(figsize=(15, 9))
# A: mean pmRA, pmDE vs A_V (error bars = dispersion)
axA = fig.add_subplot(2, 3, 1)
axA.errorbar(B['av'], B['mra'], yerr=B['sra'], fmt='o-', color='C0', label='<pmRA> +/- disp', capsize=3)
axA.errorbar(B['av']+0.3, B['mde'], yerr=B['sde'], fmt='s-', color='C1', label='<pmDE> +/- disp', capsize=3)
axA.axhline(0, color='k', lw=0.5)
axA.set_xlabel('A_V (JWST F200W-F444W, CT06)'); axA.set_ylabel('mean PM (mas/yr)')
axA.legend(fontsize=8); axA.set_title('mean PM vector vs extinction (bars=dispersion)')
# A2: same but bars = error on the mean (is the mean SHIFT significant?)
axB = fig.add_subplot(2, 3, 2)
axB.errorbar(B['av'], B['mra'], yerr=B['era'], fmt='o-', color='C0', label='<pmRA> +/- SEM', capsize=3)
axB.errorbar(B['av']+0.3, B['mde'], yerr=B['ede'], fmt='s-', color='C1', label='<pmDE> +/- SEM', capsize=3)
axB.axhline(0, color='k', lw=0.5)
axB.set_xlabel('A_V'); axB.set_ylabel('mean PM (mas/yr)')
axB.legend(fontsize=8); axB.set_title('mean PM vs extinction (bars=SEM)')
# C: dispersion vs A_V
axC = fig.add_subplot(2, 3, 3)
axC.plot(B['av'], B['sra'], 'o-', label='sigma_RA')
axC.plot(B['av'], B['sde'], 's-', label='sigma_Dec')
axC.plot(B['av'], B['stot'], '^-', color='k', label='sigma_tot')
axC.set_xlabel('A_V'); axC.set_ylabel('PM dispersion (mas/yr)')
axC.legend(fontsize=8); axC.set_title('PM dispersion vs extinction')
# D: PM vectors per A_V bin (from origin), colored by A_V
axD = fig.add_subplot(2, 3, 4)
cmap = plt.cm.viridis((B['av']-B['av'].min())/(np.ptp(B["av"])+1e-9))
for r, c in zip(B, cmap):
    axD.arrow(0, 0, r['mra'], r['mde'], color=c, width=0.03, head_width=0.15,
              length_includes_head=True)
    axD.add_patch(plt.Circle((r['mra'], r['mde']), r['stot'], ec=c, fc='none', lw=0.8, alpha=0.6))
axD.axhline(0, color='k', lw=0.3); axD.axvline(0, color='k', lw=0.3)
axD.set_xlim(-8, 4); axD.set_ylim(-10, 2); axD.set_aspect('equal')
axD.set_xlabel('<pmRA> (mas/yr)'); axD.set_ylabel('<pmDE> (mas/yr)')
axD.set_title('mean PM vector per A_V bin (circle=sigma_tot)')
# E: N vs A_V
axE = fig.add_subplot(2, 3, 5)
axE.bar(range(len(B)), B['n']); axE.set_xticks(range(len(B)))
axE.set_xticklabels([f'{a:.0f}' for a in B['av']], fontsize=8)
axE.set_xlabel('A_V bin (median)'); axE.set_ylabel('N stars'); axE.set_title('stars per bin')
# F: VPD colored by A_V
axF = fig.add_subplot(2, 3, 6)
sctr = axF.scatter(pra, pde, c=AV, s=5, cmap='viridis', alpha=0.5)
axF.set_xlim(-20,15); axF.set_ylim(-22,12); axF.set_aspect('equal')
axF.set_xlabel('pmRA (mas/yr)'); axF.set_ylabel('pmDec (mas/yr)')
fig.colorbar(sctr, ax=axF, label='A_V'); axF.set_title('VPD colored by A_V')
fig.suptitle('JW-VVV proper motion vs JWST extinction (trustworthy set)', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT}/pm_vs_extinction.png', dpi=130, bbox_inches='tight')
print('wrote', f'{OUT}/pm_vs_extinction.png')
