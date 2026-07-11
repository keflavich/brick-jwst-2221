#!/usr/bin/env python
"""(1) Quantify the PRECISION-RESCUE: VIRAC catalog PMs degrade steeply with
faintness (e_pm median 2.9, ->6.7 at Ks 16-17), while the JW-VVV 2-epoch PM is
limited by VIRAC's 2014 POSITION error (not its PM error), giving a ~1.8 mas/yr
floor regardless.  So JW-VVV 'rescues' the many VIRAC stars whose own PM error
is too large to use.  (2) VPD + bulk-subtracted PM map of the trustworthy set.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.table import Table

OUT = '/orange/adamginsburg/jwst/brick/astrometry_diag/pm_flystar'
pm = Table.read(f'{OUT}/pm_jwst_virac_f212n.fits')
tr = pm[pm['trustworthy']]
JW_FLOOR = 1.8            # empirical JW-VVV resid_mad vs VIRAC (mas/yr), F212N frame

epm = np.asarray(tr['virac_epmtot'], float)     # VIRAC catalog PM uncertainty
n = len(tr)
print(f'trustworthy JW-VVV PMs: {n}')
# JW-VVV precision (empirical floor) beats VIRAC when VIRAC e_pm > floor
rescued = epm > JW_FLOOR
strong = epm > 3.0                              # VIRAC PM effectively unusable
print(f'  VIRAC e_pm > {JW_FLOOR} (JW-VVV more precise): {rescued.sum()} ({100*rescued.mean():.0f}%)')
print(f'  VIRAC e_pm > 3 (VIRAC PM unusable, JW rescues): {strong.sum()} ({100*strong.mean():.0f}%)')
print(f'  VIRAC e_pm < 1 (both good, cross-validation):   {(epm<1).sum()} ({100*(epm<1).mean():.0f}%)')

# whole-catalog usable-PM counts at matched quality (~2 mas/yr)
allpm = Table.read('/orange/adamginsburg/jwst/brick/astrometry_retie_qualcuts_20251211/refcache/virac2.fits')
eall = np.hypot(np.asarray(allpm['e_pmRA'],float), np.asarray(allpm['e_pmDE'],float))
print(f'\nwhole VIRAC2 footprint {len(allpm)}: e_pm<2 usable {int((eall<2).sum())} ({100*(eall<2).mean():.0f}%)')

# ---- bulk motion + residuals -------------------------------------------------
pra = np.asarray(tr['pm_ra'], float); pde = np.asarray(tr['pm_dec'], float)
bulk_ra, bulk_de = np.median(pra), np.median(pde)
rra, rde = pra - bulk_ra, pde - bulk_de
print(f'\nbulk PM (median, VIRAC frame): ({bulk_ra:+.2f},{bulk_de:+.2f}) mas/yr; '
      f'dispersion ({np.std(rra):.2f},{np.std(rde):.2f})')

# ---- figure ------------------------------------------------------------------
fig = plt.figure(figsize=(15, 9))
# A: VPD raw
axA = fig.add_subplot(2, 3, 1)
axA.hexbin(pra, pde, gridsize=45, cmap='viridis', mincnt=1, extent=[-25,25,-25,25])
axA.plot(bulk_ra, bulk_de, 'r+', ms=16, mew=2)
axA.set_xlim(-25,25); axA.set_ylim(-25,25); axA.set_aspect('equal')
axA.set_xlabel('pm_RA (mas/yr)'); axA.set_ylabel('pm_Dec (mas/yr)')
axA.set_title(f'JW-VVV VPD (trustworthy n={n})\nbulk ({bulk_ra:+.1f},{bulk_de:+.1f})')
# B: bulk-subtracted VPD
axB = fig.add_subplot(2, 3, 2)
axB.hexbin(rra, rde, gridsize=45, cmap='magma', mincnt=1, extent=[-20,20,-20,20])
th=np.linspace(0,2*np.pi,100)
for r in (5,10): axB.plot(r*np.cos(th), r*np.sin(th), 'w-', lw=0.5, alpha=0.5)
axB.set_xlim(-20,20); axB.set_ylim(-20,20); axB.set_aspect('equal')
axB.set_xlabel('Δpm_RA (mas/yr)'); axB.set_ylabel('Δpm_Dec (mas/yr)')
axB.set_title('bulk-subtracted VPD')
# C: spatial residual-PM vector field
axC = fig.add_subplot(2, 3, 3)
ra0 = np.asarray(tr['ra0'], float); dec0 = np.asarray(tr['dec0'], float)
q = axC.quiver(ra0, dec0, rra, rde, np.hypot(rra,rde), cmap='plasma',
               scale=180, width=0.003, clim=(0,12))
axC.invert_xaxis(); axC.set_aspect('equal', adjustable='datalim')
axC.set_xlabel('RA (deg)'); axC.set_ylabel('Dec (deg)')
axC.set_title('residual PM vectors (bulk removed)')
fig.colorbar(q, ax=axC, label='|Δpm| (mas/yr)', shrink=0.8)
# D: rescue - e_pm vs Ks with JW floor
axD = fig.add_subplot(2, 3, 4)
axD.scatter(tr['Ks'], epm, s=4, alpha=0.3, label='VIRAC e_pm (trustworthy)')
axD.axhline(JW_FLOOR, color='r', lw=1.5, label=f'JW-VVV floor {JW_FLOOR}')
axD.axhline(3, color='orange', ls='--', lw=1, label='VIRAC unusable (3)')
axD.set_xlabel('Ks (mag)'); axD.set_ylabel('VIRAC e_pm_tot (mas/yr)')
axD.set_ylim(0,15); axD.legend(fontsize=7); axD.set_title('precision rescue vs magnitude')
# E: cumulative usable-PM count vs threshold
axE = fig.add_subplot(2, 3, 5)
thr = np.linspace(0.5, 8, 60)
axE.plot(thr, [ (eall<t).sum() for t in thr ], label='VVV-only usable (e_pm<t)')
axE.axhline(n, color='seagreen', lw=1.5, label=f'JW-VVV trustworthy ({n}, ~{JW_FLOOR})')
axE.set_xlabel('PM error threshold (mas/yr)'); axE.set_ylabel('N stars usable')
axE.legend(fontsize=7); axE.set_title('usable-PM yield vs precision cut')
# F: text summary
axF = fig.add_subplot(2, 3, 6); axF.axis('off')
txt = (f'YIELD (F212N, VIRAC2 frame)\n'
       f'  VVV-only (any PM):      34,488\n'
       f'  VVV-only usable e_pm<2: {int((eall<2).sum()):,}\n'
       f'  JW-VVV trustworthy:     {n:,}\n\n'
       f'PRECISION RESCUE (of trustworthy)\n'
       f'  VIRAC e_pm>{JW_FLOOR}: {rescued.sum()} ({100*rescued.mean():.0f}%)\n'
       f'  VIRAC e_pm>3 (unusable): {strong.sum()} ({100*strong.mean():.0f}%)\n\n'
       f'DEPTH RESCUE (fainter than VIRAC2)\n'
       f'  VIRAC2 rolls over Ks~17;\n'
       f'  no position-only VVV catalog\n'
       f'  reachable (VSA down) -> minimal\n\n'
       f'JW-VVV floor {JW_FLOOR} mas/yr set by\n'
       f'VIRAC 2014 POSITION err (~3.6mas),\n'
       f'NOT its PM err -> flat vs Ks')
axF.text(0.02, 0.98, txt, va='top', ha='left', fontsize=9, family='monospace')
fig.suptitle('JW-VVV proper motions: VPD, spatial residuals, and the precision-rescue of faint VIRAC PMs', fontsize=12)
fig.tight_layout()
fig.savefig(f'{OUT}/pm_vpd_and_rescue.png', dpi=130, bbox_inches='tight')
print('wrote', f'{OUT}/pm_vpd_and_rescue.png')
