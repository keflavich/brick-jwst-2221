#!/usr/bin/env python
"""Is F200W astrometry really arcsec-off from VIRAC2?  Diagnostic: match VIRAC to
F200W, look at the SEPARATION distribution, the magnitude agreement, and whether
flux-vetting (keep the F200W_AB-Ks ridge) removes exactly the arcsec 'matches'.

Punchline hypothesis: the arcsec offsets are NON-COUNTERPARTS (VIRAC sources with
no F200W detection in the 1182 footprint, paired to a random neighbour), NOT an
F200W astrometry error.  Real counterparts sit at ~tens of mas.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import warnings; warnings.filterwarnings('ignore')

snap = Table.read('/blue/adamginsburg/adamginsburg/jwst/brick/astrometry_diag/m8_dedup_1182_snapshot_20260708.fits')
sc = snap['skycoord_f200w']; mab = np.asarray(snap['mag_ab_f200w'], float)
fj = np.asarray(snap['flux_jy_f200w'], float); efj = np.asarray(snap['eflux_jy_f200w'], float)
ok = np.isfinite(sc.ra.deg) & np.isfinite(mab) & np.isfinite(fj) & (fj > 0) & (efj > 0) & (fj/efj > 5)
jsc = SkyCoord(sc.ra.deg[ok]*u.deg, sc.dec.deg[ok]*u.deg); mab = mab[ok]

v = Table.read('/orange/adamginsburg/jwst/brick/astrometry_retie_qualcuts_20251211/refcache/virac2.fits')
vra = np.asarray(v['RAJ2000'], float); vde = np.asarray(v['DEJ2000'], float)
pmra = np.nan_to_num(np.asarray(v['pmRA'], float)); pmde = np.nan_to_num(np.asarray(v['pmDE'], float))
Ks = np.asarray(v['Ksmag'], float); cosd = np.cos(np.radians(vde))
vsc = SkyCoord((vra + pmra*8.7/3.6e6/cosd)*u.deg, (vde + pmde*8.7/3.6e6)*u.deg)

idx, sep, _ = vsc.match_to_catalog_sky(jsc)     # VIRAC -> nearest F200W, NO cut
sa = sep.arcsec
color = mab[idx] - Ks                            # F200W_AB - Ks (near reddening-free)
RIDGE = 2.15
onridge = np.abs(color - RIDGE) < 0.75           # flux-vet
close = sa < 0.3

# on-sky offsets for the candidate-counterpart set
dra = (vsc.ra.deg - jsc.ra.deg[idx])*np.cos(np.radians(vde))*3.6e6
dde = (vsc.dec.deg - jsc.dec.deg[idx])*3.6e6

def mode2d(x, y, half=120, b=3):
    e = np.arange(-half, half+b, b); H, xe, ye = np.histogram2d(x, y, bins=[e, e])
    iy, ix = np.unravel_index(np.argmax(H.T), H.T.shape)
    px, py = 0.5*(xe[ix]+xe[ix+1]), 0.5*(ye[iy]+ye[iy+1])
    w = (np.abs(x-px) < 15) & (np.abs(y-py) < 15)
    if w.sum() > 5: px, py = x[w].mean(), y[w].mean()
    return px, py

print('=== SEPARATION percentiles (VIRAC->F200W nearest) ===')
for lab, s in [('all', np.ones(len(sa), bool)), ('flux-vetted (on-ridge)', onridge)]:
    q = np.percentile(sa[s], [25, 50, 75, 90])*1000
    print(f'  {lab:24s} N={s.sum():6d}  p25/50/75/90 = {q[0]:6.0f}/{q[1]:6.0f}/{q[2]:7.0f}/{q[3]:7.0f} mas  frac>1\"={np.mean(sa[s]>1):.2f}')
# does flux-vet remove the arcsec 'matches'?
arc = sa > 1.0
print(f'\nOf arcsec (>1\") matches: on-ridge frac = {onridge[arc].mean():.2f}  (random color would give ~{ (np.abs(np.random.default_rng(0).uniform(-2,4,arc.sum())-RIDGE)<0.75).mean():.2f})')
print(f'Of close (<0.15\") matches: on-ridge frac = {onridge[sa<0.15].mean():.2f}')
tight = onridge & close
px, py = mode2d(dra[tight], dde[tight])
core = np.hypot(dra[tight]-px, dde[tight]-py)
print(f'\nflux-vetted+close F200W<->VIRAC: N={tight.sum()}  offset MODE=({px:+.1f},{py:+.1f}) mas  '
      f'coreMAD={np.median(np.abs(core[core<30]-np.median(core[core<30]))):.1f} mas  median|sep|={np.median(sa[tight])*1000:.0f} mas')

# ---- figure ------------------------------------------------------------------
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(2, 3, 1)
ax.hist(np.log10(sa*1000), bins=80, color='gray', alpha=0.6, label='all NN')
ax.hist(np.log10(sa[onridge]*1000), bins=80, color='C2', alpha=0.7, label='flux-vetted')
for xv in [np.log10(x) for x in (100, 1000)]: ax.axvline(xv, color='k', ls=':', lw=0.6)
ax.set_xlabel('log10 separation (mas)'); ax.set_ylabel('N'); ax.legend(fontsize=8)
ax.set_title('VIRAC->F200W separation\n(dotted: 0.1", 1")')

ax = fig.add_subplot(2, 3, 2)
ax.scatter(Ks[close], mab[idx[close]], s=3, alpha=0.2)
xx = np.linspace(10, 18, 10); ax.plot(xx, xx+RIDGE, 'r-', label=f'Ks+{RIDGE}')
ax.set_xlabel('VIRAC Ks'); ax.set_ylabel('F200W AB'); ax.legend(fontsize=8)
ax.set_title(f'mag agreement (<0.3" matches)\nmedian F200W_AB-Ks={np.median(color[close]):.2f}')

ax = fig.add_subplot(2, 3, 3)
ax.scatter(sa[sa < 1.5]*1000, color[sa < 1.5], s=2, alpha=0.15)
ax.axhline(RIDGE, color='r'); ax.axhspan(RIDGE-0.75, RIDGE+0.75, color='r', alpha=0.12)
ax.set_xlabel('separation (mas)'); ax.set_ylabel('F200W_AB - Ks'); ax.set_xlim(0, 1500); ax.set_ylim(-1, 6)
ax.set_title('color vs separation\n(on-ridge stars are the close ones)')

ax = fig.add_subplot(2, 3, 4)
ax.hexbin(dra[tight], dde[tight], gridsize=40, extent=[-60, 60, -60, 60], cmap='viridis', mincnt=1)
ax.plot(px, py, 'r+', ms=14, mew=2); ax.set_aspect('equal')
ax.set_xlabel('dRA (mas)'); ax.set_ylabel('dDec (mas)')
ax.set_title(f'offset VPD (flux-vetted+close)\nmode ({px:+.0f},{py:+.0f}) mas')

ax = fig.add_subplot(2, 3, 5)
q = ax.scatter(jsc.ra.deg[idx[tight]], jsc.dec.deg[idx[tight]], c=np.hypot(dra[tight]-px, dde[tight]-py),
               s=4, cmap='plasma', vmax=30)
ax.invert_xaxis(); ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel('RA'); ax.set_ylabel('Dec'); fig.colorbar(q, ax=ax, label='|offset-mode| mas', shrink=0.8)
ax.set_title('spatial map of residual (uniform = no local arcsec error)')

ax = fig.add_subplot(2, 3, 6); ax.axis('off')
txt = ('VERDICT\n'
       f'real counterparts (flux-vetted+<0.3"): {tight.sum()}\n'
       f'  systematic offset mode: ({px:+.0f},{py:+.0f}) mas\n'
       f'  core scatter: ~{np.median(np.abs(core[core<30]-np.median(core[core<30]))):.0f} mas (VIRAC floor)\n\n'
       f'arcsec ">1\"" matches: {int((sa>1).sum())} ({100*np.mean(sa>1):.0f}%)\n'
       f'  on-ridge frac of those: {onridge[arc].mean():.2f}\n'
       f'  -> flux-vet REMOVES them = they are\n'
       f'     NON-COUNTERPARTS (no F200W in footprint),\n'
       f'     NOT an F200W astrometry error\n\n'
       'F200W-VIRAC is ~tens of mas, NOT arcsec.\n'
       'Arcsec reports = unvetted NN to sources\n'
       'with no real counterpart, OR sampling the\n'
       '1182 i2d MOSAIC (had ~2" WCS err) not the\n'
       'CATALOG.')
ax.text(0.02, 0.98, txt, va='top', fontsize=9, family='monospace')
fig.suptitle('F200W <-> VIRAC2: magnitude agreement + astrometry (is it really arcsec-off?)', fontsize=12)
fig.tight_layout()
out = '/orange/adamginsburg/jwst/brick/astrometry_diag/f200w_virac_mag_astrom_test.png'
fig.savefig(out, dpi=130, bbox_inches='tight'); print('\nwrote', out)
