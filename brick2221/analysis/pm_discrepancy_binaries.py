#!/usr/bin/env python
"""Sources where the VIRAC catalog PM and the JWST-VIRAC PM disagree most, and a
test of whether the disagreement is a VVV blend/binary.

Two independent PM estimates per star:
  PM_VIRAC       = (pmRA, pmDE) from the VIRAC2 catalog (VVV-internal, ~2010-2015)
  PM_JWST-VIRAC  = (JWST_F200W_pos@2022.70 - VIRAC_pos@2014.0) / 8.70 yr
Discrepancy = |PM_JWST-VIRAC - PM_VIRAC|.  A VVV blend corrupts the VIRAC centroid
-> spurious PM_VIRAC; JWST resolves the pair.  So rank by discrepancy, flag
multiplicity (>=2 JWST F200W sources within one VVV PSF), and cut out both surveys.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u

BASE = '/orange/adamginsburg/jwst/brick'
OUTDIR = f'{BASE}/astrometry_diag/pm_binaries'
os.makedirs(OUTDIR, exist_ok=True)

VIRAC_EPOCH = 2014.0
JWST_EPOCH = 2022.70
DT = JWST_EPOCH - VIRAC_EPOCH          # 8.70 yr
VVV_PSF = 0.9                          # arcsec, VVV seeing scale for blend test

SNAP = f'{BASE}/astrometry_diag/m8_dedup_1182_snapshot_20260708.fits'
VIRAC = f'{BASE}/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'
VVV = '/orange/adamginsburg/jwst/vvv/vvv_dr4_Hband_GC_mosaic_0p25.fits'
F200 = f'{BASE}/F200W/pipeline/jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits'

# ---- JWST F200W sources ------------------------------------------------------
snap = Table.read(SNAP)
scj = snap['skycoord_f200w']
fj = np.asarray(snap['flux_jy_f200w'], float)
efj = np.asarray(snap['eflux_jy_f200w'], float)
ra_j, dec_j = scj.ra.deg, scj.dec.deg
okj = np.isfinite(ra_j) & np.isfinite(dec_j) & np.isfinite(fj) & (fj > 0) & np.isfinite(efj) & (efj > 0)
j = SkyCoord(ra_j[okj]*u.deg, dec_j[okj]*u.deg)
fj_ok = fj[okj]; snr_ok = fj_ok/efj[okj]
mag_f200_ab = -2.5*np.log10(fj_ok/3631.0)
print(f'JWST F200W usable: {okj.sum():,}')

# ---- VIRAC2 native 2014, propagate to JWST epoch -----------------------------
vt = Table.read(VIRAC)
vra = np.asarray(vt['RAJ2000'], float); vde = np.asarray(vt['DEJ2000'], float)
pmRA = np.asarray(vt['pmRA'], float); pmDE = np.asarray(vt['pmDE'], float)   # mas/yr, pmRA=mu_a*cosd
epmRA = np.asarray(vt['e_pmRA'], float); epmDE = np.asarray(vt['e_pmDE'], float)
Ks = np.asarray(vt['Ksmag'], float)
cosd = np.cos(np.radians(vde))
vra_p = vra + (pmRA/cosd)*DT/3.6e6       # deg
vde_p = vde + pmDE*DT/3.6e6
okv = np.isfinite(vra_p) & np.isfinite(vde_p) & np.isfinite(pmRA) & np.isfinite(pmDE)
vp = SkyCoord(vra_p[okv]*u.deg, vde_p[okv]*u.deg)       # VIRAC propagated to 2022.70
v2014 = SkyCoord(vra[okv]*u.deg, vde[okv]*u.deg)
pmRA, pmDE, epmRA, epmDE, Ks = pmRA[okv], pmDE[okv], epmRA[okv], epmDE[okv], Ks[okv]
print(f'VIRAC2 with PM: {okv.sum():,}')

# ---- match VIRAC(propagated) -> nearest JWST F200W ---------------------------
idx, sep, _ = vp.match_to_catalog_sky(j)
jm = j[idx]
# JWST-VIRAC PM from native 2014 endpoint to matched JWST 2022.70 position
dra_pm = (jm.ra.deg - vra[okv]) * cosd[okv] * 3.6e6 / DT     # mas/yr on-sky
dde_pm = (jm.dec.deg - vde[okv]) * 3.6e6 / DT
# frame tie: median over confident matches (tight, bright)
tie_ok = (sep.arcsec < 0.12) & (snr_ok[idx] > 20)
tie_ra = np.median(dra_pm[tie_ok] - pmRA[tie_ok])
tie_de = np.median(dde_pm[tie_ok] - pmDE[tie_ok])
print(f'frame tie (PM units): ({tie_ra:+.2f},{tie_de:+.2f}) mas/yr from {tie_ok.sum()} bright tight')
disc_ra = (dra_pm - tie_ra) - pmRA
disc_de = (dde_pm - tie_de) - pmDE
disc = np.hypot(disc_ra, disc_de)              # mas/yr
vpm = np.hypot(pmRA, pmDE)

# ---- find CLEAN ISOLATED DOUBLES ---------------------------------------------
# The Brick is confusion-limited: nearest-neighbour-in-a-crowd is meaningless.
# A convincing "VVV binary" = exactly 2 comparable JWST F200W stars inside the VVV
# beam and NOTHING else nearby, so VVV sees one blob and VIRAC centroids the pair.
from astropy.coordinates import search_around_sky
R_IN, R_OUT = 0.5, 1.0         # arcsec: pair radius, contamination radius
vi, ji, d2d, _ = search_around_sky(vp, j, R_OUT*u.arcsec)   # vi->vp, ji->j
sepas = d2d.arcsec
from collections import defaultdict
grp = defaultdict(list)
for a, b, s in zip(vi, ji, sepas):
    grp[a].append((s, b))

snr_m = snr_ok[idx]
cand_rows = []
for a, lst in grp.items():
    inr = sorted([(fj_ok[b], s, b) for s, b in lst if s < R_IN], reverse=True)  # bright->faint
    if len(inr) < 2:                           # need a pair inside the VVV beam
        continue
    (f1, s1, b1), (f2, s2, b2) = inr[0], inr[1]
    ratio = f1 / f2
    if ratio > 3:                              # two brightest comparable
        continue
    if snr_ok[b1] < 15 or snr_ok[b2] < 15:
        continue
    # any OTHER source (in R_OUT) must be >=1.5 mag fainter than the 2nd -> negligible for the blob
    others = [fl for fl, s, b in inr[2:]] + [fj_ok[b] for s, b in lst if R_IN <= s < R_OUT]
    if others and max(others) > f2/4.0:        # a 3rd comparable source -> messy, skip
        continue
    compsep = j[b1].separation(j[b2]).arcsec
    if not (0.12 < compsep < R_IN):
        continue
    # flux-weighted JWST centroid (what VVV/VIRAC would centroid on)
    w = f1 + f2
    cra = (j[b1].ra.deg*f1 + j[b2].ra.deg*f2)/w
    cde = (j[b1].dec.deg*f1 + j[b2].dec.deg*f2)/w
    jpmRA = (cra - vra[okv][a])*cosd[okv][a]*3.6e6/DT - tie_ra
    jpmDE = (cde - vde[okv][a])*3.6e6/DT - tie_de
    d = np.hypot(jpmRA - pmRA[a], jpmDE - pmDE[a])
    cand_rows.append(dict(a=a, disc=d, compsep=compsep, ratio=ratio,
                          jpmRA=jpmRA, jpmDE=jpmDE, b1=b1, b2=b2,
                          f1=f1, f2=f2))
print(f'\ndominant comparable pairs (2 bright in {R_IN}", ratio<3, no 3rd >f2/4 within '
      f'{R_OUT}", both SNR>15): {len(cand_rows):,}')
cand_rows.sort(key=lambda z: z['disc'], reverse=True)
if not cand_rows:
    import sys; print('no candidates -> nothing to plot'); sys.exit(0)

topN = 12
top = cand_rows[:topN]
print('\n  #  disc  vpm  jwstpm  compsep ratio  Ks   f200_1 f200_2   RA         Dec')
rows = []
for r, cr in enumerate(top):
    k = cr['a']; c = vp[k]
    jpm = np.hypot(cr['jpmRA'], cr['jpmDE'])
    m1 = -2.5*np.log10(cr['f1']/3631.); m2 = -2.5*np.log10(cr['f2']/3631.)
    print(f'{r:3d} {cr["disc"]:5.1f} {vpm[k]:5.1f} {jpm:6.1f} {cr["compsep"]:6.2f} {cr["ratio"]:5.1f} '
          f'{Ks[k]:5.1f} {m1:6.2f} {m2:6.2f}  {c.ra.deg:.6f} {c.dec.deg:.6f}')
    rows.append(dict(ra=c.ra.deg, dec=c.dec.deg, ra2014=vra[okv][k], dec2014=vde[okv][k],
                     disc=float(cr['disc']), vpm=float(vpm[k]), jwstpm=float(jpm),
                     compsep=float(cr['compsep']), ratio=float(cr['ratio']), Ks=float(Ks[k]),
                     f200w_ab_1=float(m1), f200w_ab_2=float(m2),
                     pmRA=float(pmRA[k]), pmDE=float(pmDE[k]),
                     jpmRA=float(cr['jpmRA']), jpmDE=float(cr['jpmDE'])))
if rows:
    Table(rows).write(f'{OUTDIR}/pm_discrepancy_top.fits', overwrite=True)

# ---- cutouts -----------------------------------------------------------------
vvv_h = fits.open(VVV, memmap=True); vvv_data = vvv_h[0].data; vvv_wcs = WCS(vvv_h[0].header)
f2_h = fits.open(F200, memmap=True); f2_data = f2_h[1].data; f2_wcs = WCS(f2_h[1].header)
CUT = 2.5*u.arcsec
PMSCALE = 0.05                 # arcsec drawn per (mas/yr) for PM arrows

nrow = len(top)
fig, axes = plt.subplots(nrow, 2, figsize=(7.0, 3.4*max(nrow,1)), squeeze=False)
for r, cr in enumerate(top):
    k = cr['a']; c = vp[k]; c14 = v2014[k]
    b1, b2 = cr['b1'], cr['b2']
    for col, (data, wcs, name, pxsc) in enumerate(
            [(vvv_data, vvv_wcs, 'VVV H (1.6um)', 0.25),
             (f2_data, f2_wcs, 'JWST F200W (2.0um)', 0.031)]):
        ax = axes[r][col]
        try:
            cut = Cutout2D(data, c, CUT, wcs=wcs)
        except ValueError:
            ax.text(0.5,0.5,'no coverage',ha='center',transform=ax.transAxes,fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); continue
        img = cut.data.astype(float); fin = np.isfinite(img) & (img != 0)
        lo, hi = (np.nanpercentile(img[fin],[3,99.7]) if fin.sum()>10 else (np.nanmin(img),np.nanmax(img)))
        ny, nx = img.shape
        ax.imshow(img, origin='lower', cmap='gray', vmin=lo, vmax=hi)
        ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)   # lock to cutout
        cw = cut.wcs
        # VIRAC epochs
        vx, vy = cw.world_to_pixel(c);   ax.plot(vx, vy, '+', color='cyan', ms=13, mew=1.8)
        v14x, v14y = cw.world_to_pixel(c14); ax.plot(v14x, v14y, 'x', color='yellow', ms=8, mew=1.5)
        # the two JWST pair components (red circles), other JWST dets (small orange)
        near = j[j.separation(c) < 1.6*CUT]
        for s in near:
            sx, sy = cw.world_to_pixel(s)
            ax.plot(sx, sy, 'o', mfc='none', mec='orange', ms=5, mew=0.8)
        for b, lab in [(b1,'1'), (b2,'2')]:
            sx, sy = cw.world_to_pixel(j[b])
            ax.plot(sx, sy, 'o', mfc='none', mec='red', ms=9, mew=1.6)
        # PM vectors from VIRAC@2022.7: VIRAC catalog (magenta), JWST-VIRAC (lime)
        for (pra, pde, cl) in [(pmRA[k], pmDE[k], 'magenta'),
                               (cr['jpmRA'], cr['jpmDE'], 'lime')]:
            dxpix = -(pra*PMSCALE)/pxsc            # +RA = -x (E left)
            dypix = (pde*PMSCALE)/pxsc
            ax.arrow(vx, vy, dxpix, dypix, color=cl, width=0.2/pxsc*0.02,
                     head_width=0.06/pxsc, length_includes_head=True, alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=8)
        if col == 0:
            ax.set_ylabel(f'#{r}  disc={cr["disc"]:.0f} mas/yr\nVIRACpm={vpm[k]:.0f}  sep={cr["compsep"]:.2f}"\n'
                          f'Ks={Ks[k]:.1f}  rat={cr["ratio"]:.1f}', fontsize=7.5)
fig.suptitle('PM-discrepancy blends (2.5" cutouts)\ncyan+ VIRAC@2022.7  yellow x VIRAC@2014  '
             'red o = JWST pair  orange o = other JWST\nmagenta arrow = VIRAC catalog PM   lime arrow = JWST-VIRAC PM',
             fontsize=9, y=1.001)
fig.tight_layout()
out = f'{OUTDIR}/pm_discrepancy_cutouts.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
print('\nwrote', out)
