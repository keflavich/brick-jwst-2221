"""
Coordinate-matched comparison of flux_err (old vs new Brick catalog).
Matches each quality-passing source in the old 20251211 catalog to the
closest source in the new 2026-04-23 per-filter catalog within 0.15 arcsec,
then compares flux_err on a source-by-source basis.

This removes population-composition effects (old catalog is brighter on avg)
and gives a direct "did the error go up or down for the same star?" answer.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path

CATDIR = Path('/blue/adamginsburg/adamginsburg/jwst/brick/catalogs')
OUTDIR = Path('/blue/adamginsburg/adamginsburg/jwst/brick/catalog_comparison_diagnostics')
OUTDIR.mkdir(exist_ok=True)

OLD_CAT = CATDIR / 'basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20251211.fits'
MATCH_RADIUS = 0.15  # arcsec
QFIT_CUT, CFIT_CUT = 0.4, 0.1

FILTERS = ['f182m', 'f187n', 'f212n', 'f405n', 'f410m', 'f444w', 'f466n']

print("Loading old catalog …")
old_data = fits.open(OLD_CAT)[1].data
print(f"  {len(old_data)} rows")

matched_stats = {}

for filt in FILTERS:
    new_path = CATDIR / f'{filt}_merged_indivexp_merged_dao_basic.fits'
    if not new_path.exists():
        continue

    nd = fits.open(new_path)[1].data

    # --- Old quality mask ---
    om = (
        (old_data[f'qfit_{filt}'] < QFIT_CUT) &
        (old_data[f'cfit_{filt}'] < CFIT_CUT) &
        (~old_data[f'near_saturated_{filt}_{filt}'].astype(bool)) &
        (old_data[f'flux_{filt}'] > 0)
    )
    # --- New quality mask ---
    nm = (
        (nd['qfit'] < QFIT_CUT) &
        (nd['cfit'] < CFIT_CUT) &
        (~nd['is_saturated'].astype(bool)) &
        (nd['flux'] > 0)
    )

    osc = SkyCoord(
        ra=old_data[f'skycoord_{filt}.ra'][om] * u.deg,
        dec=old_data[f'skycoord_{filt}.dec'][om] * u.deg,
    )
    nsc = SkyCoord(
        ra=nd['skycoord.ra'][nm] * u.deg,
        dec=nd['skycoord.dec'][nm] * u.deg,
    )

    idx, sep, _ = osc.match_to_catalog_sky(nsc)
    matched = sep < MATCH_RADIUS * u.arcsec

    n_match = matched.sum()
    frac = matched.mean() * 100
    print(f"\n{filt.upper()}: {n_match} matched ({frac:.1f}%)")

    o_flux = old_data[f'flux_{filt}'][om][matched].astype(float)
    o_err  = old_data[f'flux_err_{filt}'][om][matched].astype(float)
    o_errp = old_data[f'flux_err_prop_{filt}'][om][matched].astype(float)
    n_flux = nd['flux'][nm][idx[matched]].astype(float)
    n_err  = nd['flux_err'][nm][idx[matched]].astype(float)
    n_errp = nd['flux_err_prop'][nm][idx[matched]].astype(float)

    err_ratio  = n_err  / o_err
    errp_ratio = n_errp / o_errp

    ok = np.isfinite(err_ratio) & (err_ratio > 0)
    okp = np.isfinite(errp_ratio) & (errp_ratio > 0)

    matched_stats[filt] = dict(
        n_match=n_match, frac=frac,
        o_flux=o_flux, o_err=o_err, o_errp=o_errp,
        n_flux=n_flux, n_err=n_err, n_errp=n_errp,
        err_ratio=err_ratio, errp_ratio=errp_ratio,
    )

    r = err_ratio[ok]
    print(f"  flux_err  new/old: p16={np.percentile(r,16):.3f}  p50={np.percentile(r,50):.3f}  p84={np.percentile(r,84):.3f}  mean={r.mean():.3f}")
    rp = errp_ratio[okp]
    print(f"  flux_errp new/old: p16={np.percentile(rp,16):.3f}  p50={np.percentile(rp,50):.3f}  p84={np.percentile(rp,84):.3f}  mean={rp.mean():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Matched: flux_err ratio histogram per filter
# ══════════════════════════════════════════════════════════════════════════
filts_avail = [f for f in FILTERS if f in matched_stats]
ncols = 4
nrows = int(np.ceil(len(filts_avail) / ncols))

fig7, axes7 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for idx, filt in enumerate(filts_avail):
    ax = axes7[idx // ncols][idx % ncols]
    s = matched_stats[filt]
    r = s['err_ratio']
    ok = np.isfinite(r) & (r > 0)
    r = r[ok]

    bins = np.linspace(0, 3, 80)
    ax.hist(r, bins=bins, color='steelblue', alpha=0.7, density=True)
    ax.axvline(1.0, color='k', ls='--', lw=1.2, label='no change')
    med = np.median(r)
    ax.axvline(med, color='tomato', ls='-', lw=1.5, label=f'median={med:.2f}')
    ax.set_xlabel('flux_err(new) / flux_err(old)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'{filt.upper()} (N={s["n_match"]:,})', fontsize=10)
    ax.legend(fontsize=7)

for idx in range(len(filts_avail), nrows * ncols):
    axes7[idx // ncols][idx % ncols].set_visible(False)

fig7.suptitle('Matched-source flux_err ratio: new/old (same star, 0.15″ match)', fontsize=11)
fig7.tight_layout()
out7 = OUTDIR / 'matched_err_ratio_histogram.png'
fig7.savefig(out7, dpi=150, bbox_inches='tight')
plt.close(fig7)
print(f"\nSaved {out7}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Matched: flux_err ratio vs log(flux) (binned median + scatter)
# ══════════════════════════════════════════════════════════════════════════
fig8, axes8 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)
rng = np.random.default_rng(42)

for idx, filt in enumerate(filts_avail):
    ax = axes8[idx // ncols][idx % ncols]
    s = matched_stats[filt]
    r = s['err_ratio']
    flux = s['o_flux']
    ok = np.isfinite(r) & (r > 0) & (flux > 0)
    r, flux = r[ok], flux[ok]

    # Subsample for scatter plot
    nsub = min(3000, len(r))
    si = rng.choice(len(r), nsub, replace=False)
    ax.scatter(flux[si], r[si], s=1.5, alpha=0.3, color='steelblue', rasterized=True)

    # Binned median
    edges = np.percentile(np.log10(flux), np.linspace(2, 98, 15))
    mids, meds, p16s, p84s = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (np.log10(flux) >= lo) & (np.log10(flux) < hi)
        if in_bin.sum() > 10:
            mids.append(10**((lo+hi)/2))
            meds.append(np.median(r[in_bin]))
            p16s.append(np.percentile(r[in_bin], 16))
            p84s.append(np.percentile(r[in_bin], 84))
    if mids:
        mids, meds, p16s, p84s = map(np.array, (mids, meds, p16s, p84s))
        ax.plot(mids, meds, 'tomato', lw=1.5, label='median')
        ax.fill_between(mids, p16s, p84s, alpha=0.25, color='tomato', label='p16–p84')

    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xscale('log')
    ax.set_ylim(0, 3)
    ax.set_xlabel('flux [counts] (old catalog)', fontsize=9)
    ax.set_ylabel('flux_err new/old', fontsize=9)
    ax.set_title(filt.upper(), fontsize=10)
    ax.legend(fontsize=7)

for idx in range(len(filts_avail), nrows * ncols):
    axes8[idx // ncols][idx % ncols].set_visible(False)

fig8.suptitle('Matched flux_err ratio vs source brightness (old flux)', fontsize=11)
fig8.tight_layout()
out8 = OUTDIR / 'matched_err_ratio_vs_flux.png'
fig8.savefig(out8, dpi=150, bbox_inches='tight')
plt.close(fig8)
print(f"Saved {out8}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Matched: flux_err_prop ratio histogram (same for propagated err)
# ══════════════════════════════════════════════════════════════════════════
fig9, axes9 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for idx, filt in enumerate(filts_avail):
    ax = axes9[idx // ncols][idx % ncols]
    s = matched_stats[filt]
    rp = s['errp_ratio']
    okp = np.isfinite(rp) & (rp > 0)
    rp = rp[okp]

    bins = np.linspace(0, 3, 80)
    ax.hist(rp, bins=bins, color='darkorange', alpha=0.7, density=True)
    ax.axvline(1.0, color='k', ls='--', lw=1.2, label='no change')
    med = np.median(rp)
    ax.axvline(med, color='steelblue', ls='-', lw=1.5, label=f'median={med:.2f}')
    ax.set_xlabel('flux_err_prop(new) / flux_err_prop(old)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'{filt.upper()} (N={s["n_match"]:,})', fontsize=10)
    ax.legend(fontsize=7)

for idx in range(len(filts_avail), nrows * ncols):
    axes9[idx // ncols][idx % ncols].set_visible(False)

fig9.suptitle('Matched-source flux_err_prop ratio: new/old (propagated error)', fontsize=11)
fig9.tight_layout()
out9 = OUTDIR / 'matched_errprop_ratio_histogram.png'
fig9.savefig(out9, dpi=150, bbox_inches='tight')
plt.close(fig9)
print(f"Saved {out9}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Compact summary: per-filter median ratio + 16/84 whiskers
# ══════════════════════════════════════════════════════════════════════════
fig10, axes10 = plt.subplots(1, 2, figsize=(12, 4))

x = np.arange(len(filts_avail))
for col_idx, (errkey, title, color) in enumerate([
    ('err_ratio', 'flux_err ratio (new/old)', 'steelblue'),
    ('errp_ratio', 'flux_err_prop ratio (new/old)', 'darkorange'),
]):
    ax = axes10[col_idx]
    meds, p16s, p84s = [], [], []
    for filt in filts_avail:
        r = matched_stats[filt][errkey]
        ok = np.isfinite(r) & (r > 0)
        r = r[ok]
        meds.append(np.median(r))
        p16s.append(np.percentile(r, 16))
        p84s.append(np.percentile(r, 84))

    meds, p16s, p84s = map(np.array, (meds, p16s, p84s))
    ax.bar(x, meds, color=color, alpha=0.75, width=0.6, label='median')
    ax.errorbar(x, meds,
                yerr=[meds - p16s, p84s - meds],
                fmt='none', color='k', capsize=4, lw=1.5, label='p16–p84')
    ax.axhline(1.0, color='k', ls='--', lw=0.8, label='no change')
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in filts_avail], rotation=25, ha='right')
    ax.set_ylabel('Ratio  new / old')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(p84s.max() * 1.15, 1.5))

fig10.suptitle('Per-filter matched error ratio (new / old)  —  quality-cut sources, 0.15″ match',
               fontsize=11)
fig10.tight_layout()
out10 = OUTDIR / 'matched_err_ratio_summary.png'
fig10.savefig(out10, dpi=150, bbox_inches='tight')
plt.close(fig10)
print(f"Saved {out10}")

# ══════════════════════════════════════════════════════════════════════════
# Print final summary table
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*90)
print(f"{'Filter':<8} {'N_match':>8} {'match%':>7} "
      f"{'err_ratio p16':>14} {'err_ratio p50':>14} {'err_ratio p84':>14} "
      f"{'errp_ratio p50':>15}")
print("-"*90)
for filt in filts_avail:
    s = matched_stats[filt]
    r  = s['err_ratio'];  ok  = np.isfinite(r)  & (r > 0);  r  = r[ok]
    rp = s['errp_ratio']; okp = np.isfinite(rp) & (rp > 0); rp = rp[okp]
    print(f"{filt.upper():<8} {s['n_match']:>8,} {s['frac']:>7.1f} "
          f"{np.percentile(r,16):>14.3f} {np.percentile(r,50):>14.3f} {np.percentile(r,84):>14.3f} "
          f"{np.percentile(rp,50):>15.3f}")
print("="*90)
print(f"\nAll matched plots written to {OUTDIR}/")
