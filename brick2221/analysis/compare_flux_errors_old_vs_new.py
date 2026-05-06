"""
Compare flux measurement uncertainties between the old Brick catalog
(20251211, errors not properly passed to photometry code) and the new
per-filter catalogs produced 2026-04-23 (errors correctly propagated).

Old catalog: basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20251211.fits
New catalogs: f{filt}_merged_indivexp_merged_dao_basic.fits  (one per filter)

Quality cuts applied identically to both:
  qfit < 0.4  AND  cfit < 0.1  AND  not near/is_saturated
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

CATDIR = Path('/blue/adamginsburg/adamginsburg/jwst/brick/catalogs')
OUTDIR = Path('/blue/adamginsburg/adamginsburg/jwst/brick/catalog_comparison_diagnostics')
OUTDIR.mkdir(exist_ok=True)

OLD_CAT = CATDIR / 'basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20251211.fits'

# Filters present in both old and new
FILTERS = ['f182m', 'f187n', 'f212n', 'f405n', 'f410m', 'f444w', 'f466n']

QFIT_CUT  = 0.4
CFIT_CUT  = 0.1


def quality_mask_old(d, filt):
    qfit = d[f'qfit_{filt}']
    cfit = d[f'cfit_{filt}']
    sat  = d[f'near_saturated_{filt}_{filt}']
    flux = d[f'flux_{filt}']
    return (qfit < QFIT_CUT) & (cfit < CFIT_CUT) & (~sat.astype(bool)) & (flux > 0)


def quality_mask_new(d):
    qfit = d['qfit']
    cfit = d['cfit']
    sat  = d['is_saturated']
    flux = d['flux']
    return (qfit < QFIT_CUT) & (cfit < CFIT_CUT) & (~sat.astype(bool)) & (flux > 0)


# ── Load old catalog once ──────────────────────────────────────────────────
print("Loading old catalog …")
old_hdul = fits.open(OLD_CAT)
old_data = old_hdul[1].data
print(f"  {len(old_data)} rows in old catalog")

# ── Per-filter statistics ──────────────────────────────────────────────────
stats = {}

for filt in FILTERS:
    new_path = CATDIR / f'{filt}_merged_indivexp_merged_dao_basic.fits'
    if not new_path.exists():
        print(f"  {filt}: new catalog not found, skipping")
        continue

    # --- Old ---
    om = quality_mask_old(old_data, filt)
    o_flux = old_data[f'flux_{filt}'][om].astype(float)
    o_err  = old_data[f'flux_err_{filt}'][om].astype(float)
    o_errp = old_data[f'flux_err_prop_{filt}'][om].astype(float)
    # fractional error  (signal-to-noise proxy)
    o_snr  = o_flux / o_err
    o_frac = o_err  / o_flux

    # --- New ---
    nh = fits.open(new_path)
    nd = nh[1].data
    nm = quality_mask_new(nd)
    n_flux = nd['flux'][nm].astype(float)
    n_err  = nd['flux_err'][nm].astype(float)
    n_errp = nd['flux_err_prop'][nm].astype(float)
    n_snr  = n_flux / n_err
    n_frac = n_err  / n_flux
    nh.close()

    stats[filt] = dict(
        o_flux=o_flux, o_err=o_err, o_errp=o_errp, o_snr=o_snr, o_frac=o_frac,
        n_flux=n_flux, n_err=n_err, n_errp=n_errp, n_snr=n_snr, n_frac=n_frac,
        n_old=om.sum(), n_new=nm.sum(),
    )
    print(f"\n{filt.upper()}")
    print(f"  Old: N={om.sum():6d}  median flux_err={np.median(o_err):.3g}  median frac_err={np.median(o_frac):.3f}  median SNR={np.median(o_snr):.1f}")
    print(f"  New: N={nm.sum():6d}  median flux_err={np.median(n_err):.3g}  median frac_err={np.median(n_frac):.3f}  median SNR={np.median(n_snr):.1f}")
    ratio = np.median(n_err) / np.median(o_err)
    print(f"  New/Old median err ratio: {ratio:.3f}  ({'↑ larger' if ratio>1 else '↓ smaller'})")

old_hdul.close()

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Summary table: median flux_err old vs new per filter
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

filts_avail = [f for f in FILTERS if f in stats]
x = np.arange(len(filts_avail))
width = 0.35

# Left panel: median absolute flux_err
ax = axes[0]
med_old = [np.median(stats[f]['o_err']) for f in filts_avail]
med_new = [np.median(stats[f]['n_err']) for f in filts_avail]
ax.bar(x - width/2, med_old, width, label='Old (20251211)', color='steelblue', alpha=0.8)
ax.bar(x + width/2, med_new, width, label='New (2026-04-23)', color='darkorange', alpha=0.8)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels([f.upper() for f in filts_avail], rotation=25, ha='right')
ax.set_ylabel('Median flux_err  [counts]')
ax.set_title('Absolute flux uncertainty (quality-cut sources)')
ax.legend()

# Right panel: median fractional flux_err = err/flux
ax = axes[1]
med_old_frac = [np.median(stats[f]['o_frac']) for f in filts_avail]
med_new_frac = [np.median(stats[f]['n_frac']) for f in filts_avail]
ax.bar(x - width/2, med_old_frac, width, label='Old (20251211)', color='steelblue', alpha=0.8)
ax.bar(x + width/2, med_new_frac, width, label='New (2026-04-23)', color='darkorange', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f.upper() for f in filts_avail], rotation=25, ha='right')
ax.set_ylabel('Median fractional flux_err  (err/flux)')
ax.set_title('Fractional flux uncertainty (quality-cut sources)')
ax.legend()

fig.suptitle('Old vs New Brick catalog: flux measurement uncertainties', fontsize=13, y=1.01)
fig.tight_layout()
out1 = OUTDIR / 'err_summary_barplot.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {out1}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Per-filter: ratio new_err/old_err  vs  log10(flux)
# ══════════════════════════════════════════════════════════════════════════
ncols = 4
nrows = int(np.ceil(len(filts_avail) / ncols))
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for idx, filt in enumerate(filts_avail):
    ax = axes2[idx // ncols][idx % ncols]
    s = stats[filt]

    # We can't do a direct source-by-source match because RA/Dec matching
    # is needed. Instead show the distribution of err/flux in SNR bins.
    bins = np.logspace(np.log10(max(s['o_frac'].min(), 1e-4)),
                       np.log10(min(s['o_frac'].max(), 10)), 80)
    ax.hist(s['o_frac'], bins=bins, histtype='step', color='steelblue',
            label=f"Old  med={np.median(s['o_frac']):.3f}", density=True, lw=1.5)
    ax.hist(s['n_frac'], bins=bins, histtype='step', color='darkorange',
            label=f"New  med={np.median(s['n_frac']):.3f}", density=True, lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('flux_err / flux', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(filt.upper(), fontsize=10)
    ax.legend(fontsize=7)

# Hide unused panels
for idx in range(len(filts_avail), nrows * ncols):
    axes2[idx // ncols][idx % ncols].set_visible(False)

fig2.suptitle('Fractional flux uncertainty distribution: Old vs New', fontsize=12)
fig2.tight_layout()
out2 = OUTDIR / 'frac_err_distribution_per_filter.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved {out2}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Per-filter: absolute flux_err histograms
# ══════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for idx, filt in enumerate(filts_avail):
    ax = axes3[idx // ncols][idx % ncols]
    s = stats[filt]

    all_errs = np.concatenate([s['o_err'], s['n_err']])
    lo, hi = np.percentile(all_errs[np.isfinite(all_errs)], [1, 99])
    bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 80)

    ax.hist(s['o_err'], bins=bins, histtype='step', color='steelblue',
            label=f"Old  med={np.median(s['o_err']):.2g}", density=True, lw=1.5)
    ax.hist(s['n_err'], bins=bins, histtype='step', color='darkorange',
            label=f"New  med={np.median(s['n_err']):.2g}", density=True, lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('flux_err  [counts]', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(filt.upper(), fontsize=10)
    ax.legend(fontsize=7)

for idx in range(len(filts_avail), nrows * ncols):
    axes3[idx // ncols][idx % ncols].set_visible(False)

fig3.suptitle('Absolute flux_err distribution: Old vs New', fontsize=12)
fig3.tight_layout()
out3 = OUTDIR / 'abs_err_distribution_per_filter.png'
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"Saved {out3}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 – SNR distributions (old vs new)
# ══════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for idx, filt in enumerate(filts_avail):
    ax = axes4[idx // ncols][idx % ncols]
    s = stats[filt]

    snr_max = np.percentile(np.concatenate([s['o_snr'], s['n_snr']]), 99)
    bins = np.logspace(0, np.log10(snr_max + 1), 60)

    ax.hist(s['o_snr'], bins=bins, histtype='step', color='steelblue',
            label=f"Old  med={np.median(s['o_snr']):.1f}", density=True, lw=1.5)
    ax.hist(s['n_snr'], bins=bins, histtype='step', color='darkorange',
            label=f"New  med={np.median(s['n_snr']):.1f}", density=True, lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('SNR = flux / flux_err', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(filt.upper(), fontsize=10)
    ax.legend(fontsize=7)

for idx in range(len(filts_avail), nrows * ncols):
    axes4[idx // ncols][idx % ncols].set_visible(False)

fig4.suptitle('SNR distribution: Old vs New', fontsize=12)
fig4.tight_layout()
out4 = OUTDIR / 'snr_distribution_per_filter.png'
fig4.savefig(out4, dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"Saved {out4}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Per-filter: flux_err vs flux scatter (old left, new right)
# Show both old and new on same axes for one representative filter
# ══════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

rng = np.random.default_rng(42)

for idx, filt in enumerate(filts_avail):
    ax = axes5[idx // ncols][idx % ncols]
    s = stats[filt]

    # Subsample if large
    def subsample(arr, n=5000):
        if len(arr) > n:
            idx = rng.choice(len(arr), n, replace=False)
            return arr[idx]
        return arr

    o_f = subsample(s['o_flux'])
    o_e = subsample(s['o_err'])
    n_f = subsample(s['n_flux'])
    n_e = subsample(s['n_err'])

    ax.scatter(o_f, o_e, s=1.5, alpha=0.3, color='steelblue', label='Old', rasterized=True)
    ax.scatter(n_f, n_e, s=1.5, alpha=0.3, color='darkorange', label='New', rasterized=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('flux  [counts]', fontsize=9)
    ax.set_ylabel('flux_err  [counts]', fontsize=9)
    ax.set_title(filt.upper(), fontsize=10)
    ax.legend(fontsize=7, markerscale=4)

for idx in range(len(filts_avail), nrows * ncols):
    axes5[idx // ncols][idx % ncols].set_visible(False)

fig5.suptitle('flux_err vs flux: Old vs New (quality cuts applied, 5k subsample)', fontsize=11)
fig5.tight_layout()
out5 = OUTDIR / 'err_vs_flux_scatter.png'
fig5.savefig(out5, dpi=150, bbox_inches='tight')
plt.close(fig5)
print(f"Saved {out5}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 – New ratio: flux_err_prop / flux_err (within new catalog)
# Shows how propagated vs. scatter-based error compare in new version
# ══════════════════════════════════════════════════════════════════════════
fig6, axes6 = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False)

for idx, filt in enumerate(filts_avail):
    ax = axes6[idx // ncols][idx % ncols]
    s = stats[filt]

    # Old: flux_err_prop / flux_err
    o_ratio = s['o_errp'] / s['o_err']
    n_ratio = s['n_errp'] / s['n_err']

    ok_o = np.isfinite(o_ratio) & (o_ratio > 0)
    ok_n = np.isfinite(n_ratio) & (n_ratio > 0)

    bins = np.linspace(0, 5, 80)
    ax.hist(o_ratio[ok_o], bins=bins, histtype='step', color='steelblue',
            label=f"Old  med={np.median(o_ratio[ok_o]):.2f}", density=True, lw=1.5)
    ax.hist(n_ratio[ok_n], bins=bins, histtype='step', color='darkorange',
            label=f"New  med={np.median(n_ratio[ok_n]):.2f}", density=True, lw=1.5)
    ax.axvline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('flux_err_prop / flux_err', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(filt.upper(), fontsize=10)
    ax.legend(fontsize=7)

for idx in range(len(filts_avail), nrows * ncols):
    axes6[idx // ncols][idx % ncols].set_visible(False)

fig6.suptitle('flux_err_prop / flux_err ratio: propagated vs scatter (Old vs New)', fontsize=11)
fig6.tight_layout()
out6 = OUTDIR / 'errprop_vs_err_ratio.png'
fig6.savefig(out6, dpi=150, bbox_inches='tight')
plt.close(fig6)
print(f"Saved {out6}")

# ══════════════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print(f"{'Filter':<8} {'N_old':>8} {'N_new':>8} "
      f"{'med_err_old':>12} {'med_err_new':>12} {'new/old':>8} "
      f"{'med_frac_old':>13} {'med_frac_new':>13} {'med_SNR_old':>12} {'med_SNR_new':>12}")
print("-"*80)
for filt in filts_avail:
    s = stats[filt]
    ratio = np.median(s['n_err']) / np.median(s['o_err'])
    print(f"{filt.upper():<8} {s['n_old']:>8d} {s['n_new']:>8d} "
          f"{np.median(s['o_err']):>12.3g} {np.median(s['n_err']):>12.3g} {ratio:>8.3f} "
          f"{np.median(s['o_frac']):>13.4f} {np.median(s['n_frac']):>13.4f} "
          f"{np.median(s['o_snr']):>12.1f} {np.median(s['n_snr']):>12.1f}")
print("="*80)
print(f"\nPlots written to {OUTDIR}/")
