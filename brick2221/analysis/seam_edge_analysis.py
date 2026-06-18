"""
Internal cross-filter seam / frame / edge diagnostics for the Brick, daophot-BASIC only.

Crowdsource catalogs are deprecated (2026-06); this uses *_merged_indivexp_merged_dao_basic.fits.

For a pair of filters we match bright, well-measured, flux-ratio-clean stars and look at the
position residual (dRA, dDec, great-circle, mas) as a function of sky position. A *seam* shows up
as a bulk offset that jumps across a sky boundary (the NIRCam A/B module mosaic boundary near
Dec -28.71); a smooth *frame* misalignment shows up as a per-region bulk offset; *edge* effects
need per-exposure detector coordinates (handled separately).

Programs: F115W/F200W/F356W/F444W = prop 1182 (obs004, 2022-09-14);
          F182M/F187N/F212N/F405N/F410M/F466N = prop 2221 (obs001, 2022-08-28).

Usage:
    python seam_edge_analysis.py PAIR [PAIR ...]      # e.g. f200w-f115w f182m-f212n f200w-f182m
    (no args -> a default set of pairs)
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import mad_std
import warnings
warnings.filterwarnings('ignore')

CATDIR = '/orange/adamginsburg/jwst/brick/catalogs'
OUTDIR = '/orange/adamginsburg/jwst/brick/astrometry_seam_dao_20251211'
os.makedirs(OUTDIR, exist_ok=True)

# module boundary (Dec, deg) seen in earlier analysis
SEAM_DEC = -28.709

PROG = {'f115w': 1182, 'f200w': 1182, 'f356w': 1182, 'f444w': 1182,
        'f182m': 2221, 'f187n': 2221, 'f212n': 2221, 'f405n': 2221,
        'f410m': 2221, 'f466n': 2221}


def load(filt):
    t = Table.read(f'{CATDIR}/{filt}_merged_indivexp_merged_dao_basic.fits')
    sc = SkyCoord(t['skycoord'])
    fluxcol = 'flux' if 'flux' in t.colnames else ('flux_fit' if 'flux_fit' in t.colnames else 'flux_init')
    flux = np.asarray(t[fluxcol], float)
    qfit = np.asarray(t['qfit'], float) if 'qfit' in t.colnames else np.zeros(len(t))
    # cross-frame scatter (deg) -> mas
    stdra = np.asarray(t['std_ra'], float) * 3600e3 if 'std_ra' in t.colnames else np.full(len(t), np.nan)
    stddec = np.asarray(t['std_dec'], float) * 3600e3 if 'std_dec' in t.colnames else np.full(len(t), np.nan)
    nmatch = np.asarray(t['nmatch'], float) if 'nmatch' in t.colnames else np.ones(len(t))
    sat = np.asarray(t['is_saturated'], bool) if 'is_saturated' in t.colnames else np.zeros(len(t), bool)
    sat = np.asarray(sat, bool)
    return t, sc, flux, qfit, stdra, stddec, nmatch, sat


def clean_match(fa, fb, max_offset=0.15 * u.arcsec):
    """Mutual NN match of two filters, bright + good-quality + flux-ratio clean."""
    ta, sca, fla, qa, sra, sda, na, sata = load(fa)
    tb, scb, flb, qb, srb, sdb, nb, satb = load(fb)

    # quality pre-selection: detected in >=3 frames, not saturated, finite flux, decent fit
    def good(fl, q, n, sat, sr):
        m = np.isfinite(fl) & (fl > 0) & (~sat) & (n >= 3) & np.isfinite(sr) & (sr < 5)
        if np.isfinite(q).any():
            m &= (q < 0.2)
        return m
    ga = good(fla, qa, na, sata, sra)
    gb = good(flb, qb, nb, satb, srb)
    sca_g, scb_g = sca[ga], scb[gb]
    fla_g, flb_g = fla[ga], flb[gb]

    idx, sep, _ = sca_g.match_to_catalog_sky(scb_g)
    ridx, _, _ = scb_g.match_to_catalog_sky(sca_g)
    mutual = (ridx[idx] == np.arange(len(idx)))
    keep = mutual & (sep < max_offset)

    a = sca_g[keep]
    b = scb_g[idx[keep]]
    fa_m, fb_m = fla_g[keep], flb_g[idx[keep]]

    # flux-ratio clean (log flux ratio sigma clip) to drop crowded false matches
    lr = np.log(fb_m) - np.log(fa_m)
    for _ in range(5):
        m = np.nanmedian(lr); s = mad_std(lr, ignore_nan=True)
        ok = np.abs(lr - m) < 3 * s
        if ok.all():
            break
        a, b, fa_m, fb_m, lr = a[ok], b[ok], fa_m[ok], fb_m[ok], lr[ok]

    dra = ((b.ra - a.ra) * np.cos(a.dec.radian)).to(u.mas).value
    ddec = (b.dec - a.dec).to(u.mas).value
    return a, dra, ddec


def analyze(pair):
    fa, fb = pair.split('-')
    a, dra, ddec = clean_match(fa, fb)
    ra = a.ra.deg; dec = a.dec.deg
    n = len(a)
    md_ra, md_dec = np.median(dra), np.median(ddec)
    s_ra, s_dec = mad_std(dra), mad_std(ddec)
    print(f"\n=== {pair}  ({PROG[fa]} vs {PROG[fb]})  N={n} ===")
    print(f"  global median dRA={md_ra:6.1f}  dDec={md_dec:6.1f} mas | MAD dRA={s_ra:5.1f} dDec={s_dec:5.1f}")

    # vs-Dec profile (the seam test)
    decbins = np.linspace(np.percentile(dec, 1), np.percentile(dec, 99), 40)
    dc = 0.5 * (decbins[:-1] + decbins[1:])
    prof_ra = np.array([np.median(dra[(dec >= lo) & (dec < hi)]) if ((dec >= lo) & (dec < hi)).sum() > 20 else np.nan
                        for lo, hi in zip(decbins[:-1], decbins[1:])])
    prof_dec = np.array([np.median(ddec[(dec >= lo) & (dec < hi)]) if ((dec >= lo) & (dec < hi)).sum() > 20 else np.nan
                         for lo, hi in zip(decbins[:-1], decbins[1:])])
    # seam amplitude: median below vs above SEAM_DEC, within +-0.01 deg
    near = np.abs(dec - SEAM_DEC) < 0.012
    below = near & (dec < SEAM_DEC); above = near & (dec > SEAM_DEC)
    if below.sum() > 30 and above.sum() > 30:
        jra = np.median(dra[above]) - np.median(dra[below])
        jdec = np.median(ddec[above]) - np.median(ddec[below])
        print(f"  SEAM jump across Dec={SEAM_DEC}: dRA {jra:+.1f}  dDec {jdec:+.1f} mas  (N below/above {below.sum()}/{above.sum()})")
    else:
        jra = jdec = np.nan
        print("  SEAM jump: too few stars near boundary")

    # 2D maps
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    nb = 50
    rb = np.linspace(np.percentile(ra, 0.5), np.percentile(ra, 99.5), nb)
    db = np.linspace(np.percentile(dec, 0.5), np.percentile(dec, 99.5), nb)
    from scipy.stats import binned_statistic_2d
    for ax, val, ttl in [(axs[0, 0], dra, f'median dRA (mas)'), (axs[0, 1], ddec, 'median dDec (mas)')]:
        stat, _, _, _ = binned_statistic_2d(ra, dec, val, statistic='median', bins=[rb, db])
        im = ax.imshow(stat.T, origin='lower', extent=[rb[0], rb[-1], db[0], db[-1]], aspect='auto',
                       cmap='RdBu_r', vmin=-25, vmax=25)
        ax.axhline(SEAM_DEC, color='k', ls=':', lw=1)
        ax.invert_xaxis(); ax.set_title(f'{pair}: {ttl}'); plt.colorbar(im, ax=ax)
    axs[1, 0].plot(dc, prof_ra, 'o-', label='dRA'); axs[1, 0].plot(dc, prof_dec, 's-', label='dDec')
    axs[1, 0].axvline(SEAM_DEC, color='k', ls=':'); axs[1, 0].axhline(0, color='gray', lw=0.5)
    axs[1, 0].set_xlabel('Dec'); axs[1, 0].set_ylabel('median residual (mas)'); axs[1, 0].legend()
    axs[1, 0].set_title(f'{pair}: residual vs Dec (seam at {SEAM_DEC})')
    axs[1, 1].hist2d(dra, ddec, bins=80, range=[[-60, 60], [-60, 60]], cmap='viridis')
    axs[1, 1].axvline(0, color='w', lw=0.5); axs[1, 1].axhline(0, color='w', lw=0.5)
    axs[1, 1].set_xlabel('dRA (mas)'); axs[1, 1].set_ylabel('dDec (mas)')
    axs[1, 1].set_title(f'{pair}: offset cloud (med {md_ra:.0f},{md_dec:.0f})')
    plt.tight_layout()
    out = f'{OUTDIR}/seam_{pair}.png'
    plt.savefig(out, dpi=100); plt.close()
    print(f"  saved {out}")
    return dict(pair=pair, n=n, med_ra=md_ra, med_dec=md_dec, mad_ra=s_ra, mad_dec=s_dec,
               seam_jra=jra, seam_jdec=jdec)


if __name__ == '__main__':
    pairs = sys.argv[1:] or [
        'f200w-f115w',   # 1182 internal, SW-SW
        'f200w-f356w',   # 1182 internal, SW-LW
        'f182m-f212n',   # 2221 internal, SW-SW
        'f182m-f410m',   # 2221 internal, SW-LW
        'f200w-f182m',   # cross-program (the joint-reference pair)
    ]
    rows = []
    for p in pairs:
        try:
            rows.append(analyze(p))
        except Exception as e:
            print(f"FAILED {p}: {e!r}")
    print("\n==== SUMMARY ====")
    print(f"{'pair':14s} {'N':>7s} {'medRA':>7s} {'medDec':>7s} {'madRA':>6s} {'madDec':>6s} {'seamRA':>7s} {'seamDec':>7s}")
    for r in rows:
        print(f"{r['pair']:14s} {r['n']:7d} {r['med_ra']:7.1f} {r['med_dec']:7.1f} {r['mad_ra']:6.1f} {r['mad_dec']:6.1f} {r['seam_jra']:7.1f} {r['seam_jdec']:7.1f}")
