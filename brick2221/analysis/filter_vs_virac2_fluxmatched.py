"""
Per-filter JWST vs VIRAC2 astrometric offset, FLUX-MATCHED, for the JHK-ish bands
(F115W~J, F182M~H, F200W~Ks, F212N~Ks). daophot BASIC only.

VIRAC2 is PM-propagated PER-STAR (II/387 ref epoch 2014.0) to each program's obs epoch
(1182 F115W/F200W -> 2022.703; 2221 F182M/F212N -> 2022.655). We mutually match JWST<->VIRAC2,
then DOWN-SELECT by a reasonable flux/brightness match: fit JWST instrumental mag (-2.5 log10 flux)
vs the matched VIRAC2 band mag and keep the ~3-sigma core. This removes crowded-field false matches
that otherwise inflate the GC scatter. Each panel: offset cloud (centered on the median), the
flux-match relation, and a binned dRA/dDec sky map.

Output: astrometry_seam_dao_20251211/viracmatch_<filter>.png  + a printed summary table.
"""
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
VIRAC2CACHE = '/orange/adamginsburg/jwst/brick/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'
VIRAC2_REF_EPOCH = 2014.0
os.makedirs(OUTDIR, exist_ok=True)

# filter -> (program obs epoch, VIRAC2 band column for the flux match)
BANDS = {
    'f115w': (2022.703, 'Jmag'),
    'f182m': (2022.655, 'Hmag'),
    'f200w': (2022.703, 'Ksmag'),
    'f212n': (2022.655, 'Ksmag'),
}


def load_virac2(epoch):
    v = Table.read(VIRAC2CACHE)
    ra = np.asarray(v['RAJ2000'], float); dec = np.asarray(v['DEJ2000'], float)
    pmra = np.where(np.isfinite(np.asarray(v['pmRA'], float)), np.asarray(v['pmRA'], float), 0.)
    pmde = np.where(np.isfinite(np.asarray(v['pmDE'], float)), np.asarray(v['pmDE'], float), 0.)
    dt = epoch - VIRAC2_REF_EPOCH
    ra2 = ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec))
    dec2 = dec + pmde * dt / 3.6e6
    return v, SkyCoord(ra2 * u.deg, dec2 * u.deg)


def load_jwst(filt):
    t = Table.read(f'{CATDIR}/{filt}_merged_indivexp_merged_dao_basic.fits')
    sc = SkyCoord(t['skycoord'])
    fcol = 'flux' if 'flux' in t.colnames else ('flux_fit' if 'flux_fit' in t.colnames else 'flux_init')
    flux = np.asarray(t[fcol], float)
    qfit = np.asarray(t['qfit'], float) if 'qfit' in t.colnames else np.zeros(len(t))
    nmatch = np.asarray(t['nmatch'], float) if 'nmatch' in t.colnames else np.ones(len(t))
    stdra = np.asarray(t['std_ra'], float) * 3600e3 if 'std_ra' in t.colnames else np.full(len(t), np.nan)
    sat = np.asarray(t['is_saturated'], bool) if 'is_saturated' in t.colnames else np.zeros(len(t), bool)
    good = (np.isfinite(flux) & (flux > 0) & (~np.asarray(sat, bool)) & (nmatch >= 3)
            & (qfit < 0.2) & np.isfinite(stdra) & (stdra < 5))
    return sc[good], flux[good]


def analyze(filt, ax_cloud, ax_rel, ax_map):
    epoch, vband = BANDS[filt]
    v, vsc = load_virac2(epoch)
    vmag_all = np.asarray(v[vband], float)
    jsc, jflux = load_jwst(filt)
    # mutual NN match
    idx, sep, _ = jsc.match_to_catalog_sky(vsc)
    ridx, _, _ = vsc.match_to_catalog_sky(jsc)
    keep = (ridx[idx] == np.arange(len(idx))) & (sep < 0.3 * u.arcsec)
    a = jsc[keep]; b = vsc[idx[keep]]
    jmag = -2.5 * np.log10(jflux[keep])
    vmag = vmag_all[idx[keep]]
    ok = np.isfinite(jmag) & np.isfinite(vmag)
    a, b, jmag, vmag = a[ok], b[ok], jmag[ok], vmag[ok]

    # FLUX MATCH: robust linear fit jmag = m*vmag + c, keep 3-sigma core
    sel = np.ones(len(jmag), bool)
    for _ in range(5):
        c = np.polyfit(vmag[sel], jmag[sel], 1)
        resid = jmag - np.polyval(c, vmag)
        s = mad_std(resid[sel])
        newsel = np.abs(resid) < 3 * s
        if newsel.sum() == sel.sum():
            sel = newsel; break
        sel = newsel
    # also restrict to the well-populated bright-ish magnitude range (avoid faint incompleteness)
    vlo, vhi = np.percentile(vmag[sel], [2, 85])
    fm = sel & (vmag > vlo) & (vmag < vhi)

    dra = ((b.ra - a.ra) * np.cos(a.dec.radian)).to(u.mas).value
    ddec = (b.dec - a.dec).to(u.mas).value
    mr, md = np.median(dra[fm]), np.median(ddec[fm])
    sr, sd = mad_std(dra[fm]), mad_std(ddec[fm])
    n = fm.sum()

    # offset cloud (centered on median)
    ax_cloud.hist2d(dra[fm], ddec[fm], bins=60, range=[[mr - 60, mr + 60], [md - 60, md + 60]], cmap='viridis')
    ax_cloud.plot(mr, md, 'r+', ms=14, mew=2)
    ax_cloud.set_title(f'{filt.upper()} - VIRAC2({vband[0]}) @{epoch}\nmed ({mr:.1f},{md:.1f}) MAD ({sr:.1f},{sd:.1f}) N={n}')
    ax_cloud.set_xlabel('dRA (mas)'); ax_cloud.set_ylabel('dDec (mas)')

    # flux-match relation
    ax_rel.scatter(vmag[~fm], jmag[~fm], s=1, c='lightgray', label='rejected')
    ax_rel.scatter(vmag[fm], jmag[fm], s=1, c='C0', label='kept')
    xs = np.linspace(np.nanmin(vmag), np.nanmax(vmag), 10)
    ax_rel.plot(xs, np.polyval(c, xs), 'r-', lw=1)
    ax_rel.set_xlabel(f'VIRAC2 {vband}'); ax_rel.set_ylabel('JWST inst mag (-2.5log10 flux)')
    ax_rel.set_title(f'{filt.upper()} flux match (kept {fm.sum()}/{len(jmag)})')
    ax_rel.invert_yaxis(); ax_rel.legend(markerscale=6, loc='best', fontsize=7)

    # binned sky map of dRA (residual about its own median)
    ra = a.ra.deg[fm]; dec = a.dec.deg[fm]
    from scipy.stats import binned_statistic_2d
    nb = 35
    rb = np.linspace(np.percentile(ra, 1), np.percentile(ra, 99), nb)
    db = np.linspace(np.percentile(dec, 1), np.percentile(dec, 99), nb)
    stat, _, _, _ = binned_statistic_2d(ra, dec, dra[fm] - mr, statistic='median', bins=[rb, db])
    im = ax_map.imshow(stat.T, origin='lower', extent=[rb[0], rb[-1], db[0], db[-1]], aspect='auto',
                       cmap='RdBu_r', vmin=-20, vmax=20)
    ax_map.axhline(-28.709, color='k', ls=':', lw=1)
    ax_map.invert_xaxis(); ax_map.set_title(f'{filt.upper()} dRA-median map (mas)')
    plt.colorbar(im, ax=ax_map)
    return dict(filt=filt, vband=vband, epoch=epoch, n=int(n), med_ra=mr, med_dec=md, mad_ra=sr, mad_dec=sd)


def main():
    rows = []
    for filt in BANDS:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
        r = analyze(filt, axs[0], axs[1], axs[2])
        plt.tight_layout()
        out = f'{OUTDIR}/viracmatch_{filt}.png'
        plt.savefig(out, dpi=110); plt.close()
        print(f"saved {out}")
        rows.append(r)
    print("\n==== JWST vs VIRAC2 (flux-matched) summary ====")
    print(f"{'filt':6s} {'vband':6s} {'epoch':8s} {'N':>7s} {'medRA':>7s} {'medDec':>7s} {'madRA':>6s} {'madDec':>6s}")
    for r in rows:
        print(f"{r['filt']:6s} {r['vband']:6s} {r['epoch']:<8.3f} {r['n']:7d} {r['med_ra']:7.1f} {r['med_dec']:7.1f} {r['mad_ra']:6.1f} {r['mad_dec']:6.1f}")


if __name__ == '__main__':
    main()
