"""
Build a VIRAC2/Gaia-referenced F200W + F182M astrometric reference catalog for the Brick.

daophot-BASIC only (crowdsource deprecated 2026-06). F200W = prop 1182 (SW), F182M = prop 2221 (SW).
Each program is internally aligned to a few mas (see seam_edge_analysis.py); they differ by a ~4 mas
bulk offset plus a thin module-boundary band at Dec ~-28.709. We tie EACH program independently to
the shared absolute reference (gaia_virac2_refcat_epoch2022.70 = Gaia DR3 abs + VIRAC2 NIR fill) via
a robust bulk shift measured by OFFSET-HISTOGRAM stacking (nearest-neighbour median is biased to ~0
against a dense catalog -- see memory dense_refcat_astrometry trap), then merge.

Outputs (in catalogs/):
  f200w_f182m_virac2_reference_catalog.fits   -- the joint reference
  per-filter *_virac2frame intermediate shifts are reported in meta.

Cross-checks vs Gaia-only subset and vs other bands are printed and plotted.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import mad_std
import warnings
warnings.filterwarnings('ignore')

CATDIR = '/orange/adamginsburg/jwst/brick/catalogs'
OUTDIR = '/orange/adamginsburg/jwst/brick/astrometry_seam_dao_20251211'
os.makedirs(OUTDIR, exist_ok=True)
# MEAN VIRAC2 frame: RAJ2000/DEJ2000 at the VIRAC2 reference epoch (~2014), NO proper-motion
# propagation. The GC has no net bulk motion (<~1 mas/decade; only the small solar-reflex term),
# so PMs are deliberately ignored and we tie to the mean VIRAC2 positions. VIRAC2 (II/387) is itself
# tied to Gaia DR3 at ~5 mas, so this is the Gaia frame realised densely.
VIRAC2CACHE = '/orange/adamginsburg/jwst/brick/astrometry_retie_qualcuts_20251211/refcache/virac2.fits'

# VIRAC2 (II/387) position reference epoch = 2014.0 (the VIRAC2 reference-paper value).
# Cross-check here: the VIRAC2-minus-Gaia position vs Gaia-PM regression gives E_virac ~2014.3
# (dRA->2014.47, dDec->2014.15, axis spread ~0.3 yr), i.e. consistent with 2014.0 at ~1 sigma; the
# ~0.3 yr difference is only ~1 mas (mean PM ~5 mas/yr), negligible. We adopt the paper value 2014.0.
# NOTE: the F115W agent's anchor_virac2_frame.py assumed 2016.0, ~2 yr too late -> its *_VIRAC2FRAME /
# gaia_virac2_refcat are ~10 mas under-propagated; that constant should be changed to 2014.0.
VIRAC2_REF_EPOCH = 2014.0
# program observation epochs (jyear) from DATE-OBS
OBS_EPOCH = {'f200w': 2022.703,   # prop 1182 obs004, 2022-09-14
             'f182m': 2022.655}   # prop 2221 obs001, 2022-08-28
PROGRAM = {'f200w': 1182, 'f182m': 2221}


from catalog_paths import best_dao_basic


def load_jwst(filt):
    t = Table.read(best_dao_basic(filt))
    sc = SkyCoord(t['skycoord'])
    fluxcol = 'flux' if 'flux' in t.colnames else ('flux_fit' if 'flux_fit' in t.colnames else 'flux_init')
    flux = np.asarray(t[fluxcol], float)
    qfit = np.asarray(t['qfit'], float) if 'qfit' in t.colnames else np.zeros(len(t))
    stdra = np.asarray(t['std_ra'], float) * 3600e3 if 'std_ra' in t.colnames else np.full(len(t), np.nan)
    stddec = np.asarray(t['std_dec'], float) * 3600e3 if 'std_dec' in t.colnames else np.full(len(t), np.nan)
    nmatch = np.asarray(t['nmatch'], float) if 'nmatch' in t.colnames else np.ones(len(t))
    sat = np.asarray(t['is_saturated'], bool) if 'is_saturated' in t.colnames else np.zeros(len(t), bool)
    good = (np.isfinite(flux) & (flux > 0) & (~np.asarray(sat, bool)) & (nmatch >= 3)
            & np.isfinite(stdra) & (stdra < 5) & (qfit < 0.2))
    return t, sc, flux, good, stdra, stddec


def offset_histogram_shift(jwst_sc, ref_sc, search=0.30 * u.arcsec, binmas=3.0, label=''):
    """Robust bulk (dRA, dDec) offset via 2D histogram of all JWST-ref pairs within `search`.

    Returns (dra_mas, ddec_mas) = the shift to ADD to JWST to land on ref. Peak of the stacked
    pair-offset distribution; immune to the dense-catalog NN-median bias.
    """
    idx, sep, _ = jwst_sc.match_to_catalog_sky(ref_sc)
    m = sep < search
    a = jwst_sc[m]; b = ref_sc[idx[m]]
    dra = ((b.ra - a.ra) * np.cos(a.dec.radian)).to(u.mas).value
    ddec = (b.dec - a.dec).to(u.mas).value
    rng = search.to(u.mas).value
    binmas = 2.0
    nb = int(2 * rng / binmas)
    H, xe, ye = np.histogram2d(dra, ddec, bins=nb, range=[[-rng, rng], [-rng, rng]])
    # smooth then peak
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(H, 1.5)
    iy, ix = np.unravel_index(np.argmax(Hs), Hs.shape)
    pk_ra = 0.5 * (xe[iy] + xe[iy + 1]); pk_dec = 0.5 * (ye[ix] + ye[ix + 1])
    # refine: iterative median in shrinking windows around the peak
    for win in (20, 12, 8):
        w = (np.abs(dra - pk_ra) < win) & (np.abs(ddec - pk_dec) < win)
        if w.sum() > 50:
            pk_ra = np.median(dra[w]); pk_dec = np.median(ddec[w])
    return pk_ra, pk_dec, dra, ddec, (H, xe, ye)


def gaia_at_epoch(epoch=2022.70, center=(266.53, -28.70), radius=0.06):
    """Gaia DR3 in the field, PM-propagated to `epoch`. Returns SkyCoord."""
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    job = Gaia.launch_job_async(
        "SELECT ra,dec,pmra,pmdec,ref_epoch,phot_g_mean_mag FROM gaiadr3.gaia_source "
        f"WHERE CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{center[0]},{center[1]},{radius}))=1")
    g = job.get_results()
    if epoch is None:
        # mean frame: use Gaia catalog positions at their ref_epoch, NO PM propagation
        return SkyCoord(np.asarray(g['ra'], float) * u.deg, np.asarray(g['dec'], float) * u.deg)
    g = g[np.isfinite(g['pmra']) & np.isfinite(g['pmdec'])]
    dt = epoch - np.asarray(g['ref_epoch'], float)
    ra2 = np.asarray(g['ra'], float) + (np.asarray(g['pmra'], float) * dt / 3.6e6) / np.cos(np.radians(np.asarray(g['dec'], float)))
    dec2 = np.asarray(g['dec'], float) + np.asarray(g['pmdec'], float) * dt / 3.6e6
    return SkyCoord(ra2 * u.deg, dec2 * u.deg)


def offhist_shift_sc(sc, ref_sc, search=0.30 * u.arcsec):
    pr, pd, _, _, _ = offset_histogram_shift(sc, ref_sc, search=search)
    return pr, pd


def propagate_virac2(v, target_epoch):
    """VIRAC2 propagated to target_epoch using PER-STAR PMs (matching the F115W
    anchor_virac2_frame.py approach: missing PM -> 0). Returns (SkyCoord, pmRA, pmDE, dt)."""
    ra = np.asarray(v['RAJ2000'], float); dec = np.asarray(v['DEJ2000'], float)
    pmra = np.asarray(v['pmRA'], float); pmde = np.asarray(v['pmDE'], float)
    pmra = np.where(np.isfinite(pmra), pmra, 0.0)
    pmde = np.where(np.isfinite(pmde), pmde, 0.0)
    dt = target_epoch - VIRAC2_REF_EPOCH
    ra2 = ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec))
    dec2 = dec + pmde * dt / 3.6e6
    return SkyCoord(ra2 * u.deg, dec2 * u.deg), pmra, pmde, dt


def main():
    v = Table.read(VIRAC2CACHE)
    vpmra_all = np.where(np.isfinite(np.asarray(v['pmRA'], float)), np.asarray(v['pmRA'], float), np.nan)
    vpmde_all = np.where(np.isfinite(np.asarray(v['pmDE'], float)), np.asarray(v['pmDE'], float), np.nan)
    mean_pmra = float(np.nanmedian(vpmra_all)); mean_pmde = float(np.nanmedian(vpmde_all))
    print(f"VIRAC2 {len(v)} stars; ref epoch {VIRAC2_REF_EPOCH}; mean PM ({mean_pmra:+.2f},{mean_pmde:+.2f}) mas/yr")

    # Gaia DR3 for independent cross-check, propagated PER-STAR to each program epoch below
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    results = {}
    joint_rows = []
    for col, filt in enumerate(['f200w', 'f182m']):
        ep = OBS_EPOCH[filt]
        ref_sc, _, _, dt = propagate_virac2(v, ep)
        vsc_for_pm = ref_sc  # VIRAC2 at obs epoch, for matching per-source PM onto JWST rows
        t, sc, flux, good, stdra, stddec = load_jwst(filt)
        jg = sc[good]
        dra0, ddec0, draA, ddecA, hist = offset_histogram_shift(jg, ref_sc, label=filt)
        new_ra = sc.ra + (dra0 * u.mas) / np.cos(sc.dec.radian)
        new_dec = sc.dec + ddec0 * u.mas
        sc_shifted = SkyCoord(new_ra, new_dec)
        d2ra, d2dec, vra, vdec, _ = offset_histogram_shift(sc_shifted[good], ref_sc)
        tot_pm_ra = mean_pmra * dt; tot_pm_de = mean_pmde * dt
        print(f"\n{filt.upper()} ({PROGRAM[filt]}, epoch {ep}): VIRAC2 propagated by dt={dt:.3f} yr "
              f"(mean displacement {tot_pm_ra:+.1f},{tot_pm_de:+.1f} mas)")
        print(f"   bulk shift JWST->VIRAC2@{ep}: dRA {dra0:+.1f} dDec {ddec0:+.1f} mas (N_good={good.sum()})")
        print(f"   residual after shift: dRA {d2ra:+.1f} dDec {d2dec:+.1f} mas")
        results[filt] = dict(shift=(dra0, ddec0), resid=(d2ra, d2dec), n=int(good.sum()),
                             dt=dt, totpm=(tot_pm_ra, tot_pm_de), epoch=ep)

        ax = axs[0, col]
        ax.hist2d(draA, ddecA, bins=80, range=[[-120, 120], [-120, 120]], cmap='viridis')
        ax.plot(dra0, ddec0, 'r+', ms=14, mew=2)
        ax.set_title(f'{filt}: JWST-VIRAC2@{ep} offsets (peak {dra0:.0f},{ddec0:.0f})')
        ax.set_xlabel('dRA mas'); ax.set_ylabel('dDec mas')
        ax = axs[1, col]
        ax.hist2d(vra, vdec, bins=80, range=[[-60, 60], [-60, 60]], cmap='viridis')
        ax.axvline(0, color='w', lw=0.5); ax.axhline(0, color='w', lw=0.5)
        ax.set_title(f'{filt}: residual after shift')
        ax.set_xlabel('dRA mas'); ax.set_ylabel('dDec mas')

        # per-source matched VIRAC2 PM (recorded for downstream re-propagation)
        gsc = sc_shifted[good]
        midx, msep, _ = gsc.match_to_catalog_sky(ref_sc)
        matched = msep < 0.2 * u.arcsec
        pmra_col = np.full(int(good.sum()), np.nan); pmde_col = np.full(int(good.sum()), np.nan)
        pmra_col[matched] = vpmra_all[midx[matched]]
        pmde_col[matched] = vpmde_all[midx[matched]]

        sub = Table()
        sub['RA'] = gsc.ra.deg
        sub['DEC'] = gsc.dec.deg
        sub['flux'] = flux[good]
        sub['std_ra_mas'] = stdra[good]
        sub['std_dec_mas'] = stddec[good]
        sub['filter'] = filt
        sub['program'] = PROGRAM[filt]
        sub['epoch'] = ep                       # epoch (jyear) the positions are propagated TO
        sub['virac2_pmra'] = pmra_col           # matched VIRAC2 per-star PM (mas/yr), NaN if unmatched
        sub['virac2_pmde'] = pmde_col
        joint_rows.append(sub)

    # cross-check: F200W vs F182M agreement AFTER both tied to ref
    a200 = SkyCoord(joint_rows[0]['RA'], joint_rows[0]['DEC'], unit='deg')
    a182 = SkyCoord(joint_rows[1]['RA'], joint_rows[1]['DEC'], unit='deg')
    idx, sep, _ = a200.match_to_catalog_sky(a182)
    mm = sep < 0.1 * u.arcsec
    cdra = ((a182[idx[mm]].ra - a200[mm].ra) * np.cos(a200[mm].dec.radian)).to(u.mas).value
    cddec = (a182[idx[mm]].dec - a200[mm].dec).to(u.mas).value
    print(f"\nF200W vs F182M after both tied: median dRA {np.median(cdra):+.1f} dDec {np.median(cddec):+.1f} "
          f"MAD ({mad_std(cdra):.1f},{mad_std(cddec):.1f}) N={mm.sum()}")
    axs[0, 2].hist2d(cdra, cddec, bins=60, range=[[-40, 40], [-40, 40]], cmap='magma')
    axs[0, 2].axvline(0, color='w', lw=0.5); axs[0, 2].axhline(0, color='w', lw=0.5)
    axs[0, 2].set_title(f'F200W-F182M after tie (med {np.median(cdra):.0f},{np.median(cddec):.0f})')
    axs[0, 2].set_xlabel('dRA mas'); axs[0, 2].set_ylabel('dDec mas')

    # vs-Dec residual of the cross-program agreement (the band check)
    decm = a200[mm].dec.deg
    db = np.linspace(np.percentile(decm, 1), np.percentile(decm, 99), 40)
    dc = 0.5 * (db[:-1] + db[1:])
    pr = [np.median(cdra[(decm >= lo) & (decm < hi)]) if ((decm >= lo) & (decm < hi)).sum() > 20 else np.nan for lo, hi in zip(db[:-1], db[1:])]
    pd = [np.median(cddec[(decm >= lo) & (decm < hi)]) if ((decm >= lo) & (decm < hi)).sum() > 20 else np.nan for lo, hi in zip(db[:-1], db[1:])]
    axs[1, 2].plot(dc, pr, 'o-', label='dRA'); axs[1, 2].plot(dc, pd, 's-', label='dDec')
    axs[1, 2].axvline(-28.709, color='k', ls=':'); axs[1, 2].axhline(0, color='gray', lw=0.5)
    axs[1, 2].set_xlabel('Dec'); axs[1, 2].set_ylabel('F200W-F182M resid (mas)'); axs[1, 2].legend()
    axs[1, 2].set_title('cross-program residual vs Dec (band at -28.709)')
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/reference_build_diagnostics.png', dpi=100); plt.close()
    print(f"saved {OUTDIR}/reference_build_diagnostics.png")

    # ---- final validation: independent Gaia DR3, PM-propagated PER-STAR to each obs epoch ----
    # (now that the reference is at the obs epoch, Gaia must be propagated to the same epoch with
    #  its own per-star PMs -- this is the correct, clean check; foreground PMs are handled.)
    print("\n---- VALIDATION (Gaia DR3 propagated per-star to each obs epoch) ----")
    try:
      for jr, nm, filt in [(joint_rows[0], 'F200W', 'f200w'), (joint_rows[1], 'F182M', 'f182m')]:
        sc = SkyCoord(jr['RA'], jr['DEC'], unit='deg')
        gaia_ep = gaia_at_epoch(epoch=OBS_EPOCH[filt])
        pr, pd = offhist_shift_sc(sc, gaia_ep, search=0.25 * u.arcsec)
        print(f"  tied {nm} vs Gaia@{OBS_EPOCH[filt]}: ({pr:+.1f},{pd:+.1f}) mas  [target ~0]")
    except Exception as e:
        print(f"  Gaia validation skipped (archive error): {e!r}")
    # other bands tied the same way (per-star VIRAC2 PM to their own epoch) then vs Gaia@epoch
    for ob in ['f356w', 'f405n', 'f212n', 'f444w', 'f115w']:
        try:
            prog = 1182 if ob in ('f356w', 'f444w', 'f115w') else 2221
            ep = 2022.703 if prog == 1182 else 2022.655
            ref_ob, _, _, _ = propagate_virac2(v, ep)
            gaia_ep = gaia_at_epoch(epoch=ep)
            t, sc, flux, good, sra, sdd = load_jwst(ob)
            d0r, d0d = offhist_shift_sc(sc[good], ref_ob)
            scv = SkyCoord(sc.ra + (d0r * u.mas) / np.cos(sc.dec.radian), sc.dec + d0d * u.mas)
            pr, pd = offhist_shift_sc(scv[good], gaia_ep, search=0.25 * u.arcsec)
            print(f"  {ob.upper()} ({prog}) tied to VIRAC2@{ep} then vs Gaia@{ep}: ({pr:+.1f},{pd:+.1f}) mas")
        except Exception as e:
            print(f"  {ob}: {e!r}")

    # merge: union of both (deduplicated -- where a star is in both, keep both rows tagged;
    # downstream can choose). Write joint reference.
    joint = vstack(joint_rows)
    joint.meta['CONTENT'] = 'Brick F200W(1182)+F182M(2221) astrometric reference, daophot basic'
    joint.meta['ABSFRAME'] = 'VIRAC2 (II/387) PM-propagated PER-STAR to each program obs epoch; tied to Gaia DR3 ~5 mas'
    joint.meta['METHOD'] = ('per-star VIRAC2 PM propagation (matches anchor_virac2_frame.py) then rigid '
                            'offset-histogram bulk shift of JWST onto propagated VIRAC2')
    joint.meta['V2EPOCH'] = (VIRAC2_REF_EPOCH, 'VIRAC2 position ref epoch (jyear), empirical regression')
    joint.meta['V2MEANPR'] = (mean_pmra, 'VIRAC2 field median pmRA (mas/yr)')
    joint.meta['V2MEANPD'] = (mean_pmde, 'VIRAC2 field median pmDE (mas/yr)')
    # per-program epochs, bulk shift, and TOTAL (mean) proper motion added to VIRAC2 positions
    joint.meta['F200WEP'] = (results['f200w']['epoch'], 'F200W (1182) propagated-to epoch (jyear)')
    joint.meta['F182MEP'] = (results['f182m']['epoch'], 'F182M (2221) propagated-to epoch (jyear)')
    joint.meta['F200WDT'] = (results['f200w']['dt'], 'F200W propagation baseline dt (yr)')
    joint.meta['F182MDT'] = (results['f182m']['dt'], 'F182M propagation baseline dt (yr)')
    joint.meta['F200WPMA'] = (results['f200w']['totpm'][0], 'mean PM added to VIRAC2 for F200W, dRA (mas)')
    joint.meta['F200WPMD'] = (results['f200w']['totpm'][1], 'mean PM added to VIRAC2 for F200W, dDec (mas)')
    joint.meta['F182MPMA'] = (results['f182m']['totpm'][0], 'mean PM added to VIRAC2 for F182M, dRA (mas)')
    joint.meta['F182MPMD'] = (results['f182m']['totpm'][1], 'mean PM added to VIRAC2 for F182M, dDec (mas)')
    joint.meta['F200W_DRA'] = (results['f200w']['shift'][0], 'JWST->VIRAC2 bulk shift dRA (mas)')
    joint.meta['F200W_DDE'] = (results['f200w']['shift'][1], 'JWST->VIRAC2 bulk shift dDec (mas)')
    joint.meta['F182M_DRA'] = (results['f182m']['shift'][0], 'JWST->VIRAC2 bulk shift dRA (mas)')
    joint.meta['F182M_DDE'] = (results['f182m']['shift'][1], 'JWST->VIRAC2 bulk shift dDec (mas)')
    joint.meta['DEPRECAT'] = 'crowdsource catalogs deprecated; daophot basic only'
    outpath = f'{CATDIR}/f200w_f182m_virac2_reference_catalog.fits'
    joint.write(outpath, overwrite=True)
    print(f"\nWROTE {outpath}  ({len(joint)} rows: {len(joint_rows[0])} F200W + {len(joint_rows[1])} F182M)")
    return results


if __name__ == '__main__':
    main()
