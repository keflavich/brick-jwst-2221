import numpy as np
import time
import datetime
import os
import sys
import warnings
from astropy.io import fits
import glob
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder, find_peaks
from photutils.psf import (extract_stars, EPSFBuilder)
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import stats
from astropy.table import Table, Column, MaskedColumn
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u
from astroquery.svo_fps import SvoFps
from astropy.stats import sigma_clip, mad_std
import dask
import dask.array
import yaml # DEBUG 2025-12-11
import yaml.representer # DEBUG 2025-12-11

from tqdm.auto import tqdm

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['figure.figsize'] = (10, 8)
pl.rcParams['figure.dpi'] = 100

# https://en.wikipedia.org/wiki/AB_magnitude
ABMAG_OFFSET = 8.90

filternames = filternames_narrow = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m']
all_filternames = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m', 'f444w', 'f356w', 'f200w', 'f115w']
obs_filters = {'brick': {'2221': filternames,
                         '1182': ['f444w', 'f356w', 'f200w', 'f115w'],
                         },
               'cloudc': {'2221': filternames},
               'sickle': {'3958': ['f187n', 'f210m', 'f335m', 'f470n', 'f480m']},
               'cloudef': {'2092': ['f162m', 'f210m', 'f360m', 'f480m']},
               'sgrc': {'4147': ['f115w', 'f162m', 'f182m', 'f212n', 'f360m', 'f405n', 'f470n', 'f480m']},
               'sgrb2': {'5365': ['f150w', 'f182m', 'f187n', 'f210m', 'f212n', 'f300m', 'f360m', 'f405n', 'f410m', 'f466n', 'f480m']},
               'arches': {'2045': ['f212n', 'f323n']},
               'quintuplet': {'2045': ['f212n', 'f323n']},
               'sgra': {'1939': ['f115w', 'f212n', 'f405n']},
               }

# Using the 'brick' keyword here makes it work for now, need to figure out how to
# refactor it in cases where there are more filters available for other targets!
filter_to_project = {vv: key for target_filters in obs_filters.values() for key, val in target_filters.items() for vv in val}
# need to refactor this somehow for cloudc
# project_obsnum = {'2221': '001',
#                   '1182': '004',
#                  }

project_obsnum = {'brick': {'2221': '001',
                            '1182': '004',
                            },
                  'cloudc': {'2221': '002',
                             },
                  'sickle': {'3958': '007',
                             },
                  'cloudef': {'2092': '005',
                              },
                  'sgrc': {'4147': '012',
                           },
                  'sgrb2': {'5365': '001',
                            },
                  'arches': {'2045': '001',
                             },
                  'quintuplet': {'2045': '003',
                                 },
                  'sgra': {'1939': '001',
                           },
                  }


def getmtime(x):
    return datetime.datetime.fromtimestamp(os.path.getmtime(x)).strftime('%Y-%m-%d %H:%M:%S')


def tryint(x):
    try:
        return int(x)
    except:
        return -1


def sanity_check_individual_table(tbl):
    wl = filtername = tbl.meta['filter']
    print(f"SANITY CHECK {wl}")

    tbl = tbl.copy()
    tbl.sort('flux_jy')
    finite_fluxes = tbl['flux_jy'] > 0

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    zeropoint = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername.upper()}']['ZeroPoint'], u.Jy)

    flux_jy = tbl['flux_jy'][finite_fluxes].quantity
    abmag_tbl = tbl['mag_ab'][finite_fluxes].quantity

    vegamag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag
    abmag = (-2.5 * np.log10(flux_jy / u.Jy) + ABMAG_OFFSET) * u.mag

    print(f'Units of abmag columns are: abmag={abmag.unit}, abmag_tbl={abmag_tbl.unit}')
    assert abmag.unit == u.mag
    assert abmag_tbl.unit == u.mag
    assert flux_jy.unit == u.Jy

    # there are negative fluxes -> nan mags
    # print(f"Maximum difference between the two tables: {np.abs(abmag-abmag_tbl).max()}")
    print(f"Nanmax difference between the two tables: {np.nanmax(np.abs(abmag-abmag_tbl))}")
    # print("NaNs: (mag, flux) ", abmag_tbl[np.isnan(abmag_tbl)], flux_jy[np.isnan(abmag_tbl)])

    fluxcolname = 'flux' if 'flux' in tbl.colnames else 'flux_fit'
    print(f"Max flux in tbl for {wl}: {tbl[fluxcolname].max()};"
          f" in jy={flux_jy.max()}; magmin={abmag_tbl.min()}={np.nanmin(abmag_tbl)}, magmax={abmag_tbl.max()}={np.nanmax(abmag_tbl)}")
    print(f"100th brightest flux={flux_jy[-100]} abmag={abmag[-100]} abmag_tbl={abmag_tbl[-100]}")


def nanaverage_numpy(data, weights, **kwargs):
    print(data.shape, weights.shape)
    weights = np.where(np.isnan(data) | np.isnan(weights), 0, weights)
    bad = np.all(weights == 0, axis=1)
    weights[bad, :] = 1
    avg = np.average(np.nan_to_num(data),
                     weights=weights,
                     **kwargs
                     )
    avg[bad] = np.nan
    return avg


def nanaverage_dask(data, weights, **kwargs):
    weights = dask.array.from_array(weights)
    data = dask.array.from_array(data)
    weights = dask.array.where(dask.array.isnan(data) | dask.array.isnan(weights), 0, weights)
    bad = dask.array.all(weights == 0, axis=1)
    weights[bad, :] = 1
    avg = dask.array.average(np.nan_to_num(data), weights=weights, **kwargs)
    avg[bad] = np.nan
    return avg.compute()


def shift_individual_catalog(tbl, offsets_table, verbose=True):
    """
    offsets_table:
        A table to use to re-calculate sky coordinates from the WCS after
        shifting it.  This can be used because the catalogs are all
        intrinsically in pixel space, so changing the shift after the fact is OK.
        Using an offset table enables splitting out the re-alignment task from
        here; I want to be able to measure the alignment and be sure it's right
        before applying it.
    """
    if 'Visit' in tbl.meta:
        visit = int(tbl.meta['Visit'])
    elif 'VISIT' in tbl.meta:
        visit = int(tbl.meta['VISIT'])
    elif 'visit' in tbl.meta:
        visit = int(tbl.meta['visit'])
    else:
        print(tbl.meta)
        raise KeyError("'Visit' not found in meta")
    exposure = int(tbl.meta['EXPOSURE'][-5:])
    thismodule = tbl.meta['MODULE']
    if thismodule.endswith('a') or thismodule.endswith('b'):
        thismodule = thismodule+'long'
    filtername = tbl.meta['FILTER']

    offsets_visit_number = np.array([int(vis[-3:]) for vis in offsets_table['Visit']])

    match = ((offsets_visit_number == visit) &
             (offsets_table['Exposure'] == exposure) &
             ((offsets_table['Module'] == thismodule) | (offsets_table['Module'] == thismodule.strip('1234'))) &
             (offsets_table['Filter'] == filtername)
             )

    assert match.sum() == 1
    row = offsets_table[match]

    if 'RAOFFSET' in tbl.meta:
        raoffset = tbl.meta['RAOFFSET'] * u.arcsec
        decoffset = tbl.meta['DEOFFSET'] * u.arcsec
    else:
        # not measured, so we have to assume zero
        raoffset = 0 * u.arcsec
        decoffset = 0 * u.arcsec

    dra = row['dra'][0]*u.arcsec
    ddec = row['ddec'][0]*u.arcsec

    skycoord_colname = 'skycoord' if 'skycoord' in tbl.colnames else 'skycoord_centroid'

    skycoord = tbl[skycoord_colname]
    skycoord = SkyCoord(ra=skycoord.ra - raoffset + dra, dec=skycoord.dec - decoffset + ddec, frame=skycoord.frame)
    tbl[skycoord_colname] = skycoord

    print(f"Shifted table from {raoffset:0.4f},{decoffset:0.4f} to {dra:0.4f},{ddec:0.4f}, a difference of {dra-raoffset:0.4f},{ddec-decoffset:0.4f}")

    return tbl


def combine_singleframe(tbls, max_offset=0.10 * u.arcsec, realign=False, nanaverage=nanaverage_dask,
                        min_offset=0.10*u.arcsec,
                        offsets_table=None,
                        verbose=True
                        ):
    """

    min_offset :
        The minimum allowed offset to declare a 'new' star.  Anything below this is assumed same star.

    offsets_table:
        A table to use to re-calculate sky coordinates from the WCS after
        shifting it.  This can be used because the catalogs are all
        intrinsically in pixel space, so changing the shift after the fact is OK.
        Using an offset table enables splitting out the re-alignment task from
        here; I want to be able to measure the alignment and be sure it's right
        before applying it.
    """
    if offsets_table is not None:
        tbls = [shift_individual_catalog(tbl, offsets_table, verbose=verbose) for tbl in tbls]

    # set up DAO vs crowd column names
    if 'qf' in tbls[0].colnames:
        qfcn = 'qf'
        ffcn = 'fracflux'
        flux_error_colname = 'dflux'
        flux_colname = 'flux'
        skycoord_colname = 'skycoord'
        column_names = (flux_colname, flux_error_colname, 'qf', 'rchi2', 'fracflux', 'fwhm', 'fluxiso', 'flags', 'spread_model', 'sky', 'ra', 'dec', 'dra', 'ddec', )
        dao = False
    else:
        dao = True
        qfcn = 'qfit'
        ffcn = 'cfit'
        flux_error_colname = 'flux_err'
        flux_colname = 'flux_fit'
        # skycoord comes in as skycoord_centroid but we want it to leave as skycoord
        skycoord_colname = 'skycoord_centroid'
        column_names = (flux_colname, flux_error_colname, 'qfit', 'cfit', 'flux_init', 'flags', 'local_bkg', 'iter_detected', 'group_id', 'group_size', 'ra', 'dec', 'dra', 'ddec', )

    # Loop 1: Add new sources, which are any that don't have a match in the existing catalog closer than min_offset
    # this loop _only_ adds new sources
    for ii, tbl in enumerate(tbls):
        crds = tbl[skycoord_colname]
        # corner case: some fits resulted in flagged x, y that propagate through.  A parallel edit to crowdsource_catalogs_long.py removes these at the source, but I'm adding a catch here too
        bad = np.isnan(crds.ra) | np.isnan(crds.dec)
        if np.any(bad):
            tbl = tbl[~bad]
            crds = crds[~bad]
            tbls[ii] = tbl

        if ii == 0:
            basecrds = crds
        else:
            matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
            reverse_matches, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)

            # add new sources to the cat iff their separation from an existing source in the catalog is >min
            keep = sep > min_offset

            newcrds = crds[keep]
            basecrds = SkyCoord([basecrds, newcrds])
            print(f"Added {len(newcrds)} new sources in exposure {tbl.meta['exposure']} {tbl.meta['MODULE'] if 'MODULE' in tbl.meta else ''} [total={len(basecrds)}]")
            # f" ({mutual_matches.sum()} mutual matches ({(~mutual_matches).sum()} not), {(sep > max_offset).sum()} above {max_offset}, keeping {keep.sum()}), ", flush=True)
        print(f"Iteration {ii}: There are a total of {len(basecrds)} sources in the base coordinate list [method={'daophot' if dao else 'crowdsource'}]")

    # do one loop of re-matching
    # We use only mutual best-matches for the realignment measurement to avoid spurious matches, e.g., if there are three stars in a line, we only want to match two if they are each other's best match
    print("Starting re-matching", flush=True)
    for ii, tbl in enumerate(tbls):
        crds = tbl[skycoord_colname]

        match_inds, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_match_inds, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
        mutual_reverse_matches = (match_inds[reverse_match_inds] == np.arange(len(reverse_match_inds)))
        mutual_matches = (reverse_match_inds[match_inds] == np.arange(len(match_inds)))

        # do one iteration of bulk offset measurement
        radiff = (crds.ra[reverse_match_inds[mutual_reverse_matches]] - basecrds[mutual_reverse_matches].ra).to(u.arcsec)
        decdiff = (crds.dec[reverse_match_inds[mutual_reverse_matches]] - basecrds[mutual_reverse_matches].dec).to(u.arcsec)

        # don't allow sep=0, since that's self-reference.  Use stringent qf, fracflux
        # DEBUG print(f"len(crds) = {len(crds)} len(basecrds) = {len(basecrds)} len(match_inds)={len(match_inds)} match_inds.max={match_inds.max()} len(reverse_match_inds)={len(reverse_match_inds)} reverse_match_inds.max={reverse_match_inds.max()} len(mutual_matches)={len(mutual_matches)}")
        if dao:
            oksep = (reverse_sep[mutual_reverse_matches] < max_offset) & (reverse_sep[mutual_reverse_matches] != 0) & (tbl[reverse_match_inds[mutual_reverse_matches]][qfcn] < 0.40) & (tbl[reverse_match_inds[mutual_reverse_matches]][ffcn] < 0.40)
        else:
            oksep = (reverse_sep[mutual_reverse_matches] < max_offset) & (reverse_sep[mutual_reverse_matches] != 0) & (tbl[reverse_match_inds[mutual_reverse_matches]][qfcn] > 0.95) & (tbl[reverse_match_inds[mutual_reverse_matches]][ffcn] > 0.85)
        medsep_ra, medsep_dec = np.median(radiff[oksep]), np.median(decdiff[oksep])
        dmedsep_ra, dmedsep_dec = mad_std(radiff[oksep]), mad_std(decdiff[oksep])
        tbl.meta['ra_offset'] = medsep_ra
        tbl.meta['dec_offset'] = medsep_dec
        tbl.meta['dra_offset'] = dmedsep_ra
        tbl.meta['ddec_offset'] = dmedsep_dec

        with fits.open(tbl.meta['FILENAME']) as fh:
            if 'RAOFFSET' in fh['SCI'].header:
                dra_header = fh['SCI'].header['RAOFFSET']
                ddec_header = fh['SCI'].header['DEOFFSET']
            else:
                # assume zero
                dra_header = 0.0
                ddec_header = 0.0

        print(f"Exposure {tbl.meta['exposure']} {tbl.meta['MODULE' if 'MODULE' in tbl.meta else '']} was offset by {medsep_ra.to(u.marcsec):10.3f}+/-{dmedsep_ra.to(u.marcsec):7.3f},"
              f" {medsep_dec.to(u.marcsec):10.3f}+/-{dmedsep_dec.to(u.marcsec):7.3f} based on {oksep.sum()} matches.  dra={dra_header:7.5g} ddec={ddec_header:7.5g}")

        # for tbl0, should be nan (all self-match)
        if realign and not np.isnan(medsep_ra) and not np.isnan(medsep_dec):
            newcrds = SkyCoord(crds.ra - medsep_ra, crds.dec - medsep_dec, frame=crds.frame)
            tbl[skycoord_colname] = newcrds

    if realign:
        print("Realigning")
        # remake base coordinates after the rematching
        for ii, tbl in enumerate(tbls):
            crds = tbl[skycoord_colname]
            if ii == 0:
                basecrds = crds
            else:
                matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
                
                # add new sources to the cat iff their separation from an existing source in the catalog is >min
                keep = (sep > min_offset)

                newcrds = crds[keep]
                basecrds = SkyCoord([basecrds, newcrds])
                print(f"Added {len(newcrds)} new sources in exposure {tbl.meta['exposure']} {tbl.meta['MODULE' if 'MODULE' in tbl.meta else '']}")

    print(f"There are a total of {len(basecrds)} sources in the base coordinate list after rematching")

    assert flux_error_colname in tbls[0].colnames
    assert flux_error_colname in column_names

    # -- Memory-streaming refactor (2026-04-23) -------------------------------
    # The previous version allocated all per-column 2-D arrays of shape
    # (n_src, n_tbl) up front -- ~94 GB for the F200W brick merge, which
    # OOM'd even at 512 GB due to squaring temporaries later on.  The
    # refactor below runs in two phases:
    #
    #   Phase 1 -- allocate just ra/dec/flux/flux_err to compute the
    #              mask (sigma-clip on flux and position), weights, the
    #              position averages, and the flux/flux_err reductions.
    #              Saves per-tbl ``match_inds`` and ``keep`` so Phase 2
    #              doesn't repeat the expensive ``match_to_catalog_sky``.
    #              Peak Phase 1 memory for F200W brick ~= 4 x 4.4M x 192
    #              x 8 bytes = 27 GB.
    #   Phase 2 -- stream each remaining column one at a time: allocate
    #              one 2-D array, fill from saved match indices, compute
    #              ``_avg`` + ``std_*_avg``, free.  Peak ~7 GB per column.
    #
    # Output columns preserved: the 1-D ``_avg`` / ``std_*_avg`` columns
    # for every key in ``column_names``, plus ``skycoord_avg``, ``nmatch``,
    # ``nmatch_good``, ``std_ra``, ``std_dec``, and
    # ``f'{flux_error_colname}_prop'``.  The 2-D per-exposure arrays are
    # NOT kept in the returned table (they were the memory culprit and
    # downstream code doesn't consume them; the _allcols variant written
    # by the caller now contains only these per-source columns).
    # ------------------------------------------------------------------------

    n_src = len(basecrds)
    n_tbl = len(tbls)

    # Save per-tbl match results for Phase 2 reuse.
    # match_inds is a length-n_tbl_rows int array (index into basecrds for
    # each row in this tbl).  Kept as int32 to save RAM.
    saved_match_inds = [None] * n_tbl
    saved_keep = [None] * n_tbl

    # Phase 1: ra/dec/flux/flux_err stack.
    # ra/dec stay float64: at ~270 deg, float32 has ~0.1" quantum which
    # destroys astrometric precision (we need ~5 mas).  flux and
    # flux_err are fine in float32 (values span 10-1e6, flux ratios and
    # sigma-clip masks are tolerant to 1e-7 relative precision).
    print(f"Phase 1: stacking ra/dec/flux/flux_err for {n_src} sources x {n_tbl} tables", flush=True)
    arr_ra = np.full((n_src, n_tbl), np.nan, dtype='float64')
    arr_dec = np.full((n_src, n_tbl), np.nan, dtype='float64')
    arr_flux = np.full((n_src, n_tbl), np.nan, dtype='float32')
    arr_fluxerr = np.full((n_src, n_tbl), np.nan, dtype='float32')

    for ii, tbl in enumerate(tqdm(tbls, desc='Phase 1 (ra/dec/flux stack)')):
        crds = tbl[skycoord_colname]
        match_inds, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_match_inds, _, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
        mutual_matches = (reverse_match_inds[match_inds] == np.arange(len(match_inds)))
        keep = (sep < max_offset) & mutual_matches

        # Cast match_inds to int32 to halve memory (n_src < 2^31 in
        # practice) and keep keep as bool.
        saved_match_inds[ii] = match_inds.astype(np.int32, copy=False)
        saved_keep[ii] = keep

        mi_keep = match_inds[keep]
        arr_ra[mi_keep, ii] = crds.ra.deg[keep]
        arr_dec[mi_keep, ii] = crds.dec.deg[keep]
        arr_flux[mi_keep, ii] = tbl[flux_colname][keep]
        arr_fluxerr[mi_keep, ii] = tbl[flux_error_colname][keep]
        print(f"P1 {ii}: Added {keep.sum()} of {len(keep)} sources from exposure "
              f"{tbl.meta['exposure']} {tbl.meta['MODULE'] if 'MODULE' in tbl.meta else ''} [total={n_src}]",
              flush=True)

    print("Phase 1 stack done; computing mask / weights / position averages", flush=True)
    nmatch = np.isfinite(arr_flux).sum(axis=1).astype(np.int32)

    # Sigma-clip flux and positions to identify per-source outliers.
    clip_flux = sigma_clip(arr_flux, stdfunc='mad_std', axis=1)
    clip_ra = sigma_clip(arr_ra, stdfunc='mad_std', axis=1)
    clip_dec = sigma_clip(arr_dec, stdfunc='mad_std', axis=1)
    to_mask = clip_flux.mask | clip_ra.mask | clip_dec.mask
    keepmask = ~to_mask
    nmatch_good = keepmask.sum(axis=1).astype(np.int32)

    # free clip objects (they reference the big arrays)
    del clip_flux, clip_ra, clip_dec

    # weights: inverse-variance flux weighting, zeroed where masked
    weights = (1.0 / (arr_fluxerr**2)) * keepmask

    # position averages
    avg_ra = nanaverage(arr_ra, axis=1, weights=weights)
    avg_dec = nanaverage(arr_dec, axis=1, weights=weights)
    std_ra = nanaverage((arr_ra - avg_ra[:, None])**2, weights=weights, axis=1)**0.5
    std_dec = nanaverage((arr_dec - avg_dec[:, None])**2, weights=weights, axis=1)**0.5
    avgpos = SkyCoord(avg_ra, avg_dec, unit=(u.deg, u.deg), frame='icrs')

    # free ra/dec arrays -- no longer needed
    del arr_ra, arr_dec

    # flux and flux_err reductions
    flux_avg = nanaverage(arr_flux, weights=weights, axis=1)
    std_flux_avg = nanaverage((arr_flux - flux_avg[:, None])**2, weights=weights, axis=1)**0.5
    flux_err_avg = nanaverage(arr_fluxerr, weights=weights, axis=1)
    std_flux_err_avg = nanaverage((arr_fluxerr - flux_err_avg[:, None])**2, weights=weights, axis=1)**0.5
    flux_err_prop = (np.nansum(arr_fluxerr**2 * weights, axis=1)
                     / np.nansum(weights, axis=1))**0.5

    # free phase-1 big arrays (keep weights / keepmask -- needed in Phase 2)
    del arr_flux, arr_fluxerr

    # Build newtbl with per-source columns
    newtbl = Table()
    newtbl.meta = dict(tbls[0].meta)
    newtbl.meta['offsets'] = {tbl.meta['exposure']: (tbl.meta['ra_offset'], tbl.meta['dec_offset'])
                              for tbl in tbls}
    newtbl['skycoord_avg'] = avgpos
    newtbl['std_ra'] = std_ra
    newtbl['std_dec'] = std_dec
    newtbl['nmatch'] = nmatch
    newtbl['nmatch_good'] = nmatch_good
    newtbl[f'{flux_colname}_avg'] = flux_avg
    newtbl[f'std_{flux_colname}_avg'] = std_flux_avg
    newtbl[f'{flux_error_colname}_avg'] = flux_err_avg
    newtbl[f'std_{flux_error_colname}_avg'] = std_flux_err_avg
    newtbl[f'{flux_error_colname}_prop'] = flux_err_prop
    newtbl.meta[f'{flux_error_colname}_prop'] = 'propagated uncertainty on flux = 1/sum(weights)'

    # Phase 2: stream the remaining columns one at a time.
    # ra/dec are skipped because their summaries are already in newtbl
    # (skycoord_avg, std_ra, std_dec).  flux and flux_err are already done.
    already_done = {flux_colname, flux_error_colname, 'ra', 'dec',
                    'skycoord', skycoord_colname}
    print("Phase 2: streaming remaining columns one at a time", flush=True)
    for key in column_names:
        if key in already_done:
            continue
        if key not in tbls[0].colnames:
            print(f"  Skipping {key} (not in tbls[0])", flush=True)
            continue
        print(f"  Phase 2: streaming column {key}", flush=True)
        arr = np.full((n_src, n_tbl), np.nan, dtype='float32')
        for ii, tbl in enumerate(tbls):
            if key not in tbl.colnames:
                continue
            keep = saved_keep[ii]
            mi = saved_match_inds[ii]
            arr[mi[keep], ii] = tbl[key][keep]
        key_avg = nanaverage(arr, weights=weights, axis=1)
        std_key = nanaverage((arr - key_avg[:, None])**2, weights=weights, axis=1)**0.5
        newtbl[f'{key}_avg'] = key_avg
        newtbl[f'std_{key}_avg'] = std_key
        del arr

    # weights, keepmask, saved_match_inds, saved_keep kept until function
    # return; Python will free them after caller drops newtbl reference.
    return newtbl


def merge_catalogs(tbls, catalog_type='crowdsource', module='nrca',
                   ref_filter='f405n',
                   epsf=False, bgsub=False, desat=False, blur=False,
                   max_offset=0.10 * u.arcsec, target='brick',
                   indivexp=False,
                   qfcut=None, fracfluxcut=None,
                   min_nmatch_narrow=4,
                   basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    print(f'Starting merge catalogs: catalog_type: {catalog_type} module: {module} target: {target}', flush=True)

    epsf_ = "_epsf" if epsf else ""
    blur_ = "_blur" if blur else ""

    matching_ref_tables = [tb for tb in tbls if tb.meta['filter'] == ref_filter]
    if len(matching_ref_tables) == 0:
        ref_filter = tbls[0].meta['filter']
        print(f"Requested ref_filter not found; using fallback ref_filter={ref_filter}")
        matching_ref_tables = [tb for tb in tbls if tb.meta['filter'] == ref_filter]
    basetable = matching_ref_tables[0].copy()
    basetable.meta['astrometric_reference_wavelength'] = ref_filter

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    desat = "_unsatstar" if desat else ""
    bgsub = '_bgsub' if bgsub else ''

    reffiltercol = [ref_filter] * len(basetable)
    print(f"Started with {len(basetable)} in filter {ref_filter}", flush=True)

    # build up a reference coordinate catalog by adding in those with no matches each time
    basecrds = basetable['skycoord']
    for tb in tqdm(tbls, desc='Table Meta Loop'):
        if tb.meta['filter'] == ref_filter:
            continue
        crds = tb['skycoord']

        matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_matches, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)

        mutual_matches = (reverse_matches[matches] == np.arange(len(matches)))

        newcrds = crds[(sep > max_offset) | (~mutual_matches)]
        basecrds = SkyCoord([basecrds, newcrds])

        reffiltercol += [tb.meta['filter']] * len(newcrds)
        print(f"Added {len(newcrds)} new sources in filter {tb.meta['filter']}", flush=True)
    print(f"Base coordinate length = {len(basecrds)}", flush=True)

    basetable = Table()
    basetable['skycoord_ref'] = basecrds
    basetable['skycoord_ref_filtername'] = reffiltercol

    # flag_near_saturated(basetable, filtername=ref_filter)
    # # replace_saturated adds more rows
    # replace_saturated(basetable, filtername=ref_filter)
    # print(f"filter {basetable.meta['filter']} has {len(basetable)} rows")

    meta = {}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # for colname in basetable.colnames:
        #     basetable.rename_column(colname, colname+"_"+basetable.meta['filter'])

        for tbl in tqdm(tbls, desc='Table Loop'):
            t0 = time.time()
            wl = tbl.meta['filter']
            flag_near_saturated(tbl, filtername=wl, target=target, basepath=basepath)
            # replace_saturated adds more rows
            replace_saturated(tbl, filtername=wl, target=target, basepath=basepath)
            # DEBUG print(f"DEBUG: tbl['replaced_saturated'].sum(): {tbl['replaced_saturated'].sum()}")

            crds = tbl['skycoord']
            matches, sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
            reverse_matches, reverse_sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)

            mutual_matches = (reverse_matches[matches] == np.arange(len(matches)))
            # limit to one-to-one nearest neighbor matches
            # matches = matches[mutual_matches]
            # sep = sep[mutual_matches]

            print(f"filter {wl} has {len(tbl)} rows.  {mutual_matches.sum()} of {len(tbl)} are mutual.  Matching took {time.time()-t0:0.1f} seconds", flush=True)

            # removed Jan 21, 2023 because this *should* be handled by the pipeline now
            # # do one iteration of bulk offset measurement
            # radiff = (crds.ra[matches]-basecrds.ra).to(u.arcsec)
            # decdiff = (crds.dec[matches]-basecrds.dec).to(u.arcsec)
            # oksep = sep < max_offset
            # medsep_ra, medsep_dec = np.median(radiff[oksep]), np.median(decdiff[oksep])
            # tbl.meta[f'ra_offset_from_{ref_filter}'] = medsep_ra
            # tbl.meta[f'dec_offset_from_{ref_filter}'] = medsep_dec
            # newcrds = SkyCoord(crds.ra - medsep_ra, crds.dec - medsep_dec, frame=crds.frame)
            # tbl['skycoord'] = newcrds
            # matches, sep, _ = basecrds.match_to_catalog_sky(newcrds, nthneighbor=1)

            basetable.add_column(name=f"sep_{wl}", col=sep)
            basetable.add_column(name=f"id_{wl}", col=matches)
            matchtb = tbl[matches]
            badsep = sep > max_offset
            for cn in matchtb.colnames:
                if isinstance(matchtb[cn], SkyCoord):
                    matchtb.rename_column(cn, f"{cn}_{wl}")
                    matchtb[f'mask_{wl}'] = badsep
                else:
                    matchtb[f'{cn}_{wl}'] = MaskedColumn(data=matchtb[cn], name=f'{cn}_{wl}')
                    matchtb[f'{cn}_{wl}'].mask[badsep] = True
                    # mask non-mutual matches
                    matchtb[f'{cn}_{wl}'].mask[~mutual_matches] = True
                    if hasattr(matchtb[cn], 'meta'):
                        matchtb[f'{cn}_{wl}'].meta = matchtb[cn].meta
                    matchtb.remove_column(cn)

            print(f"Max flux in tbl for {wl}: {tbl['flux'].max()}; in jy={np.nanmax(np.array(tbl['flux_jy']))}; mag={np.nanmin(np.array(tbl['mag_ab']))}")
            print(f"merging tables step: max flux for {wl} is {matchtb['flux_'+wl].max()} {matchtb['flux_jy_'+wl].max()} {matchtb['mag_ab_'+wl].min()}")
            print(f"Basetable has length {len(basetable)} and ncols={len(basetable.colnames)} before stack")

            basetable = table.hstack([basetable, matchtb], join_type='exact')
            meta[f'{wl[1:-1]}pxdg'.upper()] = tbl.meta['pixelscale_deg2']
            meta[f'{wl[1:-1]}pxas'.upper()] = tbl.meta['pixelscale_arcsec']
            for key in tbl.meta:
                if isinstance(tbl.meta[key], (str, int, float)):
                    meta[f'{wl[1:-1]}{key[:4]}'.upper()] = tbl.meta[key]
                else:
                    # specifically to handle the case of astropy.io.fits.card objects that are unserializable
                    meta[f'{wl[1:-1]}{key[:4]}'.upper()] = str(tbl.meta[key])

            print(f"Basetable has length {len(basetable)} and ncols={len(basetable.colnames)} after stack")
            print(f"merging tables step: max flux for {wl} in merged table is {basetable['flux_'+wl].max()}"
                  f" {np.nanmax(np.array(basetable['flux_jy_'+wl]))} {np.nanmin(np.array(basetable['mag_ab_'+wl]))}")
            # DEBUG
            # DEBUG if hasattr(basetable[f'{cn}_{wl}'], 'mask'):
            # DEBUG     print(f"Table has mask sum for column {cn} {basetable[cn+'_'+wl].mask.sum()}")
            # DEBUG if 'replaced_saturated_f410m' in basetable.colnames:
            # DEBUG     print(f"'replaced_saturated_f410m' has {basetable['replaced_saturated_f410m'].sum()}")
            # There can be more stars in replaced_saturated_f410m than there were stars replaced because
            # there can be multiple stars in the merged coordinate list whose closest match is a saturated
            # star.  i.e., there could be two coordinates that both see the same F410M flux.

            bad = np.isnan(tbl['mag_ab']) & (tbl['flux'] > 0)
            if any(bad):
                raise ValueError("Bad magnitudes for good fluxes")

            print(f"Flagged {tbl[f'near_saturated_{wl}'].sum()} stars that are near saturated stars "
                  f"in filter {wl} out of {len(tbl)}.  "
                  f"There are then {basetable[f'near_saturated_{wl}_{wl}'].sum()} in the merged table.  "
                  f"There are also {basetable[f'replaced_saturated_{wl}'].sum()} replaced saturated.", flush=True)

        print(f"Stacked all rows into table with len={len(basetable)}", flush=True)
        if 'flux_jy_f410m' in basetable.colnames and 'flux_jy_f405n' in basetable.colnames:
            zeropoint410 = u.Quantity(jfilts.loc['JWST/NIRCam.F410M']['ZeroPoint'], u.Jy)
            zeropoint405 = u.Quantity(jfilts.loc['JWST/NIRCam.F405N']['ZeroPoint'], u.Jy)

            # Line-subtract the F410 continuum band
            # 0.16 is from BrA_separation
            # 0.196 is the 'post-destreak' version, which might (?) be better
            # 0.11 is the theoretical version from RecombFilterDifferencing
            # 0.16 still looks like the best; 0.175ish is the median, but 0.16ish is the mode
            # but we use 0.11, the theoretica one, because we don't necessarily expect a good match!
            f405to410_scale = 0.11
            basetable.add_column(basetable['flux_jy_f410m'] - basetable['flux_jy_f405n'] * f405to410_scale, name='flux_jy_410m405')

            basetable.add_column(-2.5*np.log10(basetable['flux_jy_410m405']) + ABMAG_OFFSET, name='mag_ab_410m405')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_410m405'] / zeropoint410), name='mag_vega_410m405')
            # Then subtract that remainder back from the F405 band to get the continuum-subtracted F405
            basetable.add_column(basetable['flux_jy_f405n'] - basetable['flux_jy_410m405'], name='flux_jy_405m410')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_405m410']) + ABMAG_OFFSET, name='mag_ab_405m410')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_405m410'] / zeropoint405), name='mag_vega_405m410')

        if 'flux_jy_f182m' in basetable.colnames and 'flux_jy_f187n' in basetable.colnames:
            zeropoint182 = u.Quantity(jfilts.loc['JWST/NIRCam.F182M']['ZeroPoint'], u.Jy)
            zeropoint187 = u.Quantity(jfilts.loc['JWST/NIRCam.F187N']['ZeroPoint'], u.Jy)

            # Line-subtract the F182 continuum band
            # 0.11 is the theoretical bandwidth fraction
            # PaA_separation_nrcb gives 0.175ish -> 0.183 with "latest"
            # 0.18 is closer to the histogram mode
            f187to182_scale = 0.11
            basetable.add_column(basetable['flux_jy_f182m'] - basetable['flux_jy_f187n'] * f187to182_scale, name='flux_jy_182m187')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_182m187']) + ABMAG_OFFSET, name='mag_ab_182m187')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_182m187'] / zeropoint182), name='mag_vega_182m187')
            # Then subtract that remainder back from the F187 band to get the continuum-subtracted F187
            basetable.add_column(basetable['flux_jy_f187n'] - basetable['flux_jy_182m187'], name='flux_jy_187m182')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_187m182']) + ABMAG_OFFSET, name='mag_ab_187m182')
            basetable.add_column(-2.5*np.log10(basetable['flux_jy_187m182'] / zeropoint187), name='mag_vega_187m182')
        """ # this adds to the file size too much
        # Add some important colors

        colors=[('f410m', 'f466n'),
                ('f405n', 'f410m'),
                ('f405n', 'f466n'),
                ('f187n', 'f182m', ),
                ('f182m', 'f410m'),
                ('f182m', 'f212n', ),
                ('f187n', 'f405n'),
                ('f187n', 'f212n'),
                #('f212n', '410m405'), no emag defined
                ('f212n', 'f410m'),
                #('182m187', '410m405'), no emag defined
                ('f356w', 'f444w'),
                ('f356w', 'f410m'),
                ('f410m', 'f444w'),
                ('f405n', 'f444w'),
                ('f444w', 'f466n'),
                ('f200w', 'f356w'),
                ('f200w', 'f212n'),
                ('f182m', 'f200w'),
                ('f115w', 'f182m'),
                ('f115w', 'f212n'),
                ('f115w', 'f200w'),
            ]
        for c1, c2 in colors:
            if f'mag_ab_{c1}' in basetable.colnames and f'mag_ab_{c2}' in basetable.colnames:
                basetable.add_column(basetable[f'mag_ab_{c1}']-basetable[f'mag_ab_{c2}'], name=f'color_{c1}-{c2}')
                basetable.add_column((basetable[f'emag_ab_{c1}']**2 + basetable[f'emag_ab_{c2}']**2)**0.5, name=f'ecolor_{c1}-{c2}')
        """

        # DEBUG for colname in basetable.colnames:
        # DEBUG     print(f"colname {colname} has mask: {hasattr(basetable[colname], 'mask')}")
        basetable.meta = meta
        if '212PXDG' not in meta:
            print("WARNING: 212PXDG not present in metadata for this target")

        indivexp = '_indivexp' if indivexp else ''
        tablename = f"{basepath}/catalogs/{catalog_type}_{module}{indivexp}_photometry_tables_merged{desat}{bgsub}{epsf_}{blur_}"
        t0 = time.time()
        print(f"Writing table {tablename} with len={len(basetable)} and ncols={len(basetable.colnames)}", flush=True)
        # use caps b/c FITS will force it to caps anyway
        basetable.meta['VERSION'] = datetime.datetime.now().isoformat()
        # FITS can mishandle masked bool columns; force saturated flags to plain bool.
        for colname in basetable.colnames:
            if 'saturated' in colname:
                col = basetable[colname]
                if hasattr(col, 'mask'):
                    fixed = np.array(col.filled(False), dtype=bool)
                else:
                    arr = np.array(col)
                    if arr.dtype.kind in 'fc':
                        fixed = np.isfinite(arr) & (arr != 0)
                    else:
                        fixed = arr.astype(bool)
                basetable.replace_column(colname, Column(fixed, name=colname))
        # DO NOT USE FITS in production, it drops critical metadata
        # I wish I had noted *what* metadata it drops, though, since I still seem to be using
        # it in production code down the line...
        # OH, I think the FITS file turns "True" into "False"?
        # Yes, specifically: it DROPS masked data types, converting "masked" into "True"?
        basetable.write(f"{tablename}.fits", overwrite=True)
        print(f"Done writing table {tablename}.fits in {time.time()-t0:0.1f} seconds", flush=True)

        # strip out bad metadata that the yaml serializer can't handle
        for k, v in basetable.meta.items():
            # Check for astropy.io.fits.card.Undefined objects explicitly
            try:
                yaml.dump({k: v}, )
            except Exception as ex:
                if isinstance(v, fits.card.Undefined):
                    basetable.meta[k] = str(v)
                    print("BAD META (Undefined):", k, type(v), v)
                    continue
                else:
                    basetable.meta[k] = str(v)
                    print("BAD META:", k, type(v), v, ex)

        t0 = time.time()
        # takes FOR-EV-ER
        try:
            basetable.write(f"{tablename}.ecsv", overwrite=True)
        except yaml.representer.RepresenterError:
            import astropy
            print("astropy version: ", astropy.__version__)
            # https://github.com/astropy/astropy/pull/18677 ?
            # DEBUG
            print("YAML RepresenterError: trying again after removing masks")
            # for colname in basetable.colnames:
            #     print("DEBUG Column: ", colname, type(basetable[colname]), hasattr(basetable[colname], 'mask'))
            # for key in basetable.meta:
            #     print("DEBUG Meta: ", key, type(basetable.meta[key]), basetable.meta[key])
            raise
        print(f"Done writing table {tablename}.ecsv in {time.time()-t0:0.1f} seconds", flush=True)

        # keep any rows with at least two qf cut pass
        if qfcut is not None:
            qfkeep = np.array([basetable[qfkey] > qfcut for qfkey in basetable.colnames if 'qf' in qfkey]).sum(axis=0) > 1
            print(f"Keeping {qfkeep.sum()} sources of {len(basetable)} with qf > {qfcut}")
            basetable = basetable[qfkeep]

        if fracfluxcut is not None:
            fracfluxkeep = np.array([basetable[fracfluxkey] > fracfluxcut for fracfluxkey in basetable.colnames if 'fracflux' in fracfluxkey]).sum(axis=0) > 1
            print(f"Keeping {fracfluxkeep.sum()} sources of {len(basetable)} with fracflux > {fracfluxcut}")
            basetable = basetable[fracfluxkeep]

        if min_nmatch_narrow is not None:
            available_nmatch_cols = [f'nmatch_good_{filn}' for filn in filternames_narrow if f'nmatch_good_{filn}' in basetable.colnames]
            if len(available_nmatch_cols) >= 2:
                match_brick_narrow = np.array([basetable[colname] > min_nmatch_narrow for colname in available_nmatch_cols]).sum(axis=0) > 1
                print(f"Keeping {match_brick_narrow.sum()} sources of {len(basetable)} with at least {min_nmatch_narrow} matches in two or more of the narrower bands")
                basetable = basetable[match_brick_narrow]
            else:
                print("Skipping narrow-band nmatch cut: insufficient narrow-band columns for this target")

        if qfcut is not None or fracfluxcut is not None:
            print(f"Saving merged version with qualcuts: {tablename}_qualcuts.fits with len={len(basetable)}")
            basetable.write(f"{tablename}_qualcuts.fits", overwrite=True)

        sep_cols = [f'sep_{filtername}' for filtername in filternames if 'w' not in filtername and f'sep_{filtername}' in basetable.colnames]
        if len(sep_cols) >= 2:
            oksep = (np.array([basetable[colname] < 0.1*u.arcsec for colname in sep_cols]).sum(axis=0) > 1)
        else:
            oksep = np.ones(len(basetable), dtype=bool)
            print("Skipping oksep cut: insufficient sep_* columns for this target")
        print(f"Writing {tablename}_qualcuts_oksep2221.fits")
        basetable[oksep].write(f"{tablename}_qualcuts_oksep2221.fits", overwrite=True)


def merge_individual_frames(module='merged', suffix="", desat=False, filtername='f410m',
                                        progid='2221',
                                        bgsub=False, epsf=False, fitpsf=False, blur=False, target='brick',
                                        exposure_numbers=np.arange(1, 25),
                                        max_visitid=10,
                                        method='crowdsource',
                                        offsets_table=None,
                                        iteration_label=None,
                                        basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):

    desat = "_unsatstar" if desat else ""
    bgsub = '_bgsub' if bgsub else ''
    fitpsf = '_fitpsf' if fitpsf else ''
    blur_ = "_blur" if blur else ""
    # iter_token is inserted *between* the {blur_} block and the
    # {method_suffix} so the per-frame filename
    # ``..._{blur_}{iter_token}_{method_suffix}{suffix}.fits``
    # matches what crowdsource_catalogs_long.py writes for iter2/iter3.
    if iteration_label in (None, ''):
        iter_token = ''
    elif str(iteration_label).startswith('_'):
        iter_token = str(iteration_label)
    else:
        iter_token = f'_{iteration_label}'
    if module == 'merged':
        modules = ['nrca', 'nrcb',]
        modules += [f'nrc{ab}{n}' for ab in 'ab' for n in range(1, 5)]
        modules += ['nrcalong', 'nrcblong']  # sgrb2 LW modules
    elif module in ('nrca', 'nrcb'):
        # Exposure-level catalogs are often saved with detector-qualified module names.
        modules = [module] + [f'{module}{n}' for n in range(1, 5)]
    else:
        modules = (module, )

    if method == 'crowdsource':
        flux_error_colname = 'dflux'
        column_names = ('flux', flux_error_colname, 'skycoord', 'qf', 'rchi2', 'fracflux', 'fwhm', 'fluxiso', 'spread_model')
        method_suffix = 'crowdsource'
        # flux_colname = 'flux_fit'
    elif method in ('dao', 'daophot', 'basic', 'daobasic', 'iterative', 'daoiterative'):
        flux_error_colname = 'flux_err'
        column_names = ('flux_fit', flux_error_colname, 'skycoord', 'qfit', 'cfit', 'flux_init', 'flags', 'local_bkg', 'iter_detected', 'group_size')
        # flux_colname = 'flux'
        method_suffix = 'daophot'
    else:
        raise ValueError(f"Method must be dao or crowdsource but was {method}")

    tblfns = [x
              for module_ in modules
              for progid in obs_filters[target]
              for visitid in range(1, max_visitid+1)
              for exposure in exposure_numbers
              for x in glob.glob(f"{basepath}/{filtername.upper()}/"
                                 f"{filtername.lower()}_{module_}_visit{visitid:03d}_vgroup*_exp{exposure:05d}{desat}{bgsub}{fitpsf}{blur_}{iter_token}"
                                 f"_{method_suffix}{suffix}.fits")
              ]
    tblfns = sorted(set(tblfns))
    print(f"Found {len(tblfns)} tables for {filtername.lower()}_*_visit*_exp*{desat}{bgsub}{fitpsf}{blur_}:")
    print("\n".join(tblfns))

    if len(tblfns) == 0:
        raise ValueError(f"No tables found matching {basepath}/{filtername.upper()}/{filtername.lower()}_{module}....{desat}{bgsub}{fitpsf}{blur_}_{method}{suffix}.fits")

    tables = [Table.read(fn) for fn in tblfns]
    for tb, fn in zip(tables, tblfns):
        if 'exposure' not in tb.meta:
            tb.meta['exposure'] = fn.split("_exp")[-1][:5]
        if 'FILENAME' not in tb.meta:
            print('tb.meta:', tb.meta)
            raise ValueError(f"Table file {fn} is not correctly formatted; it is missing FILENAME metadata")

    # Note (2026-04-23): an earlier "dedup bug" diagnosis (realign=True +
    # min_offset=0.25") was based on analysing the FIRST 200k rows of the
    # F200W brick catalog which are exposure-adjacent and therefore
    # artificially correlated.  A random 500k-row sample found only
    # ~0.1% definite duplicates (ratio>0.95 AND nmatch_sum>24 pigeonhole);
    # median flux ratio of near-neighbour pairs was 0.38, indicating
    # most "near-neighbours" are genuinely distinct close-pair sources in
    # a dense field.  min_offset=0.25" would actively mis-merge ~36% of
    # real neighbours (separations 0.10-0.15").  Keeping the historical
    # defaults.
    merged_exposure_table = combine_singleframe(tables, offsets_table=offsets_table)

    outfn = f"{basepath}/catalogs/{filtername.lower()}_{module}_indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_{method}{suffix}_allcols.fits"
    print(f"Writing {outfn} with length {len(merged_exposure_table)}")
    merged_exposure_table.write(outfn, overwrite=True)

    # make a table that is nearly equivalent to standard tables (with no 'x' or 'y' coordinate)
    minimal_version = {colname: merged_exposure_table[f'{colname}_avg']
                       for colname in column_names if f'{colname}_avg' in merged_exposure_table.colnames}
    for key in ('dra_avg', 'ddec_avg', 'std_ra', 'std_dec', 'nmatch', 'nmatch_good', f'{flux_error_colname}_prop'):
        if key in merged_exposure_table.colnames:
            minimal_version[key.split("_avg")[0]] = merged_exposure_table[key]

    minimal_table = Table(minimal_version)
    minimal_table.meta = merged_exposure_table.meta.copy()
    for ii, fn in enumerate(tblfns):
        minimal_table.meta[f'fn{ii}'] = os.path.basename(fn)

    # Ensure saturated stars are represented in indivexp merged products too.
    replace_saturated(minimal_table, filtername=filtername, target=target, basepath=basepath)

    reject = np.isnan(minimal_table['skycoord'].ra) | np.isnan(minimal_table['skycoord'].dec)
    if np.any(reject):
        print(f"Rejected {reject.sum()} sources that had nan coordinates.")
        minimal_table = minimal_table[~reject]

    outfn = f"{basepath}/catalogs/{filtername.lower()}_{module}_indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_{method}{suffix}.fits"
    print(f"Final table length is {len(minimal_table)}.  Writing {outfn}")
    minimal_table.write(outfn, overwrite=True)

    for colname in minimal_table.colnames:
        assert minimal_table[colname].ndim == 1

    return minimal_table


def merge_crowdsource(module='nrca', suffix="", desat=False, bgsub=False,
                      epsf=False, fitpsf=False, blur=False, target='brick',
                      min_qf=0.75,
                      indivexp=False,
                      basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    if epsf:
        raise NotImplementedError
    print()
    print(f'Starting merge crowdsource module: {module} suffix: {suffix} target: {target}', flush=True)
    imgfns = [x
              for obsid in obs_filters[target]
              for filn in obs_filters[target][obsid]
              for x in glob.glob(f"{basepath}/{filn.upper()}/pipeline/"
                                 f"jw0{obsid}-o{project_obsnum[target][obsid]}_t001_nircam*{filn.lower()}*{module}_i2d.fits")
              if f'{module}_' in x or f'{module}1_' in x
             ]

    desat = "_unsatstar" if desat else ""
    bgsub = '_bgsub' if bgsub else ''
    fitpsf = '_fitpsf' if fitpsf else ''
    blur_ = "_blur" if blur else ""

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    filternames = [filn for obsid in obs_filters[target] for filn in obs_filters[target][obsid]]
    print(f"Merging filters {filternames}", flush=True)
    if indivexp:
        catfns = [x
                  for filn in filternames
                  for x in glob.glob(f"{basepath}/catalogs/{filn.lower()}*{module}*indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_crowdsource{suffix}.fits")
                  ]
        if len(catfns) == 0:
            filn = 'f405n'
            raise ValueError(f"{basepath}/catalogs/{filn.lower()}*{module}*indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_crowdsource{suffix}.fits had no matches")
        if len(catfns) != len(imgfns):
            print("WARNING: Different length of imgfns & catfns!")
            print("imgfns:", imgfns)
            print("catfns:", catfns)
            print(dict(zip(imgfns, catfns)))
            raise ValueError(f"{basepath}/catalogs/FILTER*{module}*obs*indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_crowdsource{suffix}.fits had different n(imgs) than n(cats)")
    else:
        catfns = [x
                  for filn in filternames
                  for x in glob.glob(f"{basepath}/{filn.upper()}/{filn.lower()}*{module}{desat}{bgsub}{fitpsf}{blur_}_crowdsource{suffix}.fits")
                  ]
        if target == 'brick' and len(catfns) != 10:
            raise ValueError(f"len(catfns) = {len(catfns)}.  catfns: {catfns}")
        elif target == 'cloudc' and len(catfns) != 6:
            raise ValueError(f"len(catfns) = {len(catfns)}.  catfns: {catfns}")

    for catfn in catfns:
        print(catfn, getmtime(catfn))

    # added a fq cut at read time to reduce memory usage during merge
    def read_cat(catfn, min_qf=min_qf):
        tbl = Table.read(catfn)
        if min_qf is not None:
            tbl = tbl[tbl['qf'] > min_qf]
        return tbl
    tbls = [read_cat(catfn) for catfn in tqdm(catfns, desc='Reading Tables')]

    for catfn, tbl in zip(catfns, tbls):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = os.path.basename(catfn).split("_")[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # wcses = [wcs.WCS(fits.getheader(fn.replace("_crowdsource", "_crowdsource_skymodel"))) for fn in catfns]
        # imgs = [fits.getdata(fn, ext=('SCI', 1)) for fn in imgfns]
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) for fn in imgfns]

    for tbl, ww in zip(tbls, wcses):
        # Now done in the original catalog making step tbl['y'],tbl['x'] = tbl['x'],tbl['y']
        if 'skycoord' not in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x'], tbl['y'])
            tbl.add_column(crds, name='skycoord')
        else:
            crds = tbl['skycoord']
        tbl.meta['pixelscale_deg2'] = ww.proj_plane_pixel_area()
        tbl.meta['pixelscale_arcsec'] = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        print(f'Calculating Flux [Jy].  fwhm={tbl["fwhm"].mean()}, pixscale={tbl.meta["pixelscale_arcsec"]}')
        # The 'flux' is the sum of pixels whose values are each in MJy/sr.
        # To get to the correct flux, we need to multiply by the pixel area in steradians to get to megaJanskys, which can be summed
        # That's it.  There's no need to account for the FWHM.  We only needed that if tbl['flux'] was the _peak_, but it's not.
        #flux_jy = (tbl['flux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        #eflux_jy = (tbl['dflux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        flux_jy = (tbl['flux'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
        eflux_jy = (tbl['dflux'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
        with np.errstate(all='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                filtername = tbl.meta["filter"]
                zeropoint = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername.upper()}']['ZeroPoint'], u.Jy)
                print(f"Zeropoint for {filtername} is {zeropoint}.  Max flux is {flux_jy.max()}")
                abmag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag
                abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux_jy) * u.mag
                tbl.add_column(Column(flux_jy, name='flux_jy', unit=u.Jy))
                tbl.add_column(Column(eflux_jy, name='eflux_jy', unit=u.Jy))
                tbl.add_column(Column(abmag, name='mag_ab', unit=u.mag))
                tbl.add_column(Column(abmag_err, name='emag_ab', unit=u.mag))
                print(f"Max flux={tbl['flux_jy'].max()}, min mag={np.nanmin(tbl['mag_ab'])}, median={np.nanmedian(tbl['mag_ab'])}")
        if hasattr(tbl['mag_ab'], 'mask'):
            print(f'ab mag tbl col has mask sum = {tbl["mag_ab"].mask.sum()} masked values')
        if hasattr(abmag, 'mask'):
            print(f'ab mag has mask sum = {abmag.mask.sum()} masked values')
        if hasattr(tbl['flux'], 'mask'):
            print(f'flux has mask sum = {tbl["flux"].mask.sum()} masked values')

    for tbl in tbls:
        try:
            sanity_check_individual_table(tbl)
        except Exception as ex:
            print(ex)
            print(tbl.meta)
            raise ex

    merge_catalogs(tbls,
                   catalog_type=f'crowdsource{suffix}{"_desat" if desat else ""}{"_bgsub" if bgsub else ""}',
                   module=module, bgsub=bgsub, desat=desat, epsf=epsf, target=target,
                   blur=blur,
                   indivexp=indivexp,
                   qfcut=0.9,
                   fracfluxcut=0.75,
                   basepath=basepath)


def merge_daophot(module='nrca', detector='', daophot_type='basic', desat=False, bgsub=False, epsf=False, blur=False, target='brick',
                  indivexp=False,
                  basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):

    desat = "_unsatstar" if desat else ""
    bgsub = '_bgsub' if bgsub else ''
    epsf_ = "_epsf" if epsf else ""
    blur_ = "_blur" if blur else ""

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    filternames = [filn for obsid in obs_filters[target] for filn in obs_filters[target][obsid]]
    print(f"Merging daophot {daophot_type}, {detector}, {module}, {desat}, {bgsub}, {epsf_}, {blur_}. filters {filternames}")

    # Use _project_for_target_filter rather than the global filter_to_project
    # so a filter shared across targets (e.g. f187n in both brick/2221 and
    # sgrb2/5365) resolves to the project matching this run's ``target``.
    imgfns = [x
              for filn in filternames
              for _proj in (_project_for_target_filter(target, filn),)
              for x in glob.glob(f"{basepath}/{filn.upper()}/pipeline/jw0{_proj}-o{project_obsnum[target][_proj]}_t001_nircam*{filn.lower()}*{module}_i2d.fits")
              if f'{module}_' in x or f'{module}1_' in x
             ]

    if indivexp:
        # Map daophot_type -> per-filter merge filename token written by
        # merge_individual_frames(): basic uses ``dao``, iterative uses
        # ``daoiterative`` (this matches the suffix-vs-method dict in
        # main()).  The pattern was hardcoded to ``_dao_{daophot_type}``
        # which only matched basic; the daoiterative filename is
        # ``_daoiterative_iterative.fits`` so the glob always returned
        # zero matches and the cross-filter daoiterative merge never ran.
        method_name = 'dao' if daophot_type == 'basic' else 'daoiterative'
        catfns = [x
                  for filn in filternames
                  for x in glob.glob(f"{basepath}/catalogs/{filn.lower()}*{module}*indivexp_merged{desat}{bgsub}{blur_}_{method_name}_{daophot_type}.fits")
                  ]
        if len(catfns) == 0:
            raise ValueError(f"{basepath}/catalogs/<filt>*{module}*indivexp_merged{desat}{bgsub}{blur_}_{method_name}_{daophot_type}.fits had no matches across filters {filternames}")
        if len(catfns) != len(imgfns):
            print("WARNING: Different length of imgfns & catfns!")
            print("imgfns:", imgfns)
            print("catfns:", catfns)
            print(dict(zip(imgfns, catfns)))
            # raise ValueError(f"{basepath}/catalogs/FILTER*{module}*obs*indivexp_merged{desat}{bgsub}{fitpsf}{blur_}_crowdsource{suffix}.fits had different n(imgs) than n(cats)")
    else:
        catfns = [
            f"{basepath}/{filtername.upper()}/{filtername.lower()}_{module}{detector}{desat}{bgsub}{epsf_}{blur_}_daophot_{daophot_type}.fits"
            for filtername in filternames
        ]

    tbls = [Table.read(catfn) for catfn in catfns]

    for catfn, tbl, filtername in zip(catfns, tbls, filternames):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = filtername

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # wcses = [wcs.WCS(fits.getheader(fn.replace("_crowdsource", "_crowdsource_skymodel"))) for fn in catfns]
        # imgs = [fits.getdata(fn, ext=('SCI', 1)) for fn in imgfns]
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) for fn in imgfns]

    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')

    for ii, tbl in enumerate(tbls):
        ww = wcses[ii] if ii < len(wcses) else None

        if ww is not None and 'x_fit' in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x_fit'], tbl['y_fit'])
        elif ww is not None and 'x_0' in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x_0'], tbl['y_0'])
        else:
            crds = tbl['skycoord']
        if 'skycoord' not in tbl.colnames:
            tbl.add_column(crds, name='skycoord')

        if ww is not None:
            pixelscale_deg2 = ww.proj_plane_pixel_area()
            pixelscale_arcsec = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        elif 'pixelscale_deg2' in tbl.meta:
            pixelscale_deg2 = tbl.meta['pixelscale_deg2']
            pixelscale_arcsec = (u.Quantity(pixelscale_deg2, u.deg**2)**0.5).to(u.arcsec)
        elif 'PIXSCALE' in tbl.meta:
            pixelscale_arcsec = u.Quantity(float(tbl.meta['PIXSCALE']), u.arcsec)
            pixelscale_deg2 = pixelscale_arcsec.to(u.deg)**2
        else:
            raise ValueError(f"Could not determine pixel scale for {tbl.meta.get('filter', 'unknown')} table {tbl.meta.get('filename', '')}")

        tbl.meta['pixelscale_deg2'] = pixelscale_deg2
        tbl.meta['pixelscale_arcsec'] = pixelscale_arcsec
        if 'flux_fit' in tbl.colnames:
            flux = tbl['flux_fit']
        elif 'flux_0' in tbl.colnames:
            flux = tbl['flux_0']
        elif 'flux' in tbl.colnames:
            flux = tbl['flux']
        else:
            raise KeyError(f"No supported flux column found in table columns={tbl.colnames}")
        filtername = tbl.meta['filter']

        row = fwhm_tbl[fwhm_tbl['Filter'] == filtername.upper()]
        fwhm = u.Quantity(float(row['PSF FWHM (arcsec)'][0]), u.arcsec)
        fwhm_pix = float(row['PSF FWHM (pixel)'][0])
        tbl.meta['fwhm_arcsec'] = fwhm
        tbl.meta['fwhm_pix'] = fwhm_pix

        with np.errstate(all='ignore'):
            flux_jy = (flux * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
            zeropoint = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername.upper()}']['ZeroPoint'], u.Jy)
            vegamag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag
            abmag = (-2.5 * np.log10(flux_jy / u.Jy) + ABMAG_OFFSET) * u.mag
            try:
                eflux_jy = (tbl['flux_unc'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
            except KeyError:
                eflux_jy = (tbl['flux_err'] * u.MJy/u.sr * tbl.meta['pixelscale_deg2']).to(u.Jy)
            abmag_err = 2.5 / np.log(10) * eflux_jy / flux_jy * u.mag
        tbl.add_column(Column(flux_jy, name='flux_jy', unit=u.Jy))
        tbl.add_column(Column(abmag, name='mag_ab', unit=u.mag))
        tbl.add_column(Column(vegamag, name='mag_vega', unit=u.mag))
        tbl.add_column(Column(eflux_jy, name='eflux_jy', unit=u.Jy))
        tbl.add_column(Column(abmag_err, name='emag_ab', unit=u.mag))

    for tbl in tbls:
        try:
            sanity_check_individual_table(tbl)
        except Exception as ex:
            print(ex)
            print(tbl.meta)
            raise ex

    merge_catalogs(tbls, catalog_type=daophot_type, module=module, bgsub=bgsub, desat=desat, epsf=epsf, target=target,
                   blur=blur, indivexp=indivexp,
                   basepath=basepath)


def _project_for_target_filter(target, filtername):
    """Return the project_id under which ``target`` observes ``filtername``.

    ``filter_to_project`` is a global dict that collapses filter->project
    across all targets, so for a filter observed by multiple targets (e.g.
    f187n appears under both brick/2221 and sgrb2/5365) it picks whichever
    target was iterated last and breaks lookups for the other targets.
    This helper resolves the correct project for the target in hand.
    """
    target_filters = obs_filters[target]
    filt_l = filtername.lower()
    for proj, filts in target_filters.items():
        if filt_l in filts:
            return proj
    raise KeyError(
        f'filter {filtername!r} not observed by target {target!r}; '
        f'known target/filter map: {target_filters}'
    )


def load_satstar_catalog(filtername, target='brick',
                         basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    proj = _project_for_target_filter(target, filtername)
    primary = (f'{basepath}/{filtername.upper()}/pipeline/'
               f'jw0{proj}-o{project_obsnum[target][proj]}'
               f'_t001_nircam_clear-{filtername}-merged_i2d_satstar_catalog.fits')
    if os.path.exists(primary):
        print(f"Using saturated star catalog {primary}")
        return Table.read(primary)

    fallback = sorted(glob.glob(f'{basepath}/{filtername.upper()}/pipeline/*satstar_catalog.fits'))
    if len(fallback) == 0:
        print(f"No saturated star catalog files found for {filtername} in {basepath}/{filtername.upper()}/pipeline")
        return None

    print(f"Using {len(fallback)} fallback saturated star catalogs for {filtername}")
    sat_tables = [Table.read(fn) for fn in fallback]
    return table.vstack(sat_tables, metadata_conflicts='silent')


def flag_near_saturated(cat, filtername, radius=None, target='brick',
                        basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    print(f"Flagging near saturated stars for filter {filtername}")
    satstar_cat = load_satstar_catalog(filtername, target=target, basepath=basepath)
    if satstar_cat is None:
        print(f"No saturated star catalog found for {filtername}")
        cat.add_column(np.zeros(len(cat), dtype='bool'), name=f'near_saturated_{filtername}')
        return
    satstar_coords = satstar_cat['skycoord_fit']

    cat_coords = cat['skycoord']

    if radius is None:
        # 0.55" flagging radius for every NIRCam filter in the project map;
        # keep the filter list in sync with obs_filters so shared code paths
        # (sickle/sgrb2/etc.) don't KeyError on filters that aren't listed.
        radius = {# short-wave (< ~2.5 um)
                  'f115w': 0.55*u.arcsec,
                  'f150w': 0.55*u.arcsec,
                  'f162m': 0.55*u.arcsec,
                  'f182m': 0.55*u.arcsec,
                  'f187n': 0.55*u.arcsec,
                  'f200w': 0.55*u.arcsec,
                  'f210m': 0.55*u.arcsec,
                  'f212n': 0.55*u.arcsec,
                  # long-wave (> ~2.5 um)
                  'f300m': 0.55*u.arcsec,
                  'f323n': 0.55*u.arcsec,
                  'f335m': 0.55*u.arcsec,
                  'f356w': 0.55*u.arcsec,
                  'f360m': 0.55*u.arcsec,
                  'f405n': 0.55*u.arcsec,
                  'f410m': 0.55*u.arcsec,
                  'f444w': 0.55*u.arcsec,
                  'f466n': 0.55*u.arcsec,
                  'f470n': 0.55*u.arcsec,
                  'f480m': 0.55*u.arcsec,
                  }[filtername]

    satfinite = np.isfinite(satstar_coords.ra.deg) & np.isfinite(satstar_coords.dec.deg)
    catfinite = np.isfinite(cat_coords.ra.deg) & np.isfinite(cat_coords.dec.deg)

    satstar_cat = satstar_cat[satfinite]
    satstar_coords = satstar_coords[satfinite]

    valid_cat_inds = np.where(catfinite)[0]
    if len(valid_cat_inds) > 0 and len(satstar_cat) > 0:
        idx_cat_sub, idx_sat, sep, _ = satstar_coords.search_around_sky(cat_coords[catfinite], radius)
        idx_cat = valid_cat_inds[idx_cat_sub]
    else:
        idx_cat = np.array([], dtype=int)
        idx_sat = np.array([], dtype=int)

    near_sat = np.zeros(len(cat), dtype='bool')
    near_sat[idx_cat] = True

    cat.add_column(near_sat, name=f'near_saturated_{filtername}')


def replace_saturated(cat, filtername, radius=None, target='brick',
                      basepath='/blue/adamginsburg/adamginsburg/jwst/brick/'):
    satstar_cat = load_satstar_catalog(filtername, target=target, basepath=basepath)
    if satstar_cat is None:
        print(f"No saturated star catalog found for {filtername}; skipping replacement")
        cat.add_column(np.zeros(len(cat), dtype='bool'), name='replaced_saturated')
        cat.add_column(np.zeros(len(cat), dtype='bool'), name='is_saturated')
        if 'flux_fit' in cat.colnames:
            cat.rename_column('flux_fit', 'flux')
        return

    print(f"Loaded saturated star catalog for {filtername} with {len(satstar_cat)} rows")
    satstar_coords = satstar_cat['skycoord_fit']

    cat_coords = cat['skycoord']

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')

    if radius is None:
        radius = {# short-wave (< ~2.5 um)
                  'f115w': 0.05*u.arcsec,
                  'f150w': 0.05*u.arcsec,
                  'f162m': 0.05*u.arcsec,
                  'f182m': 0.05*u.arcsec,
                  'f187n': 0.05*u.arcsec,
                  'f200w': 0.05*u.arcsec,
                  'f210m': 0.05*u.arcsec,
                  'f212n': 0.05*u.arcsec,
                  # long-wave (> ~2.5 um)
                  'f300m': 0.1*u.arcsec,
                  'f323n': 0.1*u.arcsec,
                  'f335m': 0.1*u.arcsec,
                  'f356w': 0.1*u.arcsec,
                  'f360m': 0.1*u.arcsec,
                  'f405n': 0.1*u.arcsec,
                  'f410m': 0.1*u.arcsec,
                  'f444w': 0.1*u.arcsec,
                  'f466n': 0.1*u.arcsec,
                  'f470n': 0.1*u.arcsec,
                  'f480m': 0.1*u.arcsec,
                  }[filtername]

    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    fwhm = u.Quantity(fwhm_tbl[fwhm_tbl['Filter'] == filtername.upper()]['PSF FWHM (arcsec)'], u.arcsec)

    filtername_meta = cat.meta.get('filter', filtername)
    zeropoint = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername_meta.upper()}']['ZeroPoint'], u.Jy)

    pixelscale_deg2 = cat.meta.get('pixelscale_deg2', None)
    if pixelscale_deg2 is not None:
        flux_jy = (satstar_cat['flux_fit'] * u.MJy/u.sr * pixelscale_deg2).to(u.Jy)
        try:
            eflux_jy = (satstar_cat['flux_err'] * u.MJy/u.sr * pixelscale_deg2).to(u.Jy)
        except KeyError:
            eflux_jy = (satstar_cat['flux_unc'] * u.MJy/u.sr * pixelscale_deg2).to(u.Jy)
        abmag = (-2.5*np.log10(flux_jy / u.Jy) + ABMAG_OFFSET) * u.mag
        abvega = -2.5*np.log10(flux_jy / zeropoint) * u.mag
        abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux_jy) * u.mag
    else:
        print(f"Catalog metadata lacks pixelscale_deg2 for {filtername}; skipping satstar mag conversion")
        abmag = np.full(len(satstar_cat), np.nan) * u.mag
        abvega = np.full(len(satstar_cat), np.nan) * u.mag
        abmag_err = np.full(len(satstar_cat), np.nan) * u.mag
    satstar_cat['mag_ab'] = abmag
    satstar_cat['mag_vega'] = abvega
    satstar_cat['emag_ab'] = abmag_err

    satfinite = np.isfinite(satstar_coords.ra.deg) & np.isfinite(satstar_coords.dec.deg)
    catfinite = np.isfinite(cat_coords.ra.deg) & np.isfinite(cat_coords.dec.deg)

    satstar_cat = satstar_cat[satfinite]
    satstar_coords = satstar_coords[satfinite]

    valid_cat_inds = np.where(catfinite)[0]
    if len(valid_cat_inds) > 0 and len(satstar_cat) > 0:
        idx_cat_sub, idx_sat, sep, _ = satstar_coords.search_around_sky(cat_coords[catfinite], radius)
        idx_cat = valid_cat_inds[idx_cat_sub]
    else:
        idx_cat = np.array([], dtype=int)
        idx_sat = np.array([], dtype=int)

    replaced_sat = np.zeros(len(cat), dtype='bool')
    replaced_sat[idx_cat] = True

    if 'flux_err' in satstar_cat.colnames:
        flux_err_colname = 'flux_err'
        xerr_colname = 'x_err'
        yerr_colname = 'y_err'
    elif 'flux_unc' in satstar_cat.colnames:
        flux_err_colname = 'flux_unc'
        xerr_colname = 'x_0_unc'
        yerr_colname = 'y_0_unc'
    else:
        print(satstar_cat.colnames)
        raise KeyError("Missing flux error column")

    if 'flux' in cat.colnames:
        if 'dflux' in cat.colnames:
            cat_fluxerr_col = 'dflux'
        elif 'flux_err' in cat.colnames:
            cat_fluxerr_col = 'flux_err'
        elif 'eflux' in cat.colnames:
            cat_fluxerr_col = 'eflux'
        else:
            cat_fluxerr_col = None

        cat['flux'][idx_cat] = satstar_cat['flux_fit'][idx_sat]
        if cat_fluxerr_col is not None:
            cat[cat_fluxerr_col][idx_cat] = satstar_cat[flux_err_colname][idx_sat]
        cat['skycoord'][idx_cat] = satstar_cat['skycoord_fit'][idx_sat]
        if 'x' in cat.colnames:
            # the merged, individual field catalogs don't have these
            cat['x'][idx_cat] = satstar_cat['x_fit'][idx_sat]
            cat['y'][idx_cat] = satstar_cat['y_fit'][idx_sat]
            cat['dx'][idx_cat] = satstar_cat[xerr_colname][idx_sat]
            cat['dy'][idx_cat] = satstar_cat[yerr_colname][idx_sat]

        if 'mag_ab' in cat.colnames:
            cat['mag_ab'][idx_cat] = abmag[idx_sat]
        if 'mag_vega' in cat.colnames:
            cat['mag_vega'][idx_cat] = abvega[idx_sat]
        if 'emag_ab' in cat.colnames:
            cat['emag_ab'][idx_cat] = abmag_err[idx_sat]

        # ID the stars that are saturated-only (not INCluded in the orig cat)
        satstar_not_inc = np.ones(len(satstar_cat), dtype='bool')
        satstar_not_inc[idx_sat] = False
        satstar_toadd = satstar_cat[satstar_not_inc]

        satstar_toadd.rename_column('flux_fit', 'flux')
        if cat_fluxerr_col is not None:
            satstar_toadd.rename_column(flux_err_colname, cat_fluxerr_col)
        satstar_toadd.rename_column('skycoord_fit', 'skycoord')
        if 'x' in cat.colnames:
            satstar_toadd.rename_column('x_fit', 'x')
            satstar_toadd.rename_column('y_fit', 'y')
            satstar_toadd.rename_column(xerr_colname, 'dx')
            satstar_toadd.rename_column(yerr_colname, 'dy')

        for colname in cat.colnames:
            if colname not in satstar_toadd.colnames:
                satstar_toadd.add_column(np.ones(len(satstar_toadd)) * np.nan, name=colname)
        for colname in satstar_toadd.colnames:
            if colname not in cat.colnames:
                satstar_toadd.remove_column(colname)

        for row in satstar_toadd:
            cat.add_row(dict(row))

    elif 'flux_fit' in cat.colnames:
        # DAOPHOT
        cat['flux_fit'][idx_cat] = satstar_cat['flux_fit'][idx_sat]
        cat['flux_err'][idx_cat] = satstar_cat[flux_err_colname][idx_sat]
        cat['skycoord'][idx_cat] = satstar_cat['skycoord_fit'][idx_sat]
        if 'x_fit' in satstar_cat.colnames and 'x_fit' in cat.colnames:
            cat['x_fit'][idx_cat] = satstar_cat['x_fit'][idx_sat]
            cat['y_fit'][idx_cat] = satstar_cat['y_fit'][idx_sat]
            cat['x_err'][idx_cat] = satstar_cat[xerr_colname][idx_sat]
            cat['y_err'][idx_cat] = satstar_cat[yerr_colname][idx_sat]

        if 'mag_ab' in cat.colnames:
            cat['mag_ab'][idx_cat] = abmag[idx_sat]
        if 'mag_vega' in cat.colnames:
            cat['mag_vega'][idx_cat] = abvega[idx_sat]
        if 'emag_ab' in cat.colnames:
            cat['emag_ab'][idx_cat] = abmag_err[idx_sat]

        # ID the stars that are saturated-only (not INCluded in the orig cat)
        satstar_not_inc = np.ones(len(satstar_cat), dtype='bool')
        satstar_not_inc[idx_sat] = False
        satstar_toadd = satstar_cat[satstar_not_inc]

        satstar_toadd.rename_column('skycoord_fit', 'skycoord')
        satstar_toadd['skycoord_centroid'] = satstar_toadd['skycoord']

        for colname in cat.colnames:
            if colname not in satstar_toadd.colnames:
                satstar_toadd.add_column(np.ones(len(satstar_toadd))*np.nan, name=colname)
        for colname in satstar_toadd.colnames:
            if colname not in cat.colnames:
                satstar_toadd.remove_column(colname)

        #print("cat colnames: ",cat.colnames)
        #print("satstar toadd_colnames: ",satstar_toadd.colnames)
        for row in satstar_toadd:
            cat.add_row(dict(row))

    # we've added on more rows that are all 'replaced_sat'
    replaced_sat_ = np.ones(len(cat), dtype='bool')
    replaced_sat_[:len(replaced_sat)] = replaced_sat

    print(f"Replacing {len(idx_cat)} stars that are saturated of {len(cat)} "
          f"in filter {filtername}.  "
          f"{satstar_not_inc.sum()} are newly added.  The total replaced stars={replaced_sat_.sum()}")

    if 'is_saturated' in cat.colnames:
        cat.remove_column('is_saturated')
    cat.add_column(replaced_sat_.astype(bool), name='is_saturated')

    if 'replaced_saturated' in cat.colnames:
        cat.remove_column('replaced_saturated')
    cat.add_column(replaced_sat_, name='replaced_saturated')
    if 'flux_fit' in cat.colnames:
        cat.rename_column('flux_fit', 'flux')
    else:
        print(f"Catalog did not have flux_fit.  colnames={cat.colnames}.  (this is expected for crowdsource)")
    # DEBUG print(f"DEBUG: cat['replaced_saturated'].sum(): {cat['replaced_saturated'].sum()}")


def main():
    print("Starting main")
    import time
    t0 = time.time()

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--modules", dest="modules",
                      default='merged,merged-reproject',
                      help="module list", metavar="modules")
    parser.add_option('--merge-singlefields', dest='merge_singlefields',
                      default=False, action='store_true',)
    parser.add_option("--target", dest="target",
                      default='brick',
                      help="target", metavar="target")
    parser.add_option("--skip-crowdsource", dest="skip_crowdsource",
                      default=False,
                      action="store_true",
                      help="skip_crowdsource", metavar="skip_crowdsource")
    parser.add_option("--skip-daophot", dest="skip_daophot",
                      default=False,
                      action="store_true",
                      help="skip_daophot", metavar="skip_daophot")
    parser.add_option("--strict-require-blur", dest="strict_require_blur",
                      default=False,
                      action="store_true",
                      help="Fail if blur files are not found?", metavar="strict_require_blur")
    parser.add_option("--make-refcat", dest='make_refcat', default=False,
                      action='store_true')
    parser.add_option('--max-expnum', dest='max_expnum', default=24, type='int')
    parser.add_option('--indiv-merge-methods', dest='indiv_merge_methods', default='dao,crowdsource,daoiterative')
    parser.add_option('--iteration-label', dest='iteration_label', default=None,
                      help='Filter per-frame inputs to those tagged with this iteration label '
                           '(e.g. "iter2" or "iter3").  Default merges the iter1 catalogs.')
    (options, args) = parser.parse_args()

    modules = options.modules.split(",")
    target = options.target
    indiv_merge_methods = options.indiv_merge_methods.split(",")
    print("Options:", options)

    if target in ('sickle', 'cloudef', 'sgrc', 'sgrb2', 'arches', 'quintuplet', 'sgra'):
        basepath = f'/orange/adamginsburg/jwst/{target}/'
    else:
        basepath = f'/blue/adamginsburg/adamginsburg/jwst/{target}/'

    offsets_tables = {'1182': Table.read(f'/blue/adamginsburg/adamginsburg/jwst/brick/offsets/Offsets_JWST_Brick1182_F444ref.csv'),
                      '2221': None,
                      '3958': None,
                      '2092': None,
                      '4147': None,
                      '5365': None,
                      '2045': None,
                      '1939': None,
    }

    # need to have incrementing _before_ test
    index = -1

    for module in modules:
        for desat in (False, True):
            for bgsub in (False, True):
                for epsf in (False, True):
                    for fitpsf in (False, True):
                        for blur in (False, True):

                            if options.merge_singlefields:
                                singlefield_done = False
                                for progid in obs_filters[target]:
                                    for filtername in (obs_filters[target][progid]):
                                        if singlefield_done:
                                            # skip ahead to merge-all-indiv step
                                            continue
                                        index += 1
                                        print(index, filtername, progid)
                                        # enable array jobs based only on filters
                                        if os.getenv('SLURM_ARRAY_TASK_ID') is not None and int(os.getenv('SLURM_ARRAY_TASK_ID')) != index:
                                            print(f'Task={os.getenv("SLURM_ARRAY_TASK_ID")} does not match index {index}')
                                            continue

                                        for method in indiv_merge_methods:
                                            print(method)
                                            # could loop & also do _iterative...
                                            suffix = {'crowdsource': '_nsky0',
                                                      'dao': '_basic',
                                                      'daoiterative': '_iterative',
                                                      'iterative': '_iterative',
                                                      }[method]
                                            try:
                                                merge_individual_frames(module=module,
                                                                        desat=desat,
                                                                        filtername=filtername,
                                                                        progid=progid,
                                                                        bgsub=bgsub,
                                                                        epsf=epsf,
                                                                        fitpsf=fitpsf,
                                                                        blur=blur,
                                                                        suffix=suffix,
                                                                        target=target,
                                                                        exposure_numbers=np.arange(1, options.max_expnum + 1),
                                                                        offsets_table=offsets_tables[progid],
                                                                        method=method,
                                                                        iteration_label=options.iteration_label,
                                                                        basepath=basepath)
                                            except ValueError as ex:
                                                if blur and not options.strict_require_blur:
                                                    print("Skipping missing blur files")
                                                else:
                                                    raise ex
                                            print(f"Finished merge_individual_frames {suffix} {progid} {filtername} {method}")
                                            if os.getenv('SLURM_ARRAY_TASK_ID') is not None:
                                                singlefield_done = True
                                            #except Exception as ex:
                                            #    print(f"Exception: {ex}, {type(ex)}, {str(ex)}")
                                            #    exc_type, exc_obj, exc_tb = sys.exc_info()
                                            #    print(f"Exception occurred on line {exc_tb.tb_lineno}")

                            else:
                                index += 1

                            # enable array jobs
                            if os.getenv('SLURM_ARRAY_TASK_ID') is not None and int(os.getenv('SLURM_ARRAY_TASK_ID')) != index:
                                print(f'Task={os.getenv("SLURM_ARRAY_TASK_ID")} does not match index {index}')
                                continue

                            t0 = time.time()
                            print(f"Index {index}")
                            if not options.skip_crowdsource:
                                print(f'crowdsource {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}. ', flush=True)
                                try:
                                    merge_crowdsource(module=module, desat=desat, bgsub=bgsub, epsf=epsf,
                                                      fitpsf=fitpsf, target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields)
                                except Exception as ex:
                                    print(f"Living with this error: {ex}, {type(ex)}, {str(ex)}")
                                try:
                                    for suffix in ("_nsky0", ):#"_nsky15"): "_nsky1",
                                        print(f'crowdsource {suffix} {module}')
                                        merge_crowdsource(module=module, suffix=suffix, desat=desat, bgsub=bgsub, epsf=epsf,
                                                          fitpsf=fitpsf, target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields)
                                except Exception as ex:
                                    print(f"Exception: {ex}, {type(ex)}, {str(ex)}")
                                    exc_type, exc_obj, exc_tb = sys.exc_info()
                                    print(f"Exception occurred on line {exc_tb.tb_lineno}")
                                    raise ex

                                try:
                                    print(f'crowdsource unweighted {module}', flush=True)
                                    merge_crowdsource(module=module, suffix='_unweighted', desat=desat, bgsub=bgsub, epsf=epsf,
                                                      fitpsf=fitpsf, target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields)
                                except NotImplementedError:
                                    continue
                                except Exception as ex:
                                    print(f"Exception for unweighted crowdsource: {ex}, {type(ex)}, {str(ex)}")
                                    #raise ex

                                print(f'crowdsource phase done.  time elapsed={time.time()-t0}')

                            if not options.skip_daophot:
                                t0 = time.time()
                                print("DAOPHOT")
                                print(f'daophot basic {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}', flush=True)
                                try:
                                    merge_daophot(daophot_type='basic', module=module, desat=desat,
                                                  bgsub=bgsub, epsf=epsf,
                                                  target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields)
                                except Exception as ex:
                                    print(f'daophot basic {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}', flush=True)
                                    if blur and not options.strict_require_blur:
                                        print("Skipping missing blur files")
                                    else:
                                        print(f"Exception when running merge_daophot: {ex}, {type(ex)}, {str(ex)}", flush=True)
                                        exc_tb = sys.exc_info()[2]
                                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                        print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}", flush=True)
                                        print(f"Exception {ex} was in {fname} line {exc_tb.tb_next.tb_lineno}", flush=True)
                                        raise ex
                                try:
                                    print(f'daophot iterative {module} desat={desat} bgsub={bgsub} epsf={epsf} blur={blur} fitpsf={fitpsf} target={target}', flush=True)
                                    merge_daophot(daophot_type='iterative', module=module, desat=desat,
                                                  bgsub=bgsub, epsf=epsf,
                                                  target=target, basepath=basepath, blur=blur, indivexp=options.merge_singlefields)
                                except Exception as ex:
                                    if blur and not options.strict_require_blur:
                                        print("Skipping missing blur files")
                                    else:
                                        print(f"Exception running merge daophot iterative: {ex}, {type(ex)}, {str(ex)}", flush=True)
                                        exc_tb = sys.exc_info()[2]
                                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                        print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}", flush=True)
                                print(f'dao phase done.  time elapsed={time.time()-t0}')
                                print()

                            if os.getenv('SLURM_ARRAY_TASK_ID') is None:
                                if options.make_refcat:
                                    raise Exception("make_refcat is not supported with SLURM_ARRAY_TASK_ID")
                                return

    if options.make_refcat:
        import make_reftable
        make_reftable.main()

    print("Done")


if __name__ == "__main__":
    main()
