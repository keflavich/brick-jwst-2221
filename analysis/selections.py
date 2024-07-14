import numpy as np
import sys, imp
import regions
import warnings
import glob
from astropy.io import fits
from astropy.table import Table
from astropy import stats
from astropy.wcs import WCS
from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u
from astroquery.svo_fps import SvoFps

try:
    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.psf import EPSFBuilder
    from photutils.detection import find_peaks
except ImportError:
    from photutils import CircularAperture, EPSFBuilder, find_peaks, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter

import dust_extinction
from dust_extinction.parameter_averages import CCM89
from dust_extinction.averages import RRP89_MWGC, CT06_MWGC, F11_MWGC

import PIL
import pyavm

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['figure.figsize'] = (10,8)
pl.rcParams['figure.dpi'] = 100
pl.rcParams['font.size'] = 16
from astropy.table import Table
from astropy import units as u

from analysis_setup import (basepath, reg, regzoom, distance_modulus,
                            filternames, plot_tools,
                            # basetable,
                            #basetable_merged_reproject,
                            #basetable_merged,
                            #basetable_merged1182,
                            #basetable_nrca, basetable_nrcb,
                            #basetable_merged_reproject_dao_iter_bg_epsf ,
                            #basetable_merged_reproject_dao_iter_epsf,
                            #basetable_merged_reproject_dao_iter,
                            )
from plot_tools import regzoomplot, starzoom


def find_stars_in_same_pixel(xx, yy, max_offset=1):
    from scipy.spatial import KDTree

    coords = np.array([xx, yy]).T
    bad = np.any(np.isnan(coords), axis=1)
    coords = np.nan_to_num(coords)

    tree = KDTree(coords)
    dist, ind = tree.query(coords, 2)

    # re-nanify these distances; we want to ignore them
    dist[bad, :] = np.nan
    close_neighbor = ind[:, 1][dist[:, 1] < max_offset]

    return close_neighbor


def main(basetable, ww):

    # empirical test: these sources are almost certainly saturated in f410m =(
    # 3.1 is the difference between the wrong mag and right mag
    saturated_f410m = ((basetable['mag_ab_f410m'] < (13.9-3.1)) &
                       (basetable['mag_ab_f410m'] - basetable['mag_ab_f405n'] >
                        0))
    basetable['mag_ab_f410m'].mask[saturated_f410m] = True
    basetable['flux_f410m'].mask[saturated_f410m] = True

    filternames = [basetable.meta[key] for key in basetable.meta if 'FILT' in key]
    print(f"Selecting based on filters {filternames}")

    # FITS tables can't mask boolean columns
    # so, we have to mask the saturated mask using the mask on the flux for the filter
    any_saturated_ = [basetable[f'near_saturated_{x}_{x}'] &
                      ~basetable[f'flux_{x}'].mask for x in filternames]
    any_saturated_narrow_ = [basetable[f'near_saturated_{x}_{x}'] &
                            ~basetable[f'flux_{x}'].mask for x in filternames
                            if 'w' not in x.lower()
                           ]

    any_saturated = any_saturated_[0]
    for col in any_saturated_[1:]:
        print(f"{col.sum()} saturated in {col.name}")
        any_saturated = any_saturated | col
    print(f"{any_saturated.sum()} near saturated out of {len(basetable)}.  That leaves {(~any_saturated).sum()} not near unsaturated")

    any_saturated_narrow = any_saturated_narrow_[0]
    for col in any_saturated_narrow_[1:]:
        print(f"{col.sum()} saturated in {col.name}")
        any_saturated_narrow = any_saturated_narrow | col
    print(f"{any_saturated_narrow.sum()} near saturated out of {len(basetable)}.  That leaves {(~any_saturated_narrow).sum()} not near unsaturated")

    any_replaced_saturated_ = [basetable[f'replaced_saturated_{x}'] &
                               ~basetable[f'flux_{x}'].mask for x in filternames]
    any_replaced_saturated = any_replaced_saturated_[0]
    for col in any_replaced_saturated_[1:]:
        print(f"{col.sum()} saturated in {col.name}")
        any_replaced_saturated = any_replaced_saturated | col
    print(f"{any_replaced_saturated.sum()} saturated out of {len(basetable)}.  That leaves {(~any_replaced_saturated).sum()} unsaturated")

    magerr_gtpt1_any = np.logical_or.reduce([basetable[f'emag_ab_{filtername}'] > 0.1 for filtername in filternames])
    magerr_gtpt05_any = np.logical_or.reduce([basetable[f'emag_ab_{filtername}'] > 0.05 for filtername in filternames])
    magerr_gtpt1_all = np.logical_and.reduce([basetable[f'emag_ab_{filtername}'] > 0.1 for filtername in filternames])
    magerr_gtpt05_all = np.logical_and.reduce([basetable[f'emag_ab_{filtername}'] > 0.05 for filtername in filternames])

    magerr_gtpt05 = magerr_gtpt05_all
    magerr_gtpt1 = magerr_gtpt1_all

    magerr_gtpt1_notwide_any = np.logical_or.reduce([basetable[f'emag_ab_{filtername}'] > 0.1 for filtername in filternames if 'w' not in filtername.lower()])
    magerr_gtpt05_notwide_any = np.logical_or.reduce([basetable[f'emag_ab_{filtername}'] > 0.05 for filtername in filternames if 'w' not in filtername.lower()])
    magerr_gtpt1_notwide_all = np.logical_and.reduce([basetable[f'emag_ab_{filtername}'] > 0.1 for filtername in filternames if 'w' not in filtername.lower()])
    magerr_gtpt05_notwide_all = np.logical_and.reduce([basetable[f'emag_ab_{filtername}'] > 0.05 for filtername in filternames if 'w' not in filtername.lower()])

    # crowdsource parameters
    # 2024-07-13: made parameters more restrictive (qf 0.6->0.9, minfracflux 0.8 -> 0.75)
    minqf = 0.90
    maxspread = 0.25
    minfracflux = 0.75

    # daophot parameters
    max_qfit = 0.4
    max_cfit = 0.1

    for filt in filternames:
        filt = filt.lower()
        mask = basetable[f'mag_ab_{filt}'].mask

        if f'qf_{filt}' in basetable.colnames:
            # this qf threshold can be pretty stringent; 0.98 drops the number of sources a lot
            # Eddie Schlafly recommended "For unsaturated sources, I'd be deeply
            # skeptical of anything with qf < 0.6 or so; the suggestion is that we're
            # on the edge of a chip or a bad region and don't even have the peak on a
            # good pixel. I'd put tighter bounds if I wanted very good photometry, more
            # like 90-95%."
            # qf > 0.6 looks pretty decent so I'm rollin with it
            qfok = ((basetable[f'qf_{filt}'] > minqf).data & (~(basetable[f'qf_{filt}']).mask))
            qfmask = basetable[f'qf_{filt}'].mask
            # it's not very clear what the spread model does; Schafly points to
            # https://sextractor.readthedocs.io/en/latest/Model.html#model-based-star-galaxy-separation-spread-model
            # it may be useful for IDing extended sources
            # TEMPORARY July 13, 2024: 'spread_model' wasn't propagated
            if f'spread_model_{filt}' in basetable.colnames:
                spok = ((np.abs(basetable[f'spread_model_{filt}']) < maxspread) &
                        (~basetable[f'spread_model_{filt}'].mask))
            else:
                spok = np.ones(len(basetable), dtype=bool)
            # fracflux is intended to be a measure of how blended the source is. It's
            # the PSF-weighted flux of the stamp after subtracting neighbors, divided
            # by the PSF-weighted flux of the full image including neighbors. So if you
            # have no neighbors around, it's 1. If typically half the flux in one of
            # your pixels is from your neighbors, it's 0.5, where 'typically' is in a
            # PSF-weighted sense.
            ffok = ((basetable[f'fracflux_{filt}'] > minfracflux) & (~basetable[f'fracflux_{filt}'].mask))
        elif f'qfit_{filt}' in basetable.colnames:
            qfok = (basetable[f'qfit_{filt}'] < max_qfit)
            qfmask = basetable[f'qfit_{filt}'].mask
            spok = ((basetable[f'cfit_{filt}'] < max_cfit) &
                    (~basetable[f'cfit_{filt}'].mask))
            ffok = True

        basetable[f'good_{filt}'] = allok = (qfok & spok & ffok)
        print(f"Filter {filt} has qf={qfok.sum()}, spread={spok.sum()}, fracflux={ffok.sum() if hasattr(ffok, 'sum') else 1} ok,"
            f" totaling {allok.sum()}.  There are {len(basetable)} total, of which "
            f"{mask.sum()} are masked and {(~mask).sum()} are unmasked. qfmasksum={qfmask.sum()}, inverse={(~qfmask).sum()}.")

    # DO NOT exclude missing f115w; it removes too much
    all_good = np.all([basetable[f'good_{filt}'] for filt in filternames if filt.lower() != 'f115w'], axis=0)
    any_good = np.any([basetable[f'good_{filt}'] for filt in filternames  if filt.lower() != 'f115w'], axis=0)
    long_good = np.all([basetable[f'good_{filt}'] for filt in filternames if 'f4' in filt], axis=0)
    short_good = np.all([basetable[f'good_{filt}'] for filt in filternames if 'f4' not in filt], axis=0)
    print(f"Of {len(all_good)} rows, {all_good.sum()} are good in all filters.")
    print(f"Of {len(all_good)} rows, {long_good.sum()} are good in long filters.")
    print(f"Of {len(all_good)} rows, {short_good.sum()} are good in short filters.")
    print(f"Of {len(all_good)} rows, {any_good.sum()} are good in at least one filter.")

    # crowdsource
    if 'qf_f410m' in basetable.colnames:
        goodqflong = ((basetable['qf_f410m'] > minqf) & (basetable['qf_f405n'] > minqf) & (basetable['qf_f466n'] > minqf))
        if 'spread_model_f410m' in basetable.colnames:
            goodspreadlong = ((basetable['spread_model_f410m'] < maxspread) | (basetable['spread_model_f405n'] < maxspread) | (basetable['spread_model_f466n'] < maxspread))
        else:
            goodspreadlong = np.ones(len(basetable), dtype=bool)
        goodfracfluxlong = ((basetable['fracflux_f410m'] > minfracflux) | (basetable['fracflux_f405n'] > minfracflux) & (basetable['fracflux_f466n'] > minfracflux))
    elif 'qfit_f410m' in basetable.colnames:
        goodqflong = ((basetable['qfit_f410m'] < max_qfit) &
                      (basetable['qfit_f405n'] < max_qfit) &
                      (basetable['qfit_f466n'] < max_qfit))
        # I'm using the same variable name to save rewriting below... this is not a great choice
        goodspreadlong = ((basetable['cfit_f410m'] < max_cfit) |
                          (basetable['cfit_f405n'] < max_cfit) |
                          (basetable['cfit_f466n'] < max_cfit))
        goodfracfluxlong = True #((basetable['fracflux_f410m'] > minfracflux) | (basetable['fracflux_f405n'] > minfracflux) & (basetable['fracflux_f466n'] > minfracflux))

    # masked arrays don't play nice
    goodqflong = np.array(goodqflong & ~goodqflong.mask)
    try:
        goodspreadlong = np.array(goodspreadlong & ~goodspreadlong.mask)
    except AttributeError:
        pass
    goodfracfluxlong = np.array(goodfracfluxlong & (~goodfracfluxlong.mask if hasattr(goodfracfluxlong, 'mask') else True))
    allgood_long = (goodqflong & goodspreadlong & goodfracfluxlong)

    badqflong = ~goodqflong
    badspreadlong = ~goodspreadlong
    badfracfluxlong = ~goodfracfluxlong

    if 'qf_f212n' in basetable.colnames:
        goodqfshort = ((basetable['qf_f212n'] > minqf) & (basetable['qf_f182m'] > minqf) & (basetable['qf_f187n'] > minqf))
        if 'spread_model_f212n' in basetable.colnames:
            goodspreadshort = ((basetable['spread_model_f212n'] < maxspread) & (basetable['spread_model_f182m'] < maxspread) & (basetable['spread_model_f187n'] < maxspread))
        else:
            goodspreadshort = np.ones(len(basetable), dtype=bool)
        goodfracfluxshort = ((basetable['fracflux_f212n'] > minfracflux) & (basetable['fracflux_f182m'] > minfracflux) & (basetable['fracflux_f187n'] > minfracflux))
    else:
        goodqfshort = ((basetable['qfit_f212n'] < max_qfit) &
                       (basetable['qfit_f187n'] < max_qfit) &
                       (basetable['qfit_f182m'] < max_qfit))
        # I'm using the same variable name to save rewriting below... this is not a great choice
        goodspreadshort = ((basetable['cfit_f212n'] < max_cfit) |
                           (basetable['cfit_f187n'] < max_cfit) |
                           (basetable['cfit_f182m'] < max_cfit))
        goodfracfluxshort = True #((basetable['fracflux_f410m'] > minfracflux) | (basetable['fracflux_f405n'] > minfracflux) & (basetable['fracflux_f466n'] > minfracflux))

    goodqfshort = np.array(goodqfshort & ~goodqfshort.mask)
    try:
        goodspreadshort = np.array(goodspreadshort & ~goodspreadshort.mask)
    except AttributeError:
        pass
    goodfracfluxshort = np.array(goodfracfluxshort & (~goodfracfluxshort.mask if hasattr(goodfracfluxshort, 'mask') else True))
    allgood_short = (goodqfshort & goodspreadshort & goodfracfluxshort)

    badqfshort = ~goodqfshort
    badspreadshort = ~goodspreadshort
    badfracfluxshort = ~goodfracfluxshort

    print(f"QFs: {goodqfshort.sum()} good short")
    print(f"QFs: {goodqflong.sum()} good long")

    # threshold = 0.1 arcsec
    oksep_notwide = np.logical_and.reduce([basetable[f'sep_{filtername}'] < 0.1*u.arcsec for filtername in filternames if 'w' not in filtername])
    print(f"Found {oksep_notwide.sum()} of {len(oksep_notwide)} sources with separations < 0.1 arcsec (excluding wide filters)")
    oksep_noJ = np.logical_and.reduce([basetable[f'sep_{filtername}'] < 0.1*u.arcsec for filtername in filternames if 'f115w' not in filtername])
    print(f"Found {oksep_noJ.sum()} of {len(oksep_noJ)} sources with separations < 0.1 arcsec (excluding f115w)")
    oksep = np.logical_and.reduce([basetable[f'sep_{filtername}'] < 0.1*u.arcsec for filtername in filternames])
    print(f"Found {oksep.sum()} of {len(oksep)} sources with separations < 0.1 arcsec")
    oklong = oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) & (~badqflong) & (~badspreadlong) & (~badfracfluxlong)

    # This text is just to check what the offset was to account for an error I made in 2023.  We no longer need to correct that error because it's done right in cataloging.
    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    abconv = (1*u.Jy).to(u.ABmag)
    filtconv410 = -2.5*np.log10(1/jfilts.loc['JWST/NIRCam.F410M']['ZeroPoint']) - abconv.value
    filtconv466 = -2.5*np.log10(1/jfilts.loc['JWST/NIRCam.F466N']['ZeroPoint']) - abconv.value
    zeropoint_offset_410_466 = filtconv410-filtconv466
    print(f'Offset between raw ABmag for F410M-F466N = {filtconv410} - {filtconv466} = {zeropoint_offset_410_466}')
    # May 11, 2024: the new versions of the catalogs don't have this magnitude offset error
    # so this should be gone now, right?
    zeropoint_offset_410_466 = 0

    slightly_blue_410_466 =  (oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) &
        ((basetable['mag_ab_410m405'] - basetable['mag_ab_f466n']) +
        (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2 + basetable['emag_ab_f405n']**2)**0.5 < zeropoint_offset_410_466)
        & (~badqflong) & (~badspreadlong) & (~badfracfluxlong))

    veryblue_410m405_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) &
        ((basetable['mag_ab_410m405'] - basetable['mag_ab_f466n']) +
        (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2 + basetable['emag_ab_f405n']**2)**0.5 < (-1.75+zeropoint_offset_410_466))
        )
    veryblue_410_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_f410m'].mask)) &
        ((basetable['mag_ab_f410m'] - basetable['mag_ab_f466n']) +
        (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2)**0.5 < (-1.75+zeropoint_offset_410_466)
        ))

    blue_410m405_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) &
        (((basetable['mag_ab_410m405'] - basetable['mag_ab_f466n']) +
          (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2 + basetable['emag_ab_f405n']**2)**0.5) < (-0.75+zeropoint_offset_410_466))
        )
    blue_410_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_f410m'].mask)) &
        (((basetable['mag_ab_f410m'] - basetable['mag_ab_f466n']) +
          (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2)**0.5) < (-0.75+zeropoint_offset_410_466)))
    # this assertion is presumably because blue_410_466 was being computed from 410m405 before
    # assert (blue_410_466 & basetable['mag_ab_410m405'].mask).sum() == 0
    # now this is the correct assertion
    assert (blue_410m405_466 & basetable['mag_ab_410m405'].mask).sum() == 0
    blue_410_405 = (oksep & ~any_saturated & (~(basetable['mag_ab_410m405'].mask)) &
                    ((basetable['mag_ab_410m405'] - basetable['mag_ab_f405n']) +
                     (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -2))
    blue_405_410 = (oksep & ~any_saturated & (~(basetable['mag_ab_410m405'].mask)) &
                    ((basetable['mag_ab_405m410'] - basetable['mag_ab_410m405']) +
                     (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -1)
                    & ~magerr_gtpt1 & (~badqflong) & (~badspreadlong) & (~badfracfluxlong))
    blue_405_410b = (oksep & ~any_saturated & (basetable['flux_f405n'] > basetable['flux_f410m']) &
                     (~(basetable['mag_ab_f405n'].mask)) &
                     ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) +
                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -0.5)
                     & ~magerr_gtpt1 & (~badqflong) & (~badspreadlong) & (~badfracfluxlong))
    blue_187_182 = (oksep & ~any_saturated & (basetable['flux_f187n'] > basetable['flux_f182m']) &
                     (~(basetable['mag_ab_f187n'].mask)) &
                     ((basetable['mag_ab_f187n'] - basetable['mag_ab_f182m']) +
                      (basetable['emag_ab_f182m']**2 + basetable['emag_ab_f187n']**2)**0.5 < -1)
                    & ~magerr_gtpt1 & (~badqflong) & (~badspreadlong) & (~badfracfluxlong))
    print(f"Possible BrA excess (405-410 < -1): {blue_405_410.sum()}, (405-410 < -0.5): {blue_405_410b.sum()}.")

    blue_BrA_and_PaA = (oksep & ~any_saturated &
                        (basetable['flux_f405n'] > basetable['flux_f410m']) &
                        (basetable['flux_f187n'] > basetable['flux_f182m']) &
                     (~(basetable['mag_ab_f405n'].mask)) &
                     (~(basetable['mag_ab_f410m'].mask)) &
                     (~(basetable['mag_ab_f187n'].mask)) &
                     (~(basetable['mag_ab_f182m'].mask)) &
                     ((basetable['mag_ab_f187n'] - basetable['mag_ab_f182m']) +
                      (basetable['emag_ab_f182m']**2 + basetable['emag_ab_f187n']**2)**0.5 < -0.1) &
                     ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) +
                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -0.1)
                        & ~magerr_gtpt1 &
                        basetable['good_f405n'] &
                        basetable['good_f410m'] &
                        basetable['good_f187n'] &
                        basetable['good_f182m']
                       )
    prettyblue_BrA_and_PaA = (blue_BrA_and_PaA &
                     ((basetable['mag_ab_f187n'] - basetable['mag_ab_f182m']) +
                      (basetable['emag_ab_f182m']**2 + basetable['emag_ab_f187n']**2)**0.5 < -0.2) &
                     ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) +
                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -0.2)
                           )
    veryblue_BrA_and_PaA = (blue_BrA_and_PaA &
                     ((basetable['mag_ab_f187n'] - basetable['mag_ab_f182m']) +
                      (basetable['emag_ab_f182m']**2 + basetable['emag_ab_f187n']**2)**0.5 < -0.5) &
                     ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) +
                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -0.5)
                           )
                        #& (~badqflong) & (~badspreadlong) & (~badfracfluxlong))
    detected = ((~basetable['mag_ab_f405n'].mask) &
                (~basetable['mag_ab_f410m'].mask) &
                (~basetable['mag_ab_f187n'].mask) &
                (~basetable['mag_ab_f182m'].mask))
    detected_allnonwidebands = ((~basetable['mag_ab_f405n'].mask) &
                         (~basetable['mag_ab_f410m'].mask) &
                         (~basetable['mag_ab_f466n'].mask) &
                         (~basetable['mag_ab_f212n'].mask) &
                         (~basetable['mag_ab_f187n'].mask) &
                         (~basetable['mag_ab_f182m'].mask))
    detected_allbands = np.vstack([(~basetable[f'mag_ab_{filt}'].mask) for filt in filternames]).min(axis=0)
    print(f"Detected in 405, 410, 187, and 182: {detected.sum()}.  In all 2221: {detected_allnonwidebands.sum()}.  All incl 1182: {detected_allbands.sum()}")
    print(f"Very likely BrA+PaA excess (405-410 < -0.1 and 187-182 < -0.1): {blue_BrA_and_PaA.sum()}, <-0.5: {veryblue_BrA_and_PaA.sum()}.")
    print(f"Pretty blue [410-466] sources: {blue_410_466.sum()}")
    print(f"Pretty blue [410m405-466] sources: {blue_410m405_466.sum()}")
    print(f"Very blue [410-466] sources: {veryblue_410_466.sum()}")
    print(f"Very blue [410m405-466] sources: {veryblue_410m405_466.sum()}")
    print(f"Somewhat blue [410m405-466] sources: {slightly_blue_410_466.sum()}")
    print(oklong.sum(), blue_410_466.sum(), slightly_blue_410_466.sum(), blue_405_410.sum(), blue_405_410b.sum(), blue_BrA_and_PaA.sum(), detected.sum(), blue_BrA_and_PaA.sum() / detected.sum())

    neg_405m410 = basetable['flux_jy_405m410'] < 0
    print(f"Negative 405-410 colors: {neg_405m410.sum()}, Nonnegative: {(~neg_405m410).sum()}")

    any_saturated |= saturated_f410m
    all_good &= ~saturated_f410m
    print(f"There are {all_good.sum()} before flagging out any_saturated (there are {any_saturated.sum()} any_saturated)")
    all_good &= ~any_saturated
    print(f"There are {all_good.sum()} after flagging out any_saturated")

    print(f"There are {oksep.sum()} out of {len(oksep)} oksep, and {(oksep & all_good).sum()} all_good & oksep")
    all_good_phot = all_good.copy()
    all_good = all_good_phot & oksep

    exclude = (any_saturated | ~oksep_noJ | magerr_gtpt1_all |
               basetable['mag_ab_f405n'].mask | basetable['mag_ab_f410m'].mask |
               badqflong | badfracfluxlong | badspreadlong)
    print(f"Excluding {exclude.sum()} of {exclude.size} ({exclude.sum()/exclude.size*100}%)")

    # "bad" was totally broken; (bad & all_good) is very nonzero
    # bad = (any_saturated | ~oksep | magerr_gtpt1 | basetable['mag_ab_f212n'].mask |
    #        basetable['mag_ab_f410m'].mask | badqflong | badfracfluxlong |
    #        badspreadlong | badqfshort | badfracfluxshort | badspreadshort)
    bad = ~all_good
    print("'Bad' sources are those where _any_ filter is masked out")
    print(f"Not-bad:{(~bad).sum()}, bad: {bad.sum()},"# bad.mask: {bad.mask.sum()},"
          f" len(bad):{len(bad)}, len(table):{len(basetable)}.")

    for filt in filternames:
        print(f"{filt} median mag={np.nanmedian(np.array(basetable['mag_ab_'+filt][basetable['good_'+filt]]))}")

    # Basic selections for CMD, CCD plotting
    sel = reg.contains(basetable['skycoord_f410m'], ww)
    sel &= basetable['sep_f466n'].quantity < 0.1*u.arcsec
    sel &= basetable['sep_f405n'].quantity < 0.1*u.arcsec

    def ccds_withiso(basetable=basetable, sel=sel, exclude=exclude, **kwargs):
        return plot_tools.ccds_withiso(basetable=basetable, sel=sel, exclude=exclude, **kwargs)

    def cmds_withiso(basetable=basetable, sel=sel, exclude=exclude, distance_modulus=distance_modulus, **kwargs):
        return plot_tools.cmds_withiso(basetable=basetable, sel=sel, exclude=exclude, distance_modulus=distance_modulus, **kwargs)


    sel = reg.contains(basetable['skycoord_f410m'], ww)
    sel &= basetable['sep_f466n'].quantity < 0.1*u.arcsec
    sel &= basetable['sep_f405n'].quantity < 0.1*u.arcsec

    def ccds(basetable=basetable, sel=sel, **kwargs):
        return plot_tools.ccds(basetable=basetable, sel=sel, **kwargs)

    def cmds(basetable=basetable, sel=sel, **kwargs):
        return plot_tools.cmds(basetable=basetable, sel=sel, **kwargs)

    crds = basetable['skycoord_f410m']

    # not sure these are legitimately bad?
    # Feb 11, 2023: these are the same objects as 'weird blue'
    # This is needed by some plots, but isn't obviously useful
    badblue = blue_410m405_466 & ( ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) > 2)
                                 # | ((basetable['mag_ab_f410m'] - basetable['mag_ab_f466n']) > -1)
                                 )

    c187_212 = basetable['mag_ab_f187n'] - basetable['mag_ab_f212n']
    c212_405 = basetable['mag_ab_f212n'] - basetable['mag_ab_f405n']

    # Coarse color cut eyeballed in CatalogExploration_Sep2023
    recomb_excess_over_212 = c212_405 > c187_212 * (4 / 3.) + 0.35

    # calculate A_V from colors
    # super naive version
    av212410 = (basetable['mag_ab_f212n'] - basetable['mag_ab_f410m']) / (CT06_MWGC()(2.12*u.um) - CT06_MWGC()(4.10*u.um))
    # but empirically, this value appears to start at 1.2 (f212n - f410m has a locus at -1.2)
    av212410 = (1.2 + basetable['mag_ab_f212n'] - basetable['mag_ab_f410m']) / (CT06_MWGC()(2.12*u.um) - CT06_MWGC()(4.10*u.um))
    # so why not just use the 182m?
    av182212 = (basetable['mag_ab_f182m'] - basetable['mag_ab_f212n']) / (CT06_MWGC()(1.82*u.um) - CT06_MWGC()(2.12*u.um))
    # or 182-410
    av182410 = (basetable['mag_ab_f182m'] - basetable['mag_ab_f410m']) / (CT06_MWGC()(1.82*u.um) - CT06_MWGC()(4.10*u.um))
    # or 187-405
    av187405 = (basetable['mag_ab_f187n'] - basetable['mag_ab_f405n']) / (CT06_MWGC()(1.87*u.um) - CT06_MWGC()(4.05*u.um))

    if (('mag_ab_f444w' in basetable.colnames and
         'mag_ab_f356w' in basetable.colnames and
         'mag_ab_f200w' in basetable.colnames and
         'mag_ab_f115w' in basetable.colnames)):
        av356444 = (basetable['mag_ab_f356w'] - basetable['mag_ab_f444w']) / (CT06_MWGC()(3.56*u.um) - CT06_MWGC()(4.44*u.um))
        av200356 = (basetable['mag_ab_f200w'] - basetable['mag_ab_f356w']) / (CT06_MWGC()(2.00*u.um) - CT06_MWGC()(3.56*u.um))
        # CT06 doesn't work short of 2um
        av115200 = (basetable['mag_ab_f115w'] - basetable['mag_ab_f200w']) / (RRP89_MWGC()(1.15*u.um) - RRP89_MWGC()(2.00*u.um))

    if 'x_fit_f410m' in basetable.colnames:
        doubled = {filtername: find_stars_in_same_pixel(basetable[f'x_fit_{filtername}'], basetable[f'y_fit_{filtername}'])
                for filtername in filternames}
        two_stars_in_same_pixel = np.zeros(len(basetable), dtype='bool')
        for _, inds in doubled:
            two_stars_in_same_pixel[inds] = True

        print(f"Found {two_stars_in_same_pixel.sum()} stars that were doubled up.", {key: len(val) for key, val in doubled.items()})

    return locals()


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--module", dest="module", default='merged',
                      help="module to select", metavar="module")
    (options, args) = parser.parse_args()

    print(f"Selecting module {options.module}")

    # save nrca and nrcb result tables
    # print()
    # print("NRCA")
    # from analysis_setup import fh_nrca as fh, ww410_nrca as ww410, ww410_nrca as ww
    # result = main(basetable_nrca, ww=ww)
    # globals().update({key+"_a": val for key, val in result.items()})

    #print()
    #print("NRCB")
    #from analysis_setup import fh_nrcb as fh, ww410_nrcb as ww410, ww410_nrcb as ww
    #result = main(basetable_nrcb, ww=ww)
    #globals().update({key+"_b": val for key, val in result.items()})

    #print()
    #print("merged-reproject")
    from analysis_setup import fh_merged_reproject as fh, ww410_merged_reproject as ww410, ww410_merged_reproject as ww
    #result = main(basetable_merged_reproject, ww=ww)
    #globals().update({key+"_mr": val for key, val in result.items()})

    #print()
    #print("merged")
    from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
    #result = main(basetable_merged, ww=ww)
    #globals().update({key+"_m": val for key, val in result.items()})

    #if options.module == 'nrca':
    #    from analysis_setup import fh_nrca as fh, ww410_nrca as ww410, ww410_nrca as ww
    #    result = main(basetable_nrca, ww=ww)
    #    globals().update(result)
    #    basetable = basetable_nrca
    #    print("Loaded nrca")
    #elif options.module == 'nrcb':
    #    from analysis_setup import fh_nrcb as fh, ww410_nrcb as ww410, ww410_nrcb as ww
    #    result = main(basetable_nrcb, ww=ww)
    #    globals().update(result)
    #    basetable = basetable_nrcb
    #    print("Loaded nrcb")
    print()
    print(options.module)
    if options.module == 'merged':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged = basetable = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged_photometry_tables_merged.fits')
        result = main(basetable_merged, ww=ww)
        globals().update(result)
        basetable = basetable_merged
        print("Loaded merged")
    elif options.module == 'merged1182indivexp':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182 = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged_indivexp_photometry_tables_merged_qualcuts.fits')
        result = main(basetable_merged1182, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182
        print("Loaded merged1182 indivexp")
    elif options.module == 'merged1182':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182 = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged_photometry_tables_merged.fits')
        result = main(basetable_merged1182, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182
        print("Loaded merged1182")
    elif options.module == 'merged1182_blur':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182_blur = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged_photometry_tables_merged_blur.fits')
        result = main(basetable_merged1182_blur, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182_blur
        print("Loaded merged1182_blur")

    elif options.module == 'merged1182_daophot_iterative':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/iterative_merged_photometry_tables_merged.fits')
        result = main(basetable_merged1182_daophot, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182_daophot
        print("Loaded merged1182_daophot_iterative")

    elif options.module == 'merged1182_daophot_iterative_blur':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182_daophot_blur = Table.read(f'{basepath}/catalogs/iterative_merged_photometry_tables_merged_blur.fits')
        result = main(basetable_merged1182_daophot_blur, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182_daophot_blur
        print("Loaded merged1182_daophot_iterative_blur")

    elif options.module == 'merged1182_daophot_basic':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_photometry_tables_merged.fits')
        result = main(basetable_merged1182_daophot, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182_daophot
        print("Loaded merged1182_daophot_basic")

    elif options.module == 'merged1182_daophot_basic_blur':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182_daophot_blur = Table.read(f'{basepath}/catalogs/basic_merged_photometry_tables_merged_blur.fits')
        result = main(basetable_merged1182_daophot_blur, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182_daophot_blur
        print("Loaded merged1182_daophot_basic_blur")

    elif options.module == 'merged1182_daophot_basic_bgsub':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182_daophot_bgsub = Table.read(f'{basepath}/catalogs/basic_merged_photometry_tables_merged_bgsub.fits')
        result = main(basetable_merged1182_daophot_bgsub, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182_daophot_bgsub
        print("Loaded merged1182_daophot_basic_bgsub")

    elif options.module == 'merged1182_daophot_basic_bgsub_blur':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        basetable_merged1182_daophot_bgsub_blur = Table.read(f'{basepath}/catalogs/basic_merged_photometry_tables_merged_bgsub_blur.fits')
        result = main(basetable_merged1182_daophot_bgsub_blur, ww=ww)
        globals().update(result)
        basetable = basetable_merged1182_daophot_bgsub_blur
        print("Loaded merged1182_daophot_basic_bgsub_blur")



    elif options.module == 'merged-reproject':
        from analysis_setup import fh_merged_reproject as fh, ww410_merged_reproject as ww410, ww410_merged_reproject as ww
        basetable_merged_reproject = Table.read(f'{basepath}/catalogs/crowdsource_nsky0_merged-reproject_photometry_tables_merged_20231003.fits')
        result = main(basetable_merged_reproject, ww=ww)
        globals().update(result)
        basetable = basetable_merged_reproject
        print("Loaded merged-reproject")
    elif options.module == 'merged-reproject-iterative-bg-epsf':
        print("Merged DAOPHOT iterative")
        from analysis_setup import (fh_merged_reproject as fh,
                                    ww410_merged_reproject as ww410,
                                    ww410_merged_reproject as ww)
        result = main_dao(basetable_merged_reproject_dao_iter_bg_epsf, ww=ww)
        globals().update(result)
        basetable = basetable_merged_reproject_dao_iter_bg_epsf
        print("Loaded merged-reproject-iterative-bg-epsf")
    else:
        print("Loaded nothing")

    assert 'blue_410m405_466' in globals()
