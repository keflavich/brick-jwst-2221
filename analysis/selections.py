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
    from photutils.apeture import CircularAnnulus, CircularAperture
    from photutils.psf import EPSFBuilder
    from photutils.detection import find_peaks
except ImportError:
    from photutils import CircularAperture, EPSFBuilder, find_peaks, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
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
                            filternames, basetable, plot_tools, basetable,
                            basetable_merged_reproject,
                            basetable_merged, basetable_nrca, basetable_nrcb,
                            basetable_merged_reproject_dao_iter_bg_epsf ,
                            basetable_merged_reproject_dao_iter_epsf,
                            basetable_merged_reproject_dao_iter,
                            )
from plot_tools import regzoomplot, starzoom


def main(basetable, ww):

    # empirical test: these sources are almost certainly saturated in f410m =(
    # 3.1 is the difference between the wrong mag and right mag
    saturated_f410m = ((basetable['mag_ab_f410m'] < (13.9-3.1)) &
                       (basetable['mag_ab_f410m'] - basetable['mag_ab_f405n'] >
                        0))
    basetable['mag_ab_f410m'].mask[saturated_f410m] = True
    basetable['flux_f410m'].mask[saturated_f410m] = True


    # FITS tables can't mask boolean columns
    # so, we have to mask the saturated mask using the mask on the flux for the filter
    any_saturated_ = [basetable[f'near_saturated_{x}_{x}'] & ~basetable[f'flux_{x}'].mask for x in filternames]

    any_saturated = any_saturated_[0]
    for col in any_saturated_[1:]:
        print(f"{col.sum()} saturated in {col.name}")
        any_saturated = any_saturated | col
    print(f"{any_saturated.sum()} near saturated out of {len(basetable)}.  That leaves {(~any_saturated).sum()} not near unsaturated")

    any_replaced_saturated_ = [basetable[f'replaced_saturated_{x}'] &
                               ~basetable[f'flux_{x}'].mask for x in filternames]
    any_replaced_saturated = any_replaced_saturated_[0]
    for col in any_replaced_saturated_[1:]:
        print(f"{col.sum()} saturated in {col.name}")
        any_replaced_saturated = any_replaced_saturated | col
    print(f"{any_replaced_saturated.sum()} saturated out of {len(basetable)}.  That leaves {(~any_replaced_saturated).sum()} unsaturated")

    magerr_gtpt1 = np.logical_or.reduce([basetable[f'emag_ab_{filtername}'] > 0.2 for filtername in filternames])
    magerr_gtpt1.sum()

    minqf = 0.60
    maxspread = 0.25
    minfracflux = 0.8

    for filt in filternames:
        filt = filt.lower()
        mask = basetable[f'mag_ab_{filt}'].mask
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
        spok = ((np.abs(basetable[f'spread_model_{filt}']) < maxspread) &
                (~basetable[f'spread_model_{filt}'].mask))
        # fracflux is intended to be a measure of how blended the source is. It's
        # the PSF-weighted flux of the stamp after subtracting neighbors, divided
        # by the PSF-weighted flux of the full image including neighbors. So if you
        # have no neighbors around, it's 1. If typically half the flux in one of
        # your pixels is from your neighbors, it's 0.5, where 'typically' is in a
        # PSF-weighted sense.
        ffok = ((basetable[f'fracflux_{filt}'] > minfracflux) & (~basetable[f'fracflux_{filt}'].mask))
        basetable[f'good_{filt}'] = allok = (qfok & spok & ffok)
        print(f"Filter {filt} has qf={qfok.sum()}, spread={spok.sum()}, fracflux={ffok.sum()} ok,"
              f" totaling {allok.sum()}.  There are {len(basetable)} total, of which "
              f"{mask.sum()} are masked and {(~mask).sum()} are unmasked. qfmasksum={qfmask.sum()}, inverse={(~qfmask).sum()}.")

    all_good = np.all([basetable[f'good_{filt}'] for filt in filternames], axis=0)
    any_good = np.any([basetable[f'good_{filt}'] for filt in filternames], axis=0)
    long_good = np.all([basetable[f'good_{filt}'] for filt in filternames if 'f4' in filt], axis=0)
    short_good = np.all([basetable[f'good_{filt}'] for filt in filternames if 'f4' not in filt], axis=0)
    print(f"Of {len(all_good)} rows, {all_good.sum()} are good in all filters.")
    print(f"Of {len(all_good)} rows, {long_good.sum()} are good in long filters.")
    print(f"Of {len(all_good)} rows, {short_good.sum()} are good in short filters.")
    print(f"Of {len(all_good)} rows, {any_good.sum()} are good in at least one filter.")

    goodqflong = ((basetable['qf_f410m'] > minqf) & (basetable['qf_f405n'] > minqf) & (basetable['qf_f466n'] > minqf))
    goodspreadlong = ((basetable['spread_model_f410m'] < maxspread) | (basetable['spread_model_f405n'] < maxspread) | (basetable['spread_model_f466n'] < maxspread))
    goodfracfluxlong = ((basetable['fracflux_f410m'] > minfracflux) | (basetable['fracflux_f405n'] > minfracflux) & (basetable['fracflux_f466n'] > minfracflux))

    # masked arrays don't play nice
    goodqflong = np.array(goodqflong & ~goodqflong.mask)
    goodspreadlong = np.array(goodspreadlong & ~goodspreadlong.mask)
    goodfracfluxlong = np.array(goodfracfluxlong & ~goodfracfluxlong.mask)
    allgood_long = (goodqflong & goodspreadlong & goodfracfluxlong)

    badqflong = ~goodqflong
    badspreadlong = ~goodspreadlong
    badfracfluxlong = ~goodfracfluxlong

    goodqfshort = ((basetable['qf_f212n'] > minqf) & (basetable['qf_f182m'] > minqf) & (basetable['qf_f187n'] > minqf))
    goodspreadshort = ((basetable['spread_model_f212n'] < maxspread) & (basetable['spread_model_f182m'] < maxspread) & (basetable['spread_model_f187n'] < maxspread))
    goodfracfluxshort = ((basetable['fracflux_f212n'] > minfracflux) & (basetable['fracflux_f182m'] > minfracflux) & (basetable['fracflux_f187n'] > minfracflux))

    goodqfshort = np.array(goodqfshort & ~goodqfshort.mask)
    goodspreadshort = np.array(goodspreadshort & ~goodspreadshort.mask)
    goodfracfluxshort = np.array(goodfracfluxshort & ~goodfracfluxshort.mask)
    allgood_short = (goodqfshort & goodspreadshort & goodfracfluxshort)

    badqfshort = ~goodqfshort
    badspreadshort = ~goodspreadshort
    badfracfluxshort = ~goodfracfluxshort

    print(f"QFs: {goodqfshort.sum()} good short")
    print(f"QFs: {goodqflong.sum()} good long")

    # threshold = 0.1 arcsec
    oksep = np.logical_and.reduce([basetable[f'sep_{filtername}'] < 0.1*u.arcsec for filtername in filternames[1:]])
    print(f"Found {oksep.sum()} of {len(oksep)} sources with separations < 0.1 arcsec")
    oklong = oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) & (~badqflong) & (~badspreadlong) & (~badfracfluxlong)

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    abconv = (1*u.Jy).to(u.ABmag)
    filtconv410 = -2.5*np.log10(1/jfilts.loc['JWST/NIRCam.F410M']['ZeroPoint'])-abconv.value
    filtconv466 = -2.5*np.log10(1/jfilts.loc['JWST/NIRCam.F466N']['ZeroPoint'])-abconv.value
    zeropoint_offset_410_466 = filtconv410-filtconv466
    print(f'Offset between raw ABmag for F410M-F466N = {filtconv410} - {filtconv466} = {zeropoint_offset_410_466}')

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
                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2)**0.5) < (-0.75+zeropoint_offset_410_466))
                    )
    # this assertion is presumably because blue_410_466 was being computed from 410m405 before
    # assert (blue_410_466 & basetable['mag_ab_410m405'].mask).sum() == 0
    # now this is the correct assertion
    assert (blue_410m405_466 & basetable['mag_ab_410m405'].mask).sum() == 0
    blue_410_405 = oksep & ~any_saturated & (~(basetable['mag_ab_410m405'].mask)) & ((basetable['mag_ab_410m405'] - basetable['mag_ab_f405n']) + (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -2)
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
                        #& (~badqflong) & (~badspreadlong) & (~badfracfluxlong))
    detected = ((~basetable['mag_ab_f405n'].mask) &
                (~basetable['mag_ab_f410m'].mask) &
                (~basetable['mag_ab_f187n'].mask) &
                (~basetable['mag_ab_f182m'].mask))
    detected_allbands = ((~basetable['mag_ab_f405n'].mask) &
                         (~basetable['mag_ab_f410m'].mask) &
                         (~basetable['mag_ab_f466n'].mask) &
                         (~basetable['mag_ab_f212n'].mask) &
                         (~basetable['mag_ab_f187n'].mask) &
                         (~basetable['mag_ab_f182m'].mask))
    print(f"Strongly blue [410-466] sources: {blue_410_466.sum()}")
    print(f"Somewhat blue [410-466] sources: {slightly_blue_410_466.sum()}")
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

    exclude = (any_saturated | ~oksep | magerr_gtpt1 |
               basetable['mag_ab_f405n'].mask | basetable['mag_ab_f410m'].mask |
               badqflong | badfracfluxlong | badspreadlong)
    print(f"Excluding {exclude.sum()} of {exclude.size}")

    # "bad" was totally broken; (bad & all_good) is very nonzero
    # bad = (any_saturated | ~oksep | magerr_gtpt1 | basetable['mag_ab_f212n'].mask |
    #        basetable['mag_ab_f410m'].mask | badqflong | badfracfluxlong |
    #        badspreadlong | badqfshort | badfracfluxshort | badspreadshort)
    bad = ~all_good
    print("'Bad' sources are those where _any_ filter is masked out")
    print(f"Not-bad:{(~bad).sum()}, bad: {bad.sum()},"# bad.mask: {bad.mask.sum()},"
          f" len(bad):{len(bad)}, len(table):{len(basetable)}.")


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
    recomb_excess_over_212 = c212_405 > c187_212 * (4/3.) + 0.35

    return locals()


# def main_dao(basetable, ww):
#
#     # empirical test: these sources are almost certainly saturated in f410m =(
#     saturated_f410m = ((basetable['mag_ab_f410m'] < (13.9-3.1)) &
#                        (basetable['mag_ab_f410m'] - basetable['mag_ab_f405n'] >
#                         0))
#     basetable['mag_ab_f410m'].mask[saturated_f410m] = True
#     basetable['flux_f410m'].mask[saturated_f410m] = True
#
#
#     # FITS tables can't mask boolean columns
#     # so, we have to mask the saturated mask using the mask on the flux for the filter
#     any_saturated_ = [basetable[f'near_saturated_{x}_{x}'] & ~basetable[f'flux_{x}'].mask for x in filternames]
#
#     any_saturated = any_saturated_[0]
#     for col in any_saturated_[1:]:
#         print(f"{col.sum()} saturated in {col.name}")
#         any_saturated = any_saturated | col
#     print(f"{any_saturated.sum()} near saturated out of {len(basetable)}.  That leaves {(~any_saturated).sum()} not near unsaturated")
#
#     any_replaced_saturated_ = [basetable[f'replaced_saturated_{x}'] &
#                                ~basetable[f'flux_{x}'].mask for x in filternames]
#     any_replaced_saturated = any_replaced_saturated_[0]
#     for col in any_replaced_saturated_[1:]:
#         print(f"{col.sum()} saturated in {col.name}")
#         any_replaced_saturated = any_replaced_saturated | col
#     print(f"{any_replaced_saturated.sum()} saturated out of {len(basetable)}.  That leaves {(~any_replaced_saturated).sum()} unsaturated")
#
#     magerr_gtpt1 = np.logical_or.reduce([basetable[f'emag_ab_{filtername}'] > 0.2 for filtername in filternames])
#     magerr_gtpt1.sum()
#
#
#     for filt in filternames:
#         filt = filt.lower()
#         mask = basetable[f'mag_ab_{filt}'].mask
#
#         flagok = (basetable[f'flags_{filt}'] == 0)
#         basetable[f'good_{filt}'] = allok = flagok
#         print(f"Filter {filt} has {flagok.sum()} flags=0.")
#
#     all_good = np.all([basetable[f'good_{filt}'] for filt in filternames], axis=0)
#     any_good = np.any([basetable[f'good_{filt}'] for filt in filternames], axis=0)
#     long_good = np.all([basetable[f'good_{filt}'] for filt in filternames if 'f4' in filt], axis=0)
#     short_good = np.all([basetable[f'good_{filt}'] for filt in filternames if 'f4' not in filt], axis=0)
#     print(f"Of {len(all_good)} rows, {all_good.sum()} are good in all filters.")
#     print(f"Of {len(all_good)} rows, {long_good.sum()} are good in long filters.")
#     print(f"Of {len(all_good)} rows, {short_good.sum()} are good in short filters.")
#     print(f"Of {len(all_good)} rows, {any_good.sum()} are good in at least one filter.")
#
#     allgood_long = (basetable['good_f410m'] & basetable['good_f466n'] & basetable['good_f405n'])
#
#     allgood_short = (basetable['good_f212n'] & basetable['good_f187n'] & basetable['good_f182m'])
#
#     # threshold = 0.1 arcsec
#     oksep = np.logical_and.reduce([basetable[f'sep_{filtername}'] < 0.1*u.arcsec for filtername in filternames[1:]])
#     print(f"Found {oksep.sum()} of {len(oksep)} sources with separations < 0.1 arcsec")
#     oklong = oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask))
#
#     jfilts = SvoFps.get_filter_list('JWST')
#     jfilts.add_index('filterID')
#     abconv = (1*u.Jy).to(u.ABmag)
#     filtconv410 = -2.5*np.log10(1/jfilts.loc['JWST/NIRCam.F410M']['ZeroPoint'])-abconv.value
#     filtconv466 = -2.5*np.log10(1/jfilts.loc['JWST/NIRCam.F466N']['ZeroPoint'])-abconv.value
#     zeropoint_offset_410_466 = filtconv410-filtconv466
#     print(f'Offset between raw ABmag for F410M-F466N = {filtconv410} - {filtconv466} = {zeropoint_offset_410_466}')
#
#     veryblue_410m405_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) &
#                     ((basetable['mag_ab_410m405'] - basetable['mag_ab_f466n']) +
#                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2 + basetable['emag_ab_f405n']**2)**0.5 < (-1.75+zeropoint_offset_410_466)
#                     ))
#     veryblue_410_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_f410m'].mask)) &
#                     (((basetable['mag_ab_f410m'] - basetable['mag_ab_f466n']) +
#                       (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2)**0.5) < (-1.75+zeropoint_offset_410_466)
#                     ))
#
#     blue_410m405_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) &
#                     ((basetable['mag_ab_410m405'] - basetable['mag_ab_f466n']) +
#                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2 + basetable['emag_ab_f405n']**2)**0.5 < (-0.75+zeropoint_offset_410_466))
#                     )
#     blue_410_466 = (oksep & (~any_saturated) & (~(basetable['mag_ab_f410m'].mask)) &
#                     (((basetable['mag_ab_f410m'] - basetable['mag_ab_f466n']) +
#                       (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2)**0.5) < (-0.75+zeropoint_offset_410_466))
#                     )
#     slightly_blue_410_466 =  (oksep & (~any_saturated) & (~(basetable['mag_ab_410m405'].mask)) &
#                     ((basetable['mag_ab_410m405'] - basetable['mag_ab_f466n']) +
#                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f466n']**2 + basetable['emag_ab_f405n']**2)**0.5 < zeropoint_offset_410_466)
#                     )
#     assert (blue_410_466 & basetable['mag_ab_410m405'].mask).sum() == 0
#     blue_410_405 = oksep & ~any_saturated & (~(basetable['mag_ab_410m405'].mask)) & ((basetable['mag_ab_410m405'] - basetable['mag_ab_f405n']) + (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -2)
#     blue_405_410 = (oksep & ~any_saturated & (~(basetable['mag_ab_410m405'].mask)) &
#                     ((basetable['mag_ab_405m410'] - basetable['mag_ab_410m405']) +
#                      (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -1)
#                     & ~magerr_gtpt1)
#     blue_405_410b = (oksep & ~any_saturated & (basetable['flux_f405n'] > basetable['flux_f410m']) &
#                      (~(basetable['mag_ab_f405n'].mask)) &
#                      ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) +
#                       (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -0.5)
#                      & ~magerr_gtpt1)
#     blue_187_182 = (oksep & ~any_saturated & (basetable['flux_f187n'] > basetable['flux_f182m']) &
#                      (~(basetable['mag_ab_f187n'].mask)) &
#                      ((basetable['mag_ab_f187n'] - basetable['mag_ab_f182m']) +
#                       (basetable['emag_ab_f182m']**2 + basetable['emag_ab_f187n']**2)**0.5 < -1)
#                     & ~magerr_gtpt1)
#
#     blue_BrA_and_PaA = (oksep & ~any_saturated &
#                         (basetable['flux_f405n'] > basetable['flux_f410m']) &
#                         (basetable['flux_f187n'] > basetable['flux_f182m']) &
#                      (~(basetable['mag_ab_f405n'].mask)) &
#                      (~(basetable['mag_ab_f410m'].mask)) &
#                      (~(basetable['mag_ab_f187n'].mask)) &
#                      (~(basetable['mag_ab_f182m'].mask)) &
#                      ((basetable['mag_ab_f187n'] - basetable['mag_ab_f182m']) +
#                       (basetable['emag_ab_f182m']**2 + basetable['emag_ab_f187n']**2)**0.5 < -0.1) &
#                      ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) +
#                       (basetable['emag_ab_f410m']**2 + basetable['emag_ab_f405n']**2)**0.5 < -0.1)
#                         & ~magerr_gtpt1 &
#                         basetable['good_f405n'] &
#                         basetable['good_f410m'] &
#                         basetable['good_f187n'] &
#                         basetable['good_f182m']
#                        )
#     detected = ((~basetable['mag_ab_f405n'].mask) &
#                 (~basetable['mag_ab_f410m'].mask) &
#                 (~basetable['mag_ab_f187n'].mask) &
#                 (~basetable['mag_ab_f182m'].mask))
#     detected_allbands = ((~basetable['mag_ab_f405n'].mask) &
#                          (~basetable['mag_ab_f410m'].mask) &
#                          (~basetable['mag_ab_f466n'].mask) &
#                          (~basetable['mag_ab_f212n'].mask) &
#                          (~basetable['mag_ab_f187n'].mask) &
#                          (~basetable['mag_ab_f182m'].mask))
#     print(f"Strongly blue [410-466] sources: {blue_410_466.sum()}")
#     print(f"Somewhat blue [410-466] sources: {slightly_blue_410_466.sum()}")
#     print(oklong.sum(), blue_410_466.sum(), slightly_blue_410_466.sum(), blue_405_410.sum(), blue_405_410b.sum(), blue_BrA_and_PaA.sum(), detected.sum(), blue_BrA_and_PaA.sum() / detected.sum())
#
#     neg_405m410 = basetable['flux_jy_405m410'] < 0
#     print(f"Negative 405-410 colors: {neg_405m410.sum()}, Nonnegative: {(~neg_405m410).sum()}")
#
#     any_saturated |= saturated_f410m
#     all_good &= ~saturated_f410m
#
#     exclude = (any_saturated | ~oksep | magerr_gtpt1 |
#                basetable['mag_ab_f405n'].mask | basetable['mag_ab_f410m'].mask
#               )
#     print(f"Excluding {exclude.sum()} of {exclude.size}")
#
#     # "bad" was totally broken; (bad & all_good) is very nonzero
#     # bad = (any_saturated | ~oksep | magerr_gtpt1 | basetable['mag_ab_f212n'].mask |
#     #        basetable['mag_ab_f410m'].mask | badqflong | badfracfluxlong |
#     #        badspreadlong | badqfshort | badfracfluxshort | badspreadshort)
#     bad = ~all_good
#     print("'Bad' sources are those where _any_ filter is masked out")
#     print(f"Not-bad:{(~bad).sum()}, bad: {bad.sum()},"# bad.mask: {bad.mask.sum()},"
#           f" len(bad):{len(bad)}, len(table):{len(basetable)}.")
#
#
#     # Basic selections for CMD, CCD plotting
#     sel = reg.contains(basetable['skycoord_f410m'], ww)
#     sel &= basetable['sep_f466n'].quantity < 0.1*u.arcsec
#     sel &= basetable['sep_f405n'].quantity < 0.1*u.arcsec
#
#     def ccds_withiso(basetable=basetable, sel=sel, exclude=exclude, **kwargs):
#         return plot_tools.ccds_withiso(basetable=basetable, sel=sel, exclude=exclude, **kwargs)
#
#     def cmds_withiso(basetable=basetable, sel=sel, exclude=exclude, distance_modulus=distance_modulus, **kwargs):
#         return plot_tools.cmds_withiso(basetable=basetable, sel=sel, exclude=exclude, distance_modulus=distance_modulus, **kwargs)
#
#
#     sel = reg.contains(basetable['skycoord_f410m'], ww)
#     sel &= basetable['sep_f466n'].quantity < 0.1*u.arcsec
#     sel &= basetable['sep_f405n'].quantity < 0.1*u.arcsec
#
#     def ccds(basetable=basetable, sel=sel, **kwargs):
#         return plot_tools.ccds(basetable=basetable, sel=sel, **kwargs)
#
#     def cmds(basetable=basetable, sel=sel, **kwargs):
#         return plot_tools.cmds(basetable=basetable, sel=sel, **kwargs)
#
#     crds = basetable['skycoord_f410m']
#
#
#     # not sure these are legitimately bad?
#     # Feb 11, 2023: these are the same objects as 'weird blue'
#     # This is needed by some plots, but isn't obviously useful
#     badblue = blue_410_466 & ((basetable['mag_ab_f405n'] - basetable['mag_ab_f410m']) > 2)
#     #| ((basetable['mag_ab_f410m'] - basetable['mag_ab_f466n']) > -0.5) )
#
#     assert 'blue_410m405_466' in locals()
#     return locals()

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--module", dest="module", default='merged',
                      help="module to select", metavar="module")
    (options, args) = parser.parse_args()

    print(f"Selecting module {options.module}")

    # save nrca and nrcb result tables
    print()
    print("NRCA")
    from analysis_setup import fh_nrca as fh, ww410_nrca as ww410, ww410_nrca as ww
    result = main(basetable_nrca, ww=ww)
    globals().update({key+"_a": val for key, val in result.items()})

    print()
    print("NRCB")
    from analysis_setup import fh_nrcb as fh, ww410_nrcb as ww410, ww410_nrcb as ww
    result = main(basetable_nrcb, ww=ww)
    globals().update({key+"_b": val for key, val in result.items()})

    print()
    print("merged-reproject")
    from analysis_setup import fh_merged_reproject as fh, ww410_merged_reproject as ww410, ww410_merged_reproject as ww
    result = main(basetable_merged_reproject, ww=ww)
    globals().update({key+"_mr": val for key, val in result.items()})

    print()
    print("merged")
    from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
    result = main(basetable_merged, ww=ww)
    globals().update({key+"_m": val for key, val in result.items()})

    if options.module == 'nrca':
        from analysis_setup import fh_nrca as fh, ww410_nrca as ww410, ww410_nrca as ww
        result = main(basetable_nrca, ww=ww)
        globals().update(result)
        basetable = basetable_nrca
        print("Loaded nrca")
    elif options.module == 'nrcb':
        from analysis_setup import fh_nrcb as fh, ww410_nrcb as ww410, ww410_nrcb as ww
        result = main(basetable_nrcb, ww=ww)
        globals().update(result)
        basetable = basetable_nrcb
        print("Loaded nrcb")
    elif options.module == 'merged':
        from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        result = main(basetable_merged, ww=ww)
        globals().update(result)
        basetable = basetable_merged
        print("Loaded merged")
    elif options.module == 'merged-reproject':
        from analysis_setup import fh_merged_reproject as fh, ww410_merged_reproject as ww410, ww410_merged_reproject as ww
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
