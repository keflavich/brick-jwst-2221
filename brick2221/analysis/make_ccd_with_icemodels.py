import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import mpl_plot_templates
from molmass import Formula
import re
from astropy.wcs import WCS
from astropy.io import fits
from astropy import table
from astropy.table import Table
from regions import Regions
import regions
from dust_extinction.averages import CT06_MWGC, G21_MWAvg

from cycler import cycler
from tqdm.auto import tqdm

import time

from icemodels.colorcolordiagrams import plot_ccd_icemodels
from icemodels.core import composition_to_molweight

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis import plot_tools
from brick2221.analysis.make_icecolumn_fig9 import molscomps
from brick2221.analysis.iceage_fluxes import iceage_flxd, iceage_mags
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath

pl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'], ) * cycler(linestyle=['-', '--', ':', '-.'])

if 'basetable_merged1182_daophot' not in globals():
    t0 = time.time()
    basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
    result = load_table(basetable_merged1182_daophot, ww=ww)
    ok2221 = result['ok2221']
    ok1182 = result['ok1182'][ok2221]
    globals().update(result)
    basetable = basetable_merged1182_daophot[ok2221]


    # there are several bad data points in F182M that are brighter than 15.5 mag
    print(f"Loaded merged1182_daophot_basic_indivexp. Load time = {time.time()-t0:0.1f}s")

bad_to_exclude = (basetable['mag_ab_f410m'] < 13.7) & ( (basetable['mag_ab_f405n'] - basetable['mag_ab_f410m'] < -0.2) )
bad_to_exclude |= (basetable['mag_ab_f410m'] > 17) & ( (basetable['mag_ab_f405n'] - basetable['mag_ab_f410m'] < -1) )
bad_to_exclude |= (basetable['mag_ab_f182m'] < 15.5)
sel = ok = oksep_noJ[ok2221] & ~bad[ok2221] & (~bad_to_exclude)

if 'cloudccat' not in globals():
    import sys
    sys.path.append('/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament')
    t0 = time.time()
    print("Loading cloudccat.  ", end=' ', flush=True)
    import jwst_plots
    cloudccat = jwst_plots.make_cat_use()
    bad_to_exclude = (cloudccat.catalog['mag_ab_f410m'] < 13.7) & ( (cloudccat.catalog['mag_ab_f405n'] - cloudccat.catalog['mag_ab_f410m'] < -0.2) )
    bad_to_exclude |= (cloudccat.catalog['mag_ab_f410m'] > 17) & ( (cloudccat.catalog['mag_ab_f405n'] - cloudccat.catalog['mag_ab_f410m'] < -1) )
    bad_to_exclude |= (cloudccat.catalog['mag_ab_f182m'] < 15.5)
    cloudccat.catalog = cloudccat.catalog[~bad_to_exclude]
    print(f"Load time = {time.time()-t0:0.1f}s", flush=True)



def filter_selection_mask(cat):
    filters_with_detections = ['f480m', 'f410m', 'f405n', 'f360m']
    #filters_without_detections = ['f150w', 'f182m', 'f187n', 'f210m', 'f212n',]
    # 300, 360, 466 are optional
    mask_with = np.logical_and.reduce([~np.isnan(cat[f'mag_ab_{band}']) for band in filters_with_detections])
    #mask_without = np.logical_and.reduce([np.isnan(cat[f'mag_ab_{band}']) for band in filters_without_detections])
    mask = mask_with
    #mask = np.logical_and(maks_with, mask_without)

    return mask

def load_sgrb2cat():
    sgrb2cat_fn = '/orange/adamginsburg/jwst/sgrb2/NB/catalogs/crowdsource_nsky0_merged_photometry_tables_merged.fits'
    sgrb2cat = Table.read(sgrb2cat_fn)
    mask = filter_selection_mask(sgrb2cat)
    sgrb2cat = sgrb2cat[mask]
    return sgrb2cat


x = np.linspace(1.24*u.um, 5*u.um, 1000)
pp_ct06 = np.polyfit(x, CT06_MWGC()(x), 7)

def ext(x, model=CT06_MWGC()):
    if x > 1/model.x_range[1]*u.um and x < 1/model.x_range[0]*u.um:
        return model(x)
    else:
        return np.polyval(pp_ct06, x.value)

# local (solar neighborhood) values
# Asplund 2009 or https://ui.adsabs.harvard.edu/abs/2012A%26A...539A.143N/abstract
# uses HII region values
oxygen_abundance = 10**(9.3-12)
carbon_abundance = 10**(8.7-12) # = 5.01e-4

def plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -2.5, 1],
                            nh_to_av=2.21e21,
                            abundance_wrt_h2=None,
                            molids=None,
                            molcomps=None,
                            av_start=17,
                            max_column=None,
                            max_h2_column=None,
                            icemol='CO',
                            icemol2=None,
                            icemol2_col=None,
                            icemol2_abund=None,
                            ext=ext,
                            dmag_tbl=None,
                            temperature_id=0,
                            exclude=~ok,
                            iceage=False,
                            nirspec_archive=True,
                            iso_archive=False,
                            pure_ice_no_dust=False,
                            cloudc=False,
                            cloudccat=None,
                            plot_brick=True,
                            ax=None,
                            label_author=False,
                            label_temperature=False,
                            av_scale=30,
                            show_orion_2770=False,
                            hexbin=False,
                            ):
    """
    """
    if ax is None:
        ax = pl.gca()
    if plot_brick:
        plot_tools.ccd(basetable, ax=pl.gca(), color1=[x.lower() for x in color1],
                    color2=[x.lower() for x in color2], sel=False,
                    markersize=2,
                    ext=ext,
                    extvec_scale=av_scale,
                    head_width=0.1,
                    axlims=axlims,
                    allow_missing=True,
                    alpha=0.25,
                    hexbin=hexbin,
                    exclude=exclude)
    else:
        pl.xlabel(f"{color1[0]} - {color1[1]}")
        pl.ylabel(f"{color2[0]} - {color2[1]}")

    if cloudc and cloudccat is not None:

        ww = WCS(fits.getheader('/orange/adamginsburg/jwst/cloudc/images/F182_reproj_merged-fortricolor.fits'))
        crds_cloudc = cloudccat['skycoord_ref']
        cloudc_regions = [y for x in [
            '/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament/regions_/cloudc1.region',
            '/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament/regions_/cloudc2.region',
            '/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament/regions_/cloudd.region']
                    for y in regions.Regions.read(x)
        ]
        cloudc_sel = np.any([reg.contains(crds_cloudc, ww) for reg in cloudc_regions], axis=0)
        lactea_filament_regions = regions.Regions.read('/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament/regions_/filament_long.region')[0]
        lactea_sel = lactea_filament_regions.contains(crds_cloudc, ww)


        plot_tools.ccd(Table(cloudccat)[cloudc_sel], ax=pl.gca(), color1=[x.lower() for x in color1],
                       color2=[x.lower() for x in color2], sel=False,
                       ext=ext,
                       color='g',
                       alpha=0.5,
                       markersize=2,
                       extvec_scale=0,
                       head_width=0,
                       exclude=None,
                       zorder=-5,
                       selcolor=None,
                       label='Cloud C & D',
                       hexbin=hexbin
                       )

        plot_tools.ccd(Table(cloudccat)[lactea_sel], ax=pl.gca(), color1=[x.lower() for x in color1],
                       color2=[x.lower() for x in color2], sel=False,
                       ext=ext,
                       color='r',
                       alpha=0.5,
                       markersize=2,
                       extvec_scale=0,
                       head_width=0,
                       exclude=None,
                       zorder=-5,
                       selcolor=None,
                       label='3 kpc arm filament',
                       hexbin=hexbin
                       )

    if iceage:
        # av_iceage = 60
        # ext_iceage = G21_MWAvg()
        # av_iceage1 = ext_iceage(wavelength_of_filter(color1[0])) - ext_iceage(wavelength_of_filter(color1[1]))
        # av_iceage2 = ext_iceage(wavelength_of_filter(color2[0])) - ext_iceage(wavelength_of_filter(color2[1]))
        c1iceage = iceage_mags['JWST/NIRCam.'+color1[0]] - iceage_mags['JWST/NIRCam.'+color1[1]]
        c2iceage = iceage_mags['JWST/NIRCam.'+color2[0]] - iceage_mags['JWST/NIRCam.'+color2[1]]
        pl.scatter(c1iceage,
                   c2iceage,
                   s=25, c='r', marker='o')
    if nirspec_archive:
        nirspec_dir = '/orange/adamginsburg/jwst/spectra/mastDownload/JWST/'
        nirspecarchive_mags_original = Table.read(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits')

        if 'ISOKN' not in nirspecarchive_mags_original.colnames:
            checklist = Table.read(f'{nirspec_dir}/jwst_archive_spectra_checklist.csv')
            common_keys = ['Target', 'Program', 'Grating', 'Grating2', 'Object', 'SlitID',
                        'Observation', 'Visit', 'VisitGroup', 'Filename']
            nirspecarchive_mags = table.join(nirspecarchive_mags_original, checklist, keys=common_keys)
        else:
            nirspecarchive_mags = nirspecarchive_mags_original

        # 24th mag is roughly too faint in the wide bands
        c1nirspecarchive = nirspecarchive_mags['JWST/NIRCam.'+color1[0]] - nirspecarchive_mags['JWST/NIRCam.'+color1[1]]
        c2nirspecarchive = nirspecarchive_mags['JWST/NIRCam.'+color2[0]] - nirspecarchive_mags['JWST/NIRCam.'+color2[1]]
        ok = ((nirspecarchive_mags['JWST/NIRCam.'+color1[0]] != 0) &
              (nirspecarchive_mags['JWST/NIRCam.'+color1[1]] != 0) &
              (nirspecarchive_mags['JWST/NIRCam.'+color2[0]] != 0) &
              (nirspecarchive_mags['JWST/NIRCam.'+color2[1]] != 0) &
              (nirspecarchive_mags['JWST/NIRCam.'+color1[0]] < 20) &
              (nirspecarchive_mags['JWST/NIRCam.'+color1[1]] < 20) &
              (nirspecarchive_mags['JWST/NIRCam.'+color2[0]] < 20) &
              (nirspecarchive_mags['JWST/NIRCam.'+color2[1]] < 20) &
              ~(nirspecarchive_mags['JWST/NIRCam.'+color1[0]].mask) &
              ~(nirspecarchive_mags['JWST/NIRCam.'+color1[1]].mask) &
              ~(nirspecarchive_mags['JWST/NIRCam.'+color2[0]].mask) &
              ~(nirspecarchive_mags['JWST/NIRCam.'+color2[1]].mask))
        p3222 = nirspecarchive_mags['Program'] == 3222
        p5804 = nirspecarchive_mags['Program'] == 5804
        p1611 = nirspecarchive_mags['Program'] == 1611
        p2770 = nirspecarchive_mags['Program'] == 2770

        if np.all('n' in c.lower() or 'm' in c.lower() for c in color1+color2):
            ok = nirspecarchive_mags['ISOKN'] == 'y'
            assert ok.sum() > 0

            # too-faint and/or nan values are bad for orion
            orionok = ((nirspecarchive_mags['JWST/NIRCam.F410M'][p2770] < 20) &
                       (nirspecarchive_mags['neg_pixels'][p2770] < 5) &
                       ~(nirspecarchive_mags['emission_line'][p2770].astype(bool))
                       )
            if show_orion_2770:
                assert orionok.sum() > 0
            ok[p2770] &= orionok
            assert ok.sum() > 0
            #print(f'{ok.sum()} good values ')
        else:
            # use the above critera and _also_ use the narrowband ones
            ok &= nirspecarchive_mags['ISOKN'] == 'y'

        pl.scatter(c1nirspecarchive[ok & p3222],
                   c2nirspecarchive[ok & p3222],
                   s=25, c='r', marker='x', label='NIRSpec 3222 (I16293 bck)')
        pl.scatter(c1nirspecarchive[ok & p5804],
                   c2nirspecarchive[ok & p5804],
                   s=25, c='m', marker='+', label='NIRSpec 5804 (HOPS YSO)')
        pl.scatter(c1nirspecarchive[ok & p1611],
                   c2nirspecarchive[ok & p1611],
                   s=25, c='b', marker='1', label='NIRSpec 1611 (Serpens)')
        if show_orion_2770:
            pl.scatter(c1nirspecarchive[ok & p2770],
                    c2nirspecarchive[ok & p2770],
                    s=25, c='g', marker='x', label='NIRSpec 2770 (Orion)')

    if iso_archive:
        isodir = '/orange/adamginsburg/ice/iso/library/swsatlas/'
        isoarchive_mags = Table.read(f'{isodir}/iso_spectra_as_fluxes.fits')
        c1isoarchive = isoarchive_mags['JWST/NIRCam.'+color1[0]] - isoarchive_mags['JWST/NIRCam.'+color1[1]]
        c2isoarchive = isoarchive_mags['JWST/NIRCam.'+color2[0]] - isoarchive_mags['JWST/NIRCam.'+color2[1]]
        if not all(c1isoarchive == 0) and not all(c2isoarchive == 0) and int(color1[0][1:-1]) > 300 and int(color1[1][1:-1]) > 300 and int(color2[0][1:-1]) > 300 and int(color2[1][1:-1]) > 300:
            pl.scatter(c1isoarchive,
                       c2isoarchive,
                       s=25, c='b', marker='x', label='ISO')

    # Now use the model plotting from colorcolordiagrams
    if molcomps:
        return plot_ccd_icemodels(
            color1=color1,
            color2=color2,
            dmag_tbl=dmag_tbl,
            molcomps=molcomps,
            molids=molids,
            axlims=axlims,
            nh_to_av=nh_to_av,
            abundance_wrt_h2=abundance_wrt_h2,
            av_start=av_start,
            max_column=max_column,
            max_h2_column=max_h2_column,
            icemol=icemol,
            icemol2=icemol2,
            icemol2_col=icemol2_col,
            icemol2_abund=icemol2_abund,
            ext=ext,
            temperature_id=temperature_id,
            pure_ice_no_dust=pure_ice_no_dust,
            label_author=label_author,
            label_temperature=label_temperature,
        )
    else:
        return [None] * 8

mixes1 = [
    ('H2O 1', 25.0),
    ('CO 1', 25.0),
    ('CO2 1', 25.0),
    #('CH3OH 1', 25.0),
    #('CH3CH2OH 1', 25.0),
    #('H2O:CO (0.5:1)', 25.0),
    ('H2O:CO (1:1)', 25.0),
    ('H2O:CO (3:1)', 25.0),
    ('H2O:CO (5:1)', 25.0),
    #('H2O:CO (7:1)', 25.0),
    ('H2O:CO (10:1)', 25.0),
    #('H2O:CO (15:1)', 25.0),
    ('H2O:CO (20:1)', 25.0),
    ('H2O:CO:CO2:CH3OH (1:1:0.1:0.1)', 25.0),
    #('H2O:CO:CO2:CH3OH (1:1:0.1:1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
    #('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
    #('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
]

mixes2 = [
    ('H2O:CO:CO2 (1:1:1)', 25.0),
    #('H2O:CO:CO2 (3:1:0.5)', 25.0),
    ('H2O:CO:CO2 (2:1:1)', 25.0),
    ('H2O:CO:CO2 (3:1:1)', 25.0),
    ('H2O:CO:CO2 (5:1:1)', 25.0),
    ('H2O:CO:CO2 (10:1:1)', 25.0),
    ('H2O:CO:CO2 (10:1:0.5)', 25.0),
    ('H2O:CO:CO2 (15:1:1)', 25.0),
    ('H2O:CO:CO2 (20:1:1)', 25.0),
    # ('H2O:CO (1:1)', 25.0),
    # ('H2O:CO (3:1)', 25.0),
    # ('H2O:CO (5:1)', 25.0),
    # ('H2O:CO (10:1)', 25.0),
    # ('H2O:CO (15:1)', 25.0),
    # ('H2O:CO (20:1)', 25.0),
    ('H2O:CO:CO2:CH3OH (1:1:1:1)', 25.0),
    ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:1:1:1)', 25.0),
]

molcomps_ch3 = [# comes from LIDA fraser; is bad ('CO:CH3OH 1:1', 15.0),
                ('CO:HCOOH 1:1', 14.0),
                #('CO:CH3CHO (20:1)', 15.0),
                ('CO:CH3OH:CH3CHO (20:20:1)', 15.0),
                ('CO:CH3OH:CH3CH2OH (20:20:1)', 15.0),
                ('CO:CH3OCH3 (20:1)', 15.0),
                ('CO:CH3OH:CH3OCH3 (20:20:1)', 15.0),
                ('H2O:CO:CO2:CH3OH (1:1:0.1:0.1)', 25.0),
                ('H2O:CO:CO2:CH3OH (1:1:0.1:1)', 25.0),
                ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
                ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)', 25.0),
                ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
                ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)', 25.0),
                ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
               ]

colors_and_lims = ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 2.5, -2.0, 0.5)),
                   (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 2.5, -2.0, 0.5)),
                   (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 11, -0.5, 1.5)),
                   (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                   (['F200W', 'F444W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                   (['F182M', 'F212N'], ['F356W', 'F444W'], (0, 2.5, -0.5, 1.5)),
                   (['F115W', 'F200W'], ['F200W', 'F444W'], (0, 11, -0.5, 4.5)),
                   (['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                   (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 2.5, -0.1, 3.0)),
                   (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 2.5, -0.1, 3.5)),
                   (['F182M', 'F212N'], ['F212N', 'F405N'], (0, 2.5, -0.1, 3.5)),
                   (['F162M', 'F210M'], ['F360M', 'F480M'], (-0.2, 10, -1, 2.5)),
                   (['F182M', 'F212N'], ['F466N', 'F480M'], (-0.2, 10, -1, 2.5)),
                   (['F182M', 'F212N'], ['F405N', 'F410M'], (0, 2.5, -0.5, 0.2)),
                   (['F200W', 'F212N'], ['F405N', 'F410M'], (-1, 1, -0.4, 0.2)),
                   (['F115W', 'F182M'], ['F182M', 'F212N'], (-0.1, 11, 0, 2.0)),
                   (['F115W', 'F200W'], ['F182M', 'F212N'], (-0.1, 11, 0, 2.0)),
                   (['F182M', 'F212N'], ['F212N', 'F405N'], (0, 2, -0.5, 3)),
                   (['F405N', 'F466N'], ['F356W', 'F444W'], (-1.5, 1.0, -0.5, 1.5)),
                   (['F405N', 'F410M'], ['F356W', 'F444W'], (-0.5, 0.5, -0.5, 1.5)),
                   (['F405N', 'F410M'], ['F405N', 'F466N'], (-0.5, 0.5, -1.5, 1.0)),
                   (['F356W', 'F405N'], ['F405N', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                   # these three lines are to check for offsets in the 4-micron filter colors
                   (['F182M', 'F212N'], ['F410M', 'F444W'], (0, 3, -0.5, 0.5)),
                   (['F182M', 'F212N'], ['F405N', 'F444W'], (0, 3, -0.5, 0.5)),
                   (['F182M', 'F212N'], ['F466N', 'F444W'], (0, 3, -0.5, 1.5)),
                   (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 3.0)),
                   (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 3, -0.1, 3.5)),
                   (['F182M', 'F212N'], ['F212N', 'F405N'], (0, 3, -0.1, 3.5)),
                   (['F162M', 'F210M'], ['F360M', 'F480M'], (-0.2, 10, -1, 2.5)),
                   (['F182M', 'F212N'], ['F466N', 'F480M'], (-0.2, 10, -1, 2.5)),
                   (['F405N', 'F466N'], ['F356W', 'F444W'], (-1.5, 1.0, -0.5, 1.5)),
                   (['F405N', 'F410M'], ['F356W', 'F444W'], (-0.5, 0.5, -0.5, 1.5)),
                  )

co_abundance_wrt_h2 = 2.5e-4 # 2025-07-25 based on appendix icemix

if __name__ == "__main__":

    import importlib as imp
    import icemodels.colorcolordiagrams
    imp.reload(icemodels.colorcolordiagrams)
    from icemodels.colorcolordiagrams import plot_ccd_icemodels

    pl.rcParams['figure.figsize'] = (4, 4)
    pl.rcParams['figure.dpi'] = 300
    pl.rcParams['font.size'] = 10

    legend_kwargs = dict(loc='lower left', bbox_to_anchor=(0.0, 1.06,))

    if 'dmag_all' not in globals():
        dmag_all = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
        dmag_all.add_index('mol_id')
        dmag_all.add_index('composition')
        dmag_all.add_index('temperature')
        dmag_all.add_index('database')
        dmag_all.add_index('author')

    assert 'F277W' in dmag_all.colnames, f'F277W not in dmag_all.colnames'


    def texify_exponent(x):
        exp = int(f'{x:.1e}'[-3:])
        lead = float(f'{x:.1e}'.split("e")[0])
        return f'{lead} \\times 10^{{{exp}}}'


    # compare percents
    # molcomps inherited from mixes2
    color1, color2, lims = (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 2.1, -2.0, 0.5))
    max_h2_column = 5e22
    # 2e-3 is the absolute extreme, putting 100% of carbon in CO ice and having GC carbon = 2.5x Solar
    for co_to_h2 in (1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3):
        print(f'color1: {color1}, color2: {color2} co_to_h2: {co_to_h2} [mixes2 abundance comparisons]', flush=True)
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              dmag_tbl=dmag_all,
                                                                                              molcomps=mixes2,
                                                                                              abundance_wrt_h2=co_to_h2,
                                                                                              max_column=None,
                                                                                              max_h2_column=max_h2_column)
        pl.title(f"CO/H$_2 = {texify_exponent(co_to_h2)}$, $N_{{max}}(\\mathrm{{H}}_2) = {texify_exponent(max_h2_column)}$ cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2_cotoh2{co_to_h2:0.1e}_nolegend.png', bbox_inches='tight', )
        pl.close('all')

    # Fiducial value
    co_to_h2 = 2.5e-4



    # pure ices
    ice_abundance = {'CO 1': 1e-3, 'CO2 1': 1e-3, 'H2O 1': 1e-3, 'CH3OH 1': 1e-3, 'CH3CH2OH 1': 1e-3}
    for color1, color2, lims in colors_and_lims:
        print(f'color1: {color1}, color2: {color2} [pure ice comparisons]', flush=True)
        pl.figure();
        molcomps = [('CO 1', 25.0),
                    ('CO2 1', 25.0),
                    ('H2O 1', 25.0),
                    ('CH3OH 1', 25.0),
                    ('CH3CH2OH 1', 30.0), ]
        for ii, molcomp in enumerate(molcomps):
            a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                                molcomps=[molcomp],
                                                                                                dmag_tbl=dmag_all,
                                                                                                abundance_wrt_h2=ice_abundance[molcomp[0]],
                                                                                                icemol=molcomp[0].split(' ')[0],
                                                                                                max_column=None,
                                                                                                nirspec_archive=False,
                                                                                                iso_archive=False,
                                                                                                av_scale=(30 if ii == 0 else 0),
                                                                                                max_h2_column=max_h2_column)
        pl.legend(**legend_kwargs)
        pl.title(f"ice/H$_2 = 10^{{-3}}$, $N_{{max}}(\\mathrm{{H}}_2) = {texify_exponent(max_h2_column)}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_pureices.png', bbox_inches='tight', )

        pl.close('all')

    for color1, color2, lims in colors_and_lims:
        print(f'color1: {color1}, color2: {color2} [mixes1 comparisons]', flush=True)
        pl.figure();
        molcomps = mixes1
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=molcomps,
                                                                                              dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                              abundance_wrt_h2=co_to_h2,
                                                                                              max_column=None,
                                                                                              max_h2_column=max_h2_column)
        pl.legend(**legend_kwargs)
        pl.title(f"CO/H$_2 = {texify_exponent(co_to_h2)}$, $N_{{max}}(\\mathrm{{H}}_2) = {texify_exponent(max_h2_column)}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes1.png', bbox_inches='tight', )

        pl.close('all')

    for color1, color2, lims in ((['F115W', 'F200W'], ['F356W', 'F444W'], (0, 15, -0.5, 1.5)),
                                 (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                                 (['F200W', 'F444W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                                 (['F405N', 'F466N'], ['F356W', 'F444W'], (-1.5, 1.0, -0.5, 1.5)),
                                 (['F182M', 'F212N'], ['F405N', 'F410M'], (0, 3, -0.5, 0.2)),
                                 (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 3, -2.0, 0.5)),
                                 ):
        print(f'color1: {color1}, color2: {color2} [CH3 mixes comparisons]', flush=True)
        pl.figure()

        molcomps = molcomps_ch3
        for mc, tt in molcomps:
            assert len(dmag_all.loc['composition', mc]) > 0, f"Composition {mc} not found in dmag_all"
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                            molcomps=molcomps,
                                                                                            dmag_tbl=dmag_all,
                                                                                            abundance_wrt_h2=co_abundance_wrt_h2,
                                                                                            max_column=None,
                                                                                            max_h2_column=max_h2_column)
        pl.legend(**legend_kwargs)
        pl.axis(lims);
        pl.title(f"CO/H$_2 = {texify_exponent(co_abundance_wrt_h2)}$, $N_{{max}}(\\mathrm{{H}}_2) = {texify_exponent(max_h2_column)}$");
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_ch3mixes.png', bbox_inches='tight', )
        pl.close('all')

    for show_orion_2770, suffix_orion, show_iso, suffix_iso in zip([True, False, False],
                                                                   ['_orion', '', ''],
                                                                   [False, False, True],
                                                                   ['', '', '_iso']):
        pl.close('all')
        for color1, color2, lims in colors_and_lims:
            print(f'color1: {color1}, color2: {color2} {suffix_orion}{suffix_iso} [mixes2 comparisons]', flush=True)
            pl.figure();
            molcomps = mixes2
            a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                                molcomps=molcomps,
                                                                                                dmag_tbl=dmag_all.loc['database', 'mymix'],
            # molids=[0,1,2,3,4,5,18,24,25,26,27],
                                                                                            #abundance=3e-4,
                                                                                            abundance_wrt_h2=co_abundance_wrt_h2,
                                                                                            max_column=None,
                                                                                            max_h2_column=5e22,
                                                                                            iso_archive=show_iso,
                                                                                            show_orion_2770=show_orion_2770,)
            #pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
            pl.axis(lims);
            pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2{suffix_orion}{suffix_iso}_nolegend.png', bbox_inches='tight', )
            pl.legend(**legend_kwargs)
            pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2{suffix_orion}{suffix_iso}.png', bbox_inches='tight', )

            pl.figure()
            a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                                molcomps=[], iso_archive=show_iso, show_orion_2770=show_orion_2770,)
            pl.legend(**legend_kwargs)
            #pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
            pl.axis(lims);
            pl.savefig(f'{basepath}/figures/CCD_without_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}{suffix_orion}{suffix_iso}.png', bbox_inches='tight', )

            try:
                pl.figure()
                a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                                    molcomps=molcomps,
                # molids=[0,1,2,3,4,5,18,24,25,26,27],
                                                                                                #abundance=3e-4,
                                                                                                abundance_wrt_h2=co_abundance_wrt_h2,
                                                                                                cloudc=True,
                                                                                                dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                                nirspec_archive=False,
                                                                                                cloudccat=cloudccat.catalog,
                                                                                                iso_archive=show_iso,
                                                                                                show_orion_2770=show_orion_2770,
                                                                                                max_column=None,
                                                                                                max_h2_column=5e22,
                                                                                                )
                pl.axis(lims);
                pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2_withCloudC{suffix_orion}{suffix_iso}_nolegend.png', bbox_inches='tight', )
                pl.legend(**legend_kwargs)
                #pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
                pl.axis(lims);
                pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2_withCloudC{suffix_orion}{suffix_iso}.png', bbox_inches='tight', )

                pl.figure()
                a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                                    molcomps=[],
                                                                                                    cloudc=True,
                                                                                                    dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                                    nirspec_archive=False,
                                                                                                    cloudccat=cloudccat.catalog,
                                                                                                    iso_archive=show_iso,
                                                                                                    show_orion_2770=show_orion_2770,
                                                                                                    )
                pl.legend(**legend_kwargs)
                #pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
                pl.axis(lims);
                pl.savefig(f'{basepath}/figures/CCD_without_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_withCloudC{suffix_orion}{suffix_iso}.png', bbox_inches='tight', )

            except KeyError:
                print(f"No cloudc for {color1} {color2}")
                continue

            pl.close('all')


    # Westerlund 1
    for color1, color2, lims in (
                                 (['F150W', 'F444W'], ['F444W', 'F466N'], (-0.5, 8, -0.5, 0.3)),
                                 (['F212N', 'F444W'], ['F444W', 'F466N'], (-0.5, 6, -0.75, 0.3)),
                                 (['F212N', 'F277W'], ['F444W', 'F466N'], (-0.5, 4, -0.5, 0.3)),
                                 (['F150W', 'F212N'], ['F277W', 'F323N'], (-0.5, 4, -1.0, 1.0)),
                                 (['F150W', 'F212N'], ['F212N', 'F323N'], (-0.5, 4, -1.0, 2.0)),
        ):
        print(f'color1: {color1}, color2: {color2} [Westerlund 1 comparisons]')
        pl.figure()

        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=molcomps,
                                                                                              av_start=10,
                                                                                              dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                              ext=G21_MWAvg(),
                                                                                              iceage=False,
                                                                                              plot_brick=False,
                                                                                              abundance_wrt_h2=co_to_h2,
                                                                                              max_column=None,
                                                                                              max_h2_column=max_h2_column)
        pl.legend(**legend_kwargs)
        pl.title(f"CO/H$_2 = {texify_exponent(co_to_h2)}$, $N_{{max}}(\\mathrm{{H}}_2) = {texify_exponent(max_h2_column)}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2_Wd1.png', bbox_inches='tight', )

    pl.close('all')

    percent = 25

    for color1, color2, lims in ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                 (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                                 (['F182M', 'F212N'], ['F356W', 'F444W'], (0, 3, -0.5, 1.5)),
                                 (['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 3.0)),
                                 (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 3, -0.1, 3.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F405N'], (0, 3, -0.1, 3.5)),
                                 (['F182M', 'F212N'], ['F405N', 'F410M'], (0, 3, -0.4, 0.2)),
                                 (['F200W', 'F212N'], ['F405N', 'F410M'], (-1, 1, -0.4, 0.2)),
                                ):
        print(f'color1: {color1}, color2: {color2} [OCN mixes comparisons]')
        pl.figure();
        molcomps = [('CO:OCN (1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:0.02)', 25.0),
                    ('H2O:CO:OCN (2:1:0.1)', 25.0),
                    ('H2O:CO:OCN (2:1:0.5)', 25.0), ]
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                            molcomps=molcomps,
                                                                                            dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                        #abundance=3e-4,
                                                                                        abundance_wrt_h2=(percent/100.)*carbon_abundance,
                                                                                        max_column=5e19)
        pl.legend(**legend_kwargs)
        pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {5e19:.2e} cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_OCNmixes.png', bbox_inches='tight', )
        #print(f"Saved {color1} {color2} ccd plot with OCN mixes using {molcomps}")

    percent = 25

    for color1, color2, lims in ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 3, -1.5, 1.0)),
                                 #(['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                 #(['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 3.0)),
                                 (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 3, -0.1, 3.5)),
                                ):
        print(f'color1: {color1}, color2: {color2} [OCN mixes comparisons w/ CloudC]')
        pl.figure();
        molcomps = [('CO:OCN (1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:0.02)', 25.0),
                    ('H2O:CO:OCN (2:1:0.1)', 25.0),
                    ('H2O:CO:OCN (2:1:0.5)', 25.0), ]
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                            molcomps=molcomps,
                                                                                            dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                        #abundance_wrt_h2=3e-4,
                                                                                        abundance_wrt_h2=co_to_h2,
                                                                                        cloudc=True,
                                                                                        nirspec_archive=False,
                                                                                        cloudccat=cloudccat.catalog,
                                                                                        max_column=None,
                                                                                        max_h2_column=max_h2_column)
        pl.legend(**legend_kwargs)
        pl.title(f"CO/H$_2 = {texify_exponent(co_to_h2)}$, $N_{{max}}(\\mathrm{{H}}_2) = {texify_exponent(max_h2_column)}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_OCNmixes_withCloudC.png', bbox_inches='tight', )
        print(f"Saved {color1} {color2} ccd plot with OCN mixes using {molcomps}")



    pl.figure()
    percent_ice = 20
    color1 = ['F182M', 'F212N']
    color2 = ['F405N', 'F410M']
    molcomps = [
        ('Hudgins', ('CO2 (1)', '70')),
        ('Gerakines', ('CO2 (1)', '70')),
        ('Hudgins', ('CO2 (1)', '10')),
        ('Ehrenfreund', ('CO2 (1)', '10')),
        ('Hudgins', ('CO2 (1)', '30')),
        ('Hudgins', ('CO2 (1)', '50')),
        ('Ehrenfreund', ('CO2 (1)', '50')),
        ('Gerakines', ('CO2 (1)', '8')),
    ]
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(
        color1, color2,
        molcomps=molcomps,
        dmag_tbl=dmag_all,
        axlims=(0, 3, -0.5, 0.2),
        abundance_wrt_h2=(percent_ice/100.)*carbon_abundance,
        icemol='CO2',
        max_column=2e19,
        label_author=True,
        label_temperature=True,
    )
    pl.legend(**legend_kwargs)
    pl.title(f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$")
    pl.savefig(f'{basepath}/figures/CCD_icemodel_F182M-F212N_F405N-F410M_CO2only.png', bbox_inches='tight', )
    pl.close()
