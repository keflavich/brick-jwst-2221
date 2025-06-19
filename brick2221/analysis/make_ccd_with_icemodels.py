from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import mpl_plot_templates
from molmass import Formula
import re
from astropy.wcs import WCS
from astropy.io import fits
from regions import Regions
import regions

from dust_extinction.averages import CT06_MWGC, G21_MWAvg

from cycler import cycler
from tqdm.auto import tqdm

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
    basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
    result = load_table(basetable_merged1182_daophot, ww=ww)
    ok2221 = result['ok2221']
    ok1182 = result['ok1182'][ok2221]
    globals().update(result)
    basetable = basetable_merged1182_daophot[ok2221]
    # there are several bad data points in F182M that are brighter than 15.5 mag
    print("Loaded merged1182_daophot_basic_indivexp")

sel = ok = oksep_noJ[ok2221] & ~bad[ok2221] & (basetable['mag_ab_f182m'] > 15.5)

import sys
sys.path.append('/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament')
import jwst_plots
cloudccat = jwst_plots.make_cat_use()


dmag_co2 = Table.read(f'{basepath}/tables/CO2_ice_absorption_tables.ecsv')
dmag_co2.add_index('mol_id')
dmag_co2.add_index('composition')
dmag_co2.add_index('temperature')
dmag_co2.add_index('database')
dmag_co = Table.read(f'{basepath}/tables/CO_ice_absorption_tables.ecsv')
dmag_co.add_index('mol_id')
dmag_co.add_index('composition')
dmag_co.add_index('temperature')
dmag_co.add_index('database')
dmag_mine = Table.read(f'{basepath}/tables/H2O+CO_ice_absorption_tables.ecsv')
dmag_mine.add_index('mol_id')
dmag_mine.add_index('composition')
dmag_mine.add_index('temperature')
dmag_mine.add_index('database')
dmag_h2o = Table.read(f'{basepath}/tables/H2O_ice_absorption_tables.ecsv')
dmag_h2o.add_index('mol_id')
dmag_h2o.add_index('composition')
dmag_h2o.add_index('temperature')
dmag_h2o.add_index('database')
dmag_all = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
dmag_all.add_index('mol_id')
dmag_all.add_index('composition')
dmag_all.add_index('temperature')
dmag_all.add_index('database')
dmag_all.add_index('author')

assert 'F277W' in dmag_all.colnames, f'F277W not in dmag_all.colnames'

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
carbon_abundance = 10**(8.7-12)

percent_ice = 20

def plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -2.5, 1],
                            nh2_to_av=2.21e21,
                            abundance=(percent_ice/100)*carbon_abundance,
                            molids=np.unique(dmag_mine['mol_id']),
                            molcomps=None,
                            av_start=15,
                            max_column=2e20,
                            icemol='CO',
                            icemol2=None,
                            icemol2_col=None,
                            icemol2_abund=None,
                            ext=ext,
                            dmag_tbl=dmag_all,
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
                    allow_missing=True,
                    alpha=0.25,
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
                       label='Cloud C'
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
                       label='3 kpc arm filament'
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
        nirspecarchive_mags = Table.read(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits')

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
        p3222 = nirspecarchive_mags['Program'] == '03222'
        p5804 = nirspecarchive_mags['Program'] == '05804'
        p1611 = nirspecarchive_mags['Program'] == '01611'

        pl.scatter(c1nirspecarchive[ok & p3222],
                   c2nirspecarchive[ok & p3222],
                   s=25, c='r', marker='x', label='NIRSpec 3222 (I16293 bck)')
        pl.scatter(c1nirspecarchive[ok & p5804],
                   c2nirspecarchive[ok & p5804],
                   s=25, c='m', marker='+', label='NIRSpec 5804 (HOPS YSO)')
        pl.scatter(c1nirspecarchive[ok & p1611],
                   c2nirspecarchive[ok & p1611],
                   s=25, c='b', marker='1', label='NIRSpec 1611 (Serpens)')

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
            nh2_to_av=nh2_to_av,
            abundance=abundance,
            av_start=av_start,
            max_column=max_column,
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

if __name__ == "__main__":

    color1= ['F182M', 'F212N']
    color2= ['F410M', 'F466N']
    molcomps = [('H2O:CO (0.5:1)', 25.0),
                ('H2O:CO (1:1)', 25.0),
                ('H2O:CO (3:1)', 25.0),
                ('H2O:CO (5:1)', 25.0),
                ('H2O:CO (7:1)', 25.0),
                ('H2O:CO (10:1)', 25.0),
                ('H2O:CO (15:1)', 25.0),
                ('H2O:CO (20:1)', 25.0),
                ]
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, molcomps=molcomps,
                                                                                          dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                          axlims=[-0.1, 2.5, -2, 0.75],
                                                                                          icemol2='H2O', icemol2_col=1e19, abundance=(percent_ice/100)*carbon_abundance, icemol2_abund=(percent_ice/100)*oxygen_abundance, max_column=2e20)
    pl.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0, 0, 0))
    pl.title(f"{percent_ice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10^{{20}}$ cm$^{{-2}}$")# , dots show N(H$_2$O)=$10^{19}$ cm$^{-2}$")

    color1= ['F182M', 'F212N']
    color2= ['F212N', 'F466N']
    molcomps = [('H2O:CO (0.5:1)', 25.0),
                ('H2O:CO (1:1)', 25.0),
                ('H2O:CO (3:1)', 25.0),
                ('H2O:CO (5:1)', 25.0),
                ('H2O:CO (7:1)', 25.0),
                ('H2O:CO (10:1)', 25.0),
                ('H2O:CO (15:1)', 25.0),
                ('H2O:CO (20:1)', 25.0),
                ]
    pl.figure()
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -0.5, 4], molcomps=molcomps,
                                                                                          dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                          icemol2='H2O', icemol2_col=1e19, abundance=(percent_ice/100)*carbon_abundance, icemol2_abund=(percent_ice/100)*oxygen_abundance)
    pl.title(f"{percent_ice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10^{{20}}$ cm$^{{-2}}$")# , dots show N(H$_2$O)=$10^{19}$ cm$^{-2}$")
    pl.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0, 0, 0))
    pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_h2oco.png', bbox_inches='tight', dpi=150)

    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -0.5, 4], molcomps=molcomps,
                                                                                          dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                          icemol2='H2O',
                                                                                          icemol2_col=1e19,
                                                                                          abundance=(percent_ice/100)*carbon_abundance,
                                                                                          icemol2_abund=(percent_ice/100)*oxygen_abundance,
                                                                                          nirspec_archive=False,
                                                                                          )
    blue_410466 = basetable[basetable['mag_ab_f410m'] - basetable['mag_ab_f466n'] < -0.20]
    plot_tools.ccd(basetable, ax=pl.gca(), color1=[x.lower() for x in color1],
                    color2=[x.lower() for x in color2], sel=False,
                    markersize=2,
                    ext=ext,
                    extvec_scale=0,
                    head_width=0,
                    allow_missing=True,
                    color='r',
                    label='Brick [blue 410-466]',
                    exclude=~ok)
    pl.title(f"{percent_ice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10^{{20}}$ cm$^{{-2}}$")# , dots show N(H$_2$O)=$10^{19}$ cm$^{-2}$")
    pl.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0, 0, 0))
    pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_h2oco_withsel.png', bbox_inches='tight', dpi=150)


    # Search for models that can explain the wide-band filters
    percent = 25

    for color1, color2, lims in ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                 (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                                 (['F182M', 'F212N'], ['F356W', 'F444W'], (0, 3, -0.5, 1.5)),
                                 (['F115W', 'F200W'], ['F200W', 'F444W'], (0, 20, -0.5, 4.5)),
                                 (['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F356W', 'F405N'], ['F405N', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F356W', 'F466N'], ['F466N', 'F444W'], (-0.75, 1, -0.5, 1.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 3.0)),
                                 (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 3, -0.1, 3.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F405N'], (0, 3, -0.1, 3.5)),
                                 (['F162M', 'F210M'], ['F360M', 'F480M'], (-0.2, 10, -1, 2.5)),
                                 (['F182M', 'F212N'], ['F466N', 'F480M'], (-0.2, 10, -1, 2.5)),
                                ):
        pl.figure();
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=[
                                                                                                ('H2O:CO (0.5:1)', 25.0),
                                                                                                ('H2O:CO (1:1)', 25.0),
                                                                                                ('H2O:CO (3:1)', 25.0),
                                                                                                ('H2O:CO (5:1)', 25.0),
                                                                                                ('H2O:CO (7:1)', 25.0),
                                                                                                ('H2O:CO (10:1)', 25.0),
                                                                                                ('H2O:CO:CO2 (1:1:10)', 25.0),
                                                                                                ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
                                                                                                ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)', 25.0),
                                                                                                ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
                                                                                                ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)', 25.0),
                                                                                                ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
                                                                                                ],
                                                                                              dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                              abundance=(percent/100.)*carbon_abundance,
                                                                                              max_column=2e20)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.title(f"{percent}% of C in ice");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}.png', bbox_inches='tight', dpi=150)



    pl.figure()
    color1= ['F182M', 'F212N']
    color2= ['F410M', 'F466N']
    molcomps = [(x, 25.0) for x in ['H2O:CO (0.5:1)', 'H2O:CO (1:1)', 'H2O:CO (3:1)', 'H2O:CO (5:1)', 'H2O:CO (7:1)', 'H2O:CO (10:1)', 'H2O:CO (15:1)', 'H2O:CO (20:1)',]]
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                        molids=None,
                                                                                        molcomps=molcomps,
                                                                                        dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                        pure_ice_no_dust=False,
                                                                                    #abundance=3e-4,
                                                                                    #nh2_to_av=1e22,
                                                                                    abundance=0.5*carbon_abundance,
                                                                                    max_column=2e20)
    pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
    pl.title("50% of C in ice")
    pl.close('all')

    pl.figure()
    color1= ['F115W', 'F200W']
    color2= ['F356W', 'F444W']
    molcomps = [
                                            ('H2O:CO (0.5:1)', 25.0),
                                            ('H2O:CO (1:1)', 25.0),
                                            ('H2O:CO (3:1)', 25.0),
                                            ('H2O:CO (5:1)', 25.0),
                                            ('H2O:CO (7:1)', 25.0),
                                            ('H2O:CO (10:1)', 25.0),
                                            ('H2O:CO (15:1)', 25.0),
                                            ('H2O:CO (20:1)', 25.0),
    ]
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                    molcomps=molcomps,
                                                                                    #nh2_to_av=1e22,
                                                                                    abundance=0.5*carbon_abundance,
                                                                                    max_column=2e20)
    pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
    pl.axis((0, 10, -0.5, 1.5));
    pl.title("50% of CO in ice");
    pl.close('all')



    # this became redundant and I removed it but it reappeared. How much of the weirdass failures are caused by jank editors?
    # pl.figure()
    # color1= ['F115W', 'F200W']
    # color2= ['F356W', 'F444W']
    # a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, molids=[57, (64, 'lida'), 66, 67, 68, 69, 84, 86, 91, 94, 96],
    #                                                                                     #abundance=3e-4,
    #                                                                                     #nh2_to_av=1e22,
    #                                                                                     dmag_tbl=dmag_co,
    #                                                                                     abundance=0.25*carbon_abundance,
    #                                                                                     max_column=2e20)
    # pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
    # pl.axis((0, 15, -0.5, 1.5));
    # pl.title("25% of CO in ice")

    percent = 25

    for color1, color2, lims in ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                 (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                                 (['F115W', 'F200W'], ['F200W', 'F444W'], (0, 20, -0.5, 4.5)),
                                 (['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 3.0)),
                                 (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 3, -0.1, 3.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F405N'], (0, 3, -0.1, 3.5)),
                                 (['F162M', 'F210M'], ['F360M', 'F480M'], (-0.2, 10, -1, 2.5)),
                                 (['F162M', 'F212N'], ['F360M', 'F480M'], (-0.2, 10, -1, 2.5)),
                                 (['F182M', 'F212N'], ['F466N', 'F480M'], (-0.2, 10, -1, 2.5)),
                                 (['F182M', 'F212N'], ['F405N', 'F410M'], (0, 3, -0.5, 0.5)),
                                 (['F200W', 'F212N'], ['F405N', 'F410M'], (-1, 1, -0.4, 0.2)),
                                ):
        pl.figure();
        molcomps = [
                                            ('H2O:CO (0.5:1)', 25.0),
                                            ('H2O:CO (1:1)', 25.0),
                                            ('H2O:CO (3:1)', 25.0),
                                            ('H2O:CO (5:1)', 25.0),
                                            ('H2O:CO (7:1)', 25.0),
                                            ('H2O:CO (10:1)', 25.0),
                                            ('H2O:CO (15:1)', 25.0),
                                            ('H2O:CO (20:1)', 25.0),
                                            ('H2O:CO:CO2:CH3OH (1:1:0.1:0.1)', 25.0),
                                            ('H2O:CO:CO2:CH3OH (1:1:0.1:1)', 25.0),
                                            ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
                                            ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)', 25.0),
                                            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
                                            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)', 25.0),
                                            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
        ]
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=molcomps,
        # molids=[0,1,2,3,4,5,18,24,25,26,27],
                                                                                        #abundance=3e-4,
                                                                                        #nh2_to_av=1e22,
                                                                                        abundance=(percent/100.)*carbon_abundance,
                                                                                        max_column=2e20)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes1.png', bbox_inches='tight', dpi=150)

    pl.close('all')

    for color1, color2, lims in ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                 (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 5, -0.5, 1.5)),
                                 (['F182M', 'F212N'], ['F356W', 'F444W'], (0, 3, -0.5, 1.5)),
                                 (['F115W', 'F200W'], ['F200W', 'F444W'], (0, 20, -0.5, 4.5)),
                                 (['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 3.0)),
                                 (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 3, -0.1, 3.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F405N'], (0, 3, -0.1, 3.5)),
                                 (['F162M', 'F210M'], ['F360M', 'F480M'], (-0.2, 10, -1, 2.5)),
                                 (['F182M', 'F212N'], ['F466N', 'F480M'], (-0.2, 10, -1, 2.5)),
                                 (['F182M', 'F212N'], ['F405N', 'F410M'], (0, 3, -0.5, 0.5)),
                                 (['F200W', 'F212N'], ['F405N', 'F410M'], (-1, 1, -0.4, 0.2)),
                                ):
        pl.figure();
        molcomps = [
                                            ('H2O:CO:CO2 (1:1:1)', 25.0),
                                            ('H2O:CO:CO2 (3:1:0.5)', 25.0),
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
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=molcomps,
        # molids=[0,1,2,3,4,5,18,24,25,26,27],
                                                                                        #abundance=3e-4,
                                                                                        #nh2_to_av=1e22,
                                                                                        abundance=(percent/100.)*carbon_abundance,
                                                                                        max_column=2e20)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2.png', bbox_inches='tight', dpi=150)

        pl.figure()
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=[],)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        #pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_without_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}.png', bbox_inches='tight', dpi=150)

        try:
            pl.figure()
            a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                                molcomps=molcomps,
            # molids=[0,1,2,3,4,5,18,24,25,26,27],
                                                                                            #abundance=3e-4,
                                                                                            #nh2_to_av=1e22,
                                                                                            abundance=(percent/100.)*carbon_abundance,
                                                                                            cloudc=True,
                                                                                            cloudccat=cloudccat.catalog,
                                                                                            max_column=2e20)
            pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
            pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
            pl.axis(lims);
            pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2_withCloudC.png', bbox_inches='tight', dpi=150)

            pl.figure()
            a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                                molcomps=[],
                                                                                                cloudc=True,
                                                                                                cloudccat=cloudccat.catalog,)
            pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
            #pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
            pl.axis(lims);
            pl.savefig(f'{basepath}/figures/CCD_without_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_withCloudC.png', bbox_inches='tight', dpi=150)

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
        pl.figure()

        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=molcomps,
                                                                                              av_start=10,
                                                                                              ext=G21_MWAvg(),
                                                                                              iceage=False,
                                                                                              plot_brick=False,
                                                                                              abundance=(percent/100.)*carbon_abundance,
                                                                                        max_column=1e19)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {1e19:.2e} cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_mixes2_Wd1.png', bbox_inches='tight', dpi=150)

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
        pl.figure();
        molcomps = [('CO:OCN (1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:0.02)', 25.0),
                    ('H2O:CO:OCN (2:1:0.1)', 25.0),
                    ('H2O:CO:OCN (2:1:0.5)', 25.0), ]
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                            molcomps=molcomps,
                                                                                        #abundance=3e-4,
                                                                                        #nh2_to_av=1e22,
                                                                                        abundance=(percent/100.)*carbon_abundance,
                                                                                        max_column=5e19)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {5e19:.2e} cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_OCNmixes.png', bbox_inches='tight', dpi=150)
        #print(f"Saved {color1} {color2} ccd plot with OCN mixes using {molcomps}")

    percent = 25

    for color1, color2, lims in ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                 #(['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                 #(['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 3.0)),
                                 (['F182M', 'F212N'], ['F212N', 'F410M'], (0, 3, -0.1, 3.5)),
                                ):
        pl.figure();
        molcomps = [('CO:OCN (1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:1)', 25.0),
                    ('H2O:CO:OCN (1:1:0.02)', 25.0),
                    ('H2O:CO:OCN (2:1:0.1)', 25.0),
                    ('H2O:CO:OCN (2:1:0.5)', 25.0), ]
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                            molcomps=molcomps,
                                                                                        #abundance=3e-4,
                                                                                        #nh2_to_av=1e22,
                                                                                        abundance=(percent/100.)*carbon_abundance,
                                                                                        cloudc=True,
                                                                                        cloudccat=cloudccat.catalog,
                                                                                        max_column=5e19)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.title(f"{percent}% of C in ice, $N_{{max}}$ = {5e19:.2e} cm$^{{-2}}$");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}_OCNmixes_withCloudC.png', bbox_inches='tight', dpi=150)
        print(f"Saved {color1} {color2} ccd plot with OCN mixes using {molcomps}")


    for color1, color2, lims in ((['F115W', 'F200W'], ['F356W', 'F444W'], (0, 15, -0.5, 1.5)),
                                 (['F200W', 'F356W'], ['F356W', 'F444W'], (0, 15, -0.5, 1.5)),
                                 ):
        pl.figure()

        molcomps=[('CO:CH3OH 1:1', 15.0),
                  ('CO:HCOOH 1:1', 14.0),
                  ('CO:CH3CHO (20:1)', 15.0),
                  ('CO:CH3OH:CH3CHO (20:20:1)', 15.0),
                  ('CO:CH3OH:CH3CH2OH (20:20:1)', 15.0),
                  ('CO:CH3OCH3 (20:1)', 15.0),
                  ('CO:CH3OH:CH3OCH3 (20:20:1)', 15.0),
        ]
        for mc, tt in molcomps:
            assert len(dmag_all.loc['composition', mc]) > 0, f"Composition {mc} not found in dmag_all"
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                            #molids=[57, (64, 'lida'), 66, 67, 68, 69, 84, 86, 91, 94, 96],
                                                                                            #abundance=3e-4,
                                                                                            #nh2_to_av=1e22,
                                                                                            #dmag_tbl=dmag_co,
                                                                                            molcomps=molcomps,
                                                                                            dmag_tbl=dmag_all,
                                                                                            abundance=0.25*carbon_abundance,
                                                                                            max_column=2e20)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.axis((0, 15, -0.5, 1.5));
        pl.title(f"25% of C in ice, $N_{{max}}$ = {2e20:.2e} cm$^{{-2}}$");
        pl.close('all')

    pl.figure()
    color1 = ['F182M', 'F212N']
    color2 = ['F405N', 'F410M']
    molcomps = [
        ('Hudgins', ('CO2 (1)', '70K')),
        ('Gerakines', ('CO2 (1)', '70K')),
        ('Hudgins', ('CO2 (1)', '10K')),
        ('Ehrenfreund', ('CO2 (1)', '10K')),
        ('Hudgins', ('CO2 (1)', '30K')),
        ('Hudgins', ('CO2 (1)', '50K')),
        ('Ehrenfreund', ('CO2 (1)', '50K')),
        ('Gerakines', ('CO2 (1)', '8K')),
    ]
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(
        color1, color2,
        molcomps=molcomps,
        dmag_tbl=dmag_all,
        axlims=(0, 3, -0.5, 0.2),
        abundance=(percent_ice/100.)*carbon_abundance,
        icemol='CO2',
        max_column=2e19,
        label_author=True,
        label_temperature=True,
    )
    pl.legend(loc='upper left', bbox_to_anchor=(1, 1, 0, 0))
    pl.title(f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$")
    pl.savefig(f'{basepath}/figures/CCD_icemodel_F182M-F212N_F405N-F410M_CO2only.png', bbox_inches='tight', dpi=150)
    pl.close()
