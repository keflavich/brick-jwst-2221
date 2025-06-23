"""
What if we use F212N as the reference filter?
"""
from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import mpl_plot_templates
from molmass import Formula
import re
import os

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath

from icemodels.core import composition_to_molweight

from brick2221.analysis.make_icecolumn_fig9 import plot_brandt_model, molscomps, makeplot as makeplot_orig

from dust_extinction.averages import CT06_MWGC, G21_MWAvg
from dust_extinction.parameter_averages import G23

from brick2221.analysis.analysis_setup import basepath, compute_molecular_column, molscomps


if os.path.exists(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221_20250324.fits'):
    basetable = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221_20250324.fits')
    print("Loaded merged1182_daophot_basic_indivexp (2025-03-24 version)")
else:
    basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
    result = load_table(basetable_merged1182_daophot, ww=ww)
    ok2221 = result['ok2221']
    ok1182 = result['ok1182'][ok2221]
    #globals().update(result)
    basetable = basetable_merged1182_daophot[ok2221]
    del basetable_merged1182_daophot
    del result
    print("Loaded merged1182_daophot_basic_indivexp")

sel = ok = ok2221 = np.ones(len(basetable), dtype=bool)

mol = 'COplusH2O'
dmag_tbl = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
dmag_tbl.add_index('composition')

def makeplot(avfilts=['F182M', 'F212N'],
             ax=None, sel=ok2221, ok=ok2221, alpha=0.5,
             icemol='CO',
             abundance=10**(8.7-12), # roughly extrapolated from Smartt 2001A%26A...367...86S
             title='H2O:CO:CO2 (10:1:1)',
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:1)'],
             dereddened=False,
             NtoAV=2.21e21, # 1.8e22, #
             plot_brandt=True,
             av_start=15,
             logy=True,
             **kwargs  # Accept additional parameters
             ):
    """Wrapper function that calls the original makeplot with F212N-F466N color configuration"""

    # Call the original makeplot function
    av, inferred_molecular_column, ax = makeplot_orig(
        avfilts=avfilts,
        ax=ax,
        sel=sel,
        ok=ok,
        alpha=alpha,
        icemol=icemol,
        abundance=abundance,
        title=title,
        dmag_tbl=dmag_tbl,
        plot_brandt=plot_brandt,
        NHtoAV=NtoAV,
        av_start=av_start,
        color_filter1='F212N',  # Second filter in color
        color_filter2='F466N',  # First filter in color ( F212N - F466N )
        logy=logy,  # This function expects log y values
        **kwargs  # Pass through any additional keyword arguments
    )

    # Add dereddened functionality if requested
    if dereddened:
        # Calculate the extinction correction for F212N-F466N color
        try:
            av_wavelengths = [int(avf[1:-1])/100. * u.um for avf in avfilts]
            E_V = (CT06_MWGC()(av_wavelengths[0]) - CT06_MWGC()(av_wavelengths[1]))
        except ValueError:
            print("Using G21_MWAvg() instead of CT06_MWGC()")
            E_V = (G21_MWAvg()(av_wavelengths[0]) - CT06_MWGC()(av_wavelengths[1]))

        E_V_212_466 = (CT06_MWGC()(2.12*u.um) - CT06_MWGC()(4.66*u.um))

        # Calculate AV from the extinction filters
        av_calc = (basetable[f'mag_ab_{avfilts[0].lower()}'] - basetable[f'mag_ab_{avfilts[1].lower()}']) / E_V

        # Calculate observed and un-extincted F466N-F212N color
        measured_466m212 = basetable['mag_ab_f466n'] - basetable['mag_ab_f212n']
        unextincted_466m212 = measured_466m212 + E_V_212_466 * av_calc

        # Create dereddened plot
        pl.figure(2)
        pl.scatter(basetable[f'mag_ab_{avfilts[0].lower()}'][sel & ok] - basetable[f'mag_ab_{avfilts[1].lower()}'][sel & ok],
                   unextincted_466m212[sel & ok], marker='.', s=0.5, alpha=alpha)
        pl.xlabel(f"{avfilts[0]}-{avfilts[1]}")
        pl.ylabel(f"un-extincted F466N - F212N")
        pl.grid()

    return av, inferred_molecular_column, ax

def main():

    dmag_co = dmag_tbl.loc['CO 1']

    carbon_abundance = 10**(8.7-12)
    oxygen_abundance = 10**(9.3-12)

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=carbon_abundance*0.5,
             title='CO',
             dereddened=True,
             dmag_tbl=dmag_co,
             ylim=(16.5, 19.5),
             legend_kwargs={'loc': 'lower right'},
    )
    plot_brandt_model(ax)

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=carbon_abundance*0.5,
             title='H2O:CO:CO2 (5:1:0.5)',
             ylim=(16.5, 19.5),
             legend_kwargs={'loc': 'lower right'},
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (5:1:0.5)'])
    plot_brandt_model(ax)
    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=oxygen_abundance*0.5,
             atom='O',
             title='H2O:CO:CO2 (5:1:0.5)',
             ylim=(16.5, 20),
             legend_kwargs={'loc': 'lower right'},
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (5:1:0.5)'])
    plot_brandt_model(ax, molecule='H2O')

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=carbon_abundance*0.5,
             title='H2O:CO:OCN (1:1:1)',
             ylim=(16.5, 19.5),
             legend_kwargs={'loc': 'lower right'},
             dmag_tbl=dmag_tbl.loc['H2O:CO:OCN (1:1:1)'])
    plot_brandt_model(ax)
    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=oxygen_abundance*0.5,
             atom='O',
             title='H2O:CO:OCN (1:1:1)',
             ylim=(16.5, 20),
             legend_kwargs={'loc': 'lower right'},
             dmag_tbl=dmag_tbl.loc['H2O:CO:OCN (1:1:1)'])
    plot_brandt_model(ax, molecule='H2O')

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=carbon_abundance*0.5,
             title='H2O:CO (3:1)',
             ylim=(16.5, 19.5),
             dmag_tbl=dmag_tbl.loc['H2O:CO (3:1)'])
    plot_brandt_model(ax)
    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=oxygen_abundance*0.5,
             title='H2O:CO (3:1)',
             ylim=(16.5, 19.5),
             legend_kwargs={'loc': 'lower right'},
             dmag_tbl=dmag_tbl.loc['H2O:CO (3:1)'])
    plot_brandt_model(ax, molecule='H2O')

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=carbon_abundance*0.5,
             title='H2O:CO:CO2 (10:1:1)',
             ylim=(16.5, 19.5),
             legend_kwargs={'loc': 'lower right'},
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:1)'])
    plot_brandt_model(ax)
    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=oxygen_abundance*0.5,
             atom='O',
             title='H2O:CO:CO2 (10:1:1)',
             ylim=(16.5, 20),
             legend_kwargs={'loc': 'lower right'},
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:1)'])
    plot_brandt_model(ax, molecule='H2O')


    makeplot(avfilts=['F182M', 'F410M'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'})
    makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'})
    makeplot(avfilts=['F187N', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'})
    makeplot(avfilts=['F187N', 'F405N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'})
    makeplot(avfilts=['F182M', 'F200W'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'})

    pl.close('all')
    makeplot(avfilts=['F115W', 'F200W'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(), ext=G23(Rv=5.5), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'})
    makeplot(avfilts=['F115W', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(), ext=G23(Rv=5.5), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'})


    makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=oxygen_abundance*0.5,
             atom='O',
             ylim=(16.5, 20),
             title='H2O',
             dmag_tbl=dmag_tbl.loc['H2O 1'],
             legend_kwargs={'loc': 'lower right'},
             )


    dmag_tbl_this = dmag_tbl.loc['composition', 'CO CO2 (100 70)']
    makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=carbon_abundance*0.5,
             title=dmag_tbl_this['composition'][0],
             ylim=(16.5, 20),
             dmag_tbl=dmag_tbl_this,
             legend_kwargs={'loc': 'lower right'},
             )



if __name__ == "__main__":
    main()
