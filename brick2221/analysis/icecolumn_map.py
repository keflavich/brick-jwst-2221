import os
from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import mpl_plot_templates
from molmass import Formula
import regions
import re
from astropy.io import fits
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
from scipy import stats
from astropy.visualization import simple_norm

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath, compute_molecular_column, molscomps
from brick2221.analysis.make_icecolumn_fig9 import calc_av, ev

from dust_extinction.averages import CT06_MWGC, G21_MWAvg, F11_MWGC
from dust_extinction.parameter_averages import G23

from astropy.wcs import WCS


def icecolumn_map(
             basetable,
             avfilts=['F182M', 'F410M'],
             ax=None, sel=None, ok=None,
             icemol='CO',
             atom='C',
             abundance=10**(8.7-12), # roughly extrapolated from Smartt 2001A%26A...367...86S
             title='H2O:CO:CO2 (10:1:1)',
             dmag_tbl=None,
             NHtoAV=2.21e21,
             av_start=17,
             scatter=True,
             hist=False,
             ext=CT06_MWGC(),
             color_filter1='F405N',
             color_filter2='F466N',
             debug=False,
             ):


    if ax is None:
        ax = pl.gca()
    fig = ax.get_figure()

    av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext, return_av=True)

    # Dynamic color calculation based on color_filter parameters
    filter1_wav = int(color_filter1[1:-1])/100. * u.um
    filter2_wav = int(color_filter2[1:-1])/100. * u.um
    E_V_color = (ext(filter2_wav) - ext(filter1_wav))

    measured_color = basetable[f'mag_ab_{color_filter1.lower()}'] - basetable[f'mag_ab_{color_filter2.lower()}']
    unextincted_color = measured_color + E_V_color * av

    inferred_molecular_column = compute_molecular_column(unextincted_1m2=unextincted_color,
                                                         dmag_tbl=dmag_tbl,
                                                         icemol=icemol,
                                                         filter1=color_filter1,
                                                         filter2=color_filter2,
                                                         verbose=debug)

    NH2_of_AV = NHtoAV / 2. * (av - av_start)

    abundance = inferred_molecular_column / NH2_of_AV

    sel = av > av_start

    if scatter:
        inds = np.argsort(inferred_molecular_column[sel])
        sc = ax.scatter(
            basetable['skycoord_f410m'][sel][inds].dec.deg,
            basetable['skycoord_f410m'][sel][inds].ra.deg,
            c=np.log10(inferred_molecular_column[sel][inds]),
            s=10, alpha=0.5,
            cmap='viridis_r',
            vmin=18,
            )
        cb = pl.colorbar(mappable=sc)
        cb.set_label('log10(CO column density)')

        ax.set_aspect('equal')
    elif hist:
        histdata, xbin, ybin = np.histogram2d(basetable['skycoord_f410m'][sel].ra.deg, basetable['skycoord_f410m'][sel].dec.deg,
                       bins=[50, 100], weights=inferred_molecular_column[sel])
        im = ax.imshow(histdata, origin='lower', norm=simple_norm(histdata, stretch='log', vmin=1e18))
        cb = pl.colorbar(mappable=im)
        cb.set_label('CO column density')
        #ax.set_aspect(0.5)


def main():

    basetable = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221_20250324.fits')
    measured_466m410 = basetable['mag_ab_f466n'] - basetable['mag_ab_f410m']
    bad_to_exclude = (basetable['mag_ab_f410m'] < 13.7) & ( (basetable['mag_ab_f405n'] - basetable['mag_ab_f410m'] < -0.2) )
    bad_to_exclude |= (basetable['mag_ab_f410m'] > 17) & ( (basetable['mag_ab_f405n'] - basetable['mag_ab_f410m'] < -1) )
    bad_to_exclude |= (basetable['mag_ab_f182m'] < 15.5)
    sel = ok = ok2221 = ~bad_to_exclude

    dmag_tbl = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
    dmag_tbl.add_index('composition')


    pl.figure(figsize=(12, 4))
    icecolumn_map(basetable=basetable, sel=ok2221, ok=ok2221, hist=False, scatter=True, dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:1)'])
    pl.savefig(f'{basepath}/figures/icecolumn_map_CO.png', bbox_inches='tight')

    pl.figure(figsize=(12, 4))
    icecolumn_map(basetable=basetable, sel=ok2221, ok=ok2221, hist=True, scatter=False, dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:1)'])
    pl.savefig(f'{basepath}/figures/icecolumn_map_CO_hist.png', bbox_inches='tight')
    pl.close('all')


if __name__ == '__main__':
    main()