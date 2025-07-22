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

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath, compute_molecular_column, molscomps
from brick2221.analysis.make_icecolumn_fig9 import calc_av, ev

from dust_extinction.averages import CT06_MWGC, G21_MWAvg, F11_MWGC
from dust_extinction.parameter_averages import G23

from astropy.wcs import WCS


def boxplot_abundance(
             basetable,
             avfilts=['F182M', 'F410M'],
             ax=None, sel=ok2221, ok=ok2221, alpha=0.5,
             icemol='CO',
             atom='C',
             abundance=10**(8.7-12), # roughly extrapolated from Smartt 2001A%26A...367...86S
             title='H2O:CO:CO2 (10:1:1)',
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:1)'],
             plot_brandt=True,
             NHtoAV=2.21e21,
             # av_start = 20 based on Jang, An, Whittet...
             av_start=20,
             xax='AV',
             cloudccat=None,
             sgrb2cat=None,
             scatter=True,
             contour=True,
             hexbin=False,
             ext=CT06_MWGC(),
             xlim=(-5, 105),
             ylim=None,
             nbins=50,
             nlevels=4,
             show_25_percent=True,
             legend_kwargs={'loc': 'best'},
             threshold=5,
             logy=True,
             color_filter1='F405N',
             color_filter2='F466N',
             use_abundance=False,
             suffix='',
             grid=False,
             hexbin_alpha=1.0,
             debug=False,
             av_max=80,
             av_spacing=5,
             ymin=-5,
             ymax=-3,
             ):


    if ax is None:
        ax = pl.gca()
    fig = ax.get_figure()

    if grid:
        ax.grid(True, linestyle='--', alpha=0.25, zorder=-50)

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

    av_bins = np.arange(av_start, av_max, av_spacing)
    av_bin_centers = av_bins[:-1] + av_spacing / 2

    if xax == 'AV':
        xtoplot = av
        extent = (av_start, av_max, ymin, ymax)
        bins, bin_centers = av_bins, av_bin_centers
        spacing = av_spacing
        start = av_start
        first_bin = 1
    elif xax == 'NH2':
        xtoplot = NH2_of_AV
        extent = (0, (av_max - av_start) * NHtoAV / 2., ymin, ymax)
        bins, bin_centers = (av_bins - av_start) * NHtoAV / 2., (av_bin_centers - av_start) * NHtoAV / 2.
        spacing = av_spacing * NHtoAV / 2.
        start = 0
        first_bin = 1
    elif xax == 'logNH2':
        xtoplot = np.log10(NH2_of_AV)
        extent = (21.5, 23, ymin, ymax)
        bins = np.linspace(extent[0], extent[1], 15)
        bin_centers = (bins[:-1] + bins[1:]) / 2.
        spacing = np.diff(bins)[0]
        print(f"logNH2 spacing: {spacing}")
        start = 0
        first_bin = 3
        #print(f"bins: {bins}.  bin_centers: {bin_centers}")

    ax.hexbin(xtoplot[av > av_start], np.log10(abundance[av > av_start]),
              gridsize=50,
              extent=extent,
              bins='log', mincnt=1, cmap='Reds', alpha=hexbin_alpha)
    ticklabels = ax.get_xticklabels()
    ticks = ax.get_xticks()

    medians = []
    bins_to_fit = []
    for bin_center in bin_centers:
        selection = (xtoplot > bin_center - spacing / 2) & (xtoplot < bin_center + spacing / 2)
        #print(bin_center, selection.sum())
        if selection.sum() > 10:
            ret = ax.boxplot(x=np.log10(abundance[selection]), notch=True,
                    widths=[spacing/2],
                    positions=[bin_center],
                    showfliers=False,
                    orientation='vertical')
            bins_to_fit.append(bin_center)
            medians.append(ret['medians'][0].get_ydata()[0])

    xtolabel = 'A$_V$' if xax == 'AV' else 'N$_{H_2}$' if xax == 'NH2' else 'log(N$_{H_2}$)'

    poly = np.polyfit(np.array(bins_to_fit[first_bin:]) - start, medians[first_bin:], 1)
    pl.plot(bin_centers, np.polyval(poly, bin_centers - start), color='black', linestyle=':',
            label=f'log(X) = {poly[1]:.2f} + {poly[0]:.3f}{xtolabel}')

    if xax == 'logNH2':
        print(f"X = {10**poly[1]:.2e} N$_{{H_2}}^{{{poly[1]}}}$")

    # extrapolate flatly for another few bins
    bins_to_fit.append(bin_centers[-1] + spacing)
    medians.append(medians[-1])
    bins_to_fit.append(bin_centers[-1] + spacing*2)
    medians.append(medians[-1])

    poly = np.polyfit(np.array(bins_to_fit[first_bin:]) - start, medians[first_bin:], 2)
    pl.plot(bin_centers, np.polyval(poly, bin_centers - start), color='black', linestyle='--',
            label=f'log(X) = {poly[2]:.2f} + {poly[1]:.3f}{xtolabel} + {poly[0]:.5f}{xtolabel}$^2$')

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_ylim(ymin, ymax)

    ax.fill_between([0, bins[first_bin]], ymin, ymax, color='white', alpha=0.5)
    if xax == 'AV':
        ax.set_xlabel('A$_V$')
    elif xax == 'NH2':
        ax.set_xlabel('NH$_2$')
    elif xax == 'logNH2':
        ax.set_xlabel('log(N$_{H_2}$)')
    ax.set_ylabel('log(X)')
    ax.set_xlim(extent[0], extent[1])

    ax.legend(loc='lower right')


def main():

    if os.path.exists(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221_20250324.fits'):
        basetable = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221_20250324.fits')
        print("Loaded merged1182_daophot_basic_indivexp (2025-03-24 version)")
    else:

        basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
        result = load_table(basetable_merged1182_daophot, ww=ww)
        ok2221 = result['ok2221']
        ok1182 = result['ok1182']
        #globals().update(result)
        basetable = basetable_merged1182_daophot[ok2221]
        ok1182 = ok1182[ok2221]
        del result
        print("Loaded merged1182_daophot_basic_indivexp")

        try:
            basetable.write(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221_20250324.fits', overwrite=False)
        except:
            pass

    measured_466m410 = basetable['mag_ab_f466n'] - basetable['mag_ab_f410m']

    sel = ok = ok2221 = np.ones(len(basetable), dtype=bool)

    dmag_tbl = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
    dmag_tbl.add_index('composition')


    pl.clf()
    boxplot_abundance(basetable=basetable, sel=ok2221, ok=ok2221, av_start=17, av_max=80, av_spacing=5, ymin=-4.75, ymax=-3, xax='AV')
    pl.savefig(f'{basepath}/figures/boxplot_abundance_vs_AV_with_fit.pdf', bbox_inches='tight')

    pl.clf()
    boxplot_abundance(basetable=basetable, sel=ok2221, ok=ok2221, av_start=17, av_max=80, av_spacing=5, ymin=-4.75, ymax=-3, xax='NH2')
    pl.savefig(f'{basepath}/figures/boxplot_abundance_vs_NH2_with_fit.pdf', bbox_inches='tight')

    pl.clf()
    boxplot_abundance(basetable=basetable, sel=ok2221, ok=ok2221, av_start=17, av_max=80, av_spacing=5, ymin=-4.75, ymax=-3, xax='logNH2')
    pl.savefig(f'{basepath}/figures/boxplot_abundance_vs_logNH2_with_fit.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()