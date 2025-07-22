"""
Created 2025-03-19, long after publication
"""
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

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath, compute_molecular_column, molscomps

from dust_extinction.averages import CT06_MWGC, G21_MWAvg, F11_MWGC
from dust_extinction.parameter_averages import G23

from astropy.wcs import WCS


def ev(avfilts, ext=G21_MWAvg()):
    av_wavelengths = [int(avf[1:-1])/100. * u.um for avf in avfilts]
    return (ext(av_wavelengths[0]) - ext(av_wavelengths[1]))


def calc_av(avfilts=['F182M', 'F410M'], basetable=None, ext=CT06_MWGC(), return_av=True):
    try:
        E_V = ev(avfilts, ext)
    except ValueError:
        E_V = ev(avfilts, G21_MWAvg())

    if return_av:
        av = (basetable[f'mag_ab_{avfilts[0].lower()}'] - basetable[f'mag_ab_{avfilts[1].lower()}']) / E_V
        return av
    else:
        color = (basetable[f'mag_ab_{avfilts[0].lower()}'] - basetable[f'mag_ab_{avfilts[1].lower()}'])
        return color


def get_smithdata():
    from pylatexenc.latex2text import LatexNodes2Text
    from unidecode import unidecode

    tab = Table.read('https://www.nature.com/articles/s41550-025-02511-z/tables/1', format='ascii.html')

    ln2 = LatexNodes2Text()
    for colname in tab.colnames:

        for row in tab:
            try:
                row[colname] = ln2.latex_to_text(row[colname])
            except:
                pass

        # Convert LaTeX to Unicode, then Unicode to ASCII
        unicode_name = ln2.latex_to_text(colname).replace("_", "")
        ascii_name = unidecode(unicode_name)
        tab.rename_column(colname, ascii_name)

    tab['F410M'].unit = u.mJy
    tab['RA'].unit = u.deg
    tab['Dec'].unit = u.deg
    tab['NH2'].unit = 10**22 * u.cm**-2
    tab['NH2O'].unit = 10**18 * u.cm**-2
    tab['NCO2'].unit = 10**18 * u.cm**-2
    tab['NCO'].unit = 10**18 * u.cm**-2
    tab['Av'] = tab['NH2']*10**22 * u.cm**-2 * 2 / (2.21e21 * u.cm**-2)   # Convert to Av using NH2 to Av conversion factor
    tab['Av'].unit = u.mag  # Av is in magnitudes

    #tab['NCO']
    NCO_ice = np.zeros(len(tab['NCO']))#NCO_ice))#np.array(tab['NCO'])
    NCO_poserr = np.zeros_like(NCO_ice)
    NCO_negerr = np.zeros_like(NCO_ice)
    Av_ice = np.array(tab['Av'])

    for i in range(len(NCO_ice)):
        #ele = NCO_ice[i]
        NCO_ice[i] = np.float64(np.array(tab['NCO'])[i].split('_')[0])
        NCO_negerr[i] = np.abs(np.float64(np.array(tab['NCO'])[i].split('_')[1].split("^")[0]))
        NCO_poserr[i] = np.abs(np.float64(np.array(tab['NCO'])[i].split('_')[1].split("^")[1]))

    NCO_ice = NCO_ice * 1e18  # Convert to cm^-2 for consistency with other values
    NCO_poserr = NCO_poserr * 1e18  # Convert to cm^-2 for consistency with other values
    NCO_negerr = NCO_negerr * 1e18  # Convert to cm^-2 for consistency with other values
    NH2 = tab['NH2'].quantity.to(u.cm**-2)

    return NH2, NCO_ice, NCO_poserr, NCO_negerr, Av_ice

def makeplot(basetable,
             avfilts=['F182M', 'F410M'],
             ax=None, sel=None, ok=None, alpha=0.5,
             icemol='CO',
             atom='C',
             abundance=10**(8.7-12), # roughly extrapolated from Smartt 2001A%26A...367...86S
             title='H2O:CO:CO2 (10:1:1)',
             dmag_tbl=None,
             plot_brandt=True,
             NHtoAV=2.21e21,
             # av_start = 20 based on Jang, An, Whittet...
             av_start=20,
             xax='AV',
             cloudccat=None,
             sgrb2cat=None,
             smithplot=False,
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
             ):


    assert dmag_tbl is not None
    assert basetable is not None

    if ax is None:
        ax = pl.gca()
    fig = ax.get_figure()

    if grid:
        ax.grid(True, linestyle='--', alpha=0.25, zorder=-50)

    av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext, return_av=(xax == 'AV'))

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

    assert np.nanmax(inferred_molecular_column) > 1e16

    artists_and_labels = {}

    if logy:
        yvals = np.log10(inferred_molecular_column[sel & ok])
        ylabel = f"log N({icemol} ice) [cm$^{{-2}}$] using {color_filter1}-{color_filter2} color"
    else:
        print(f"logy is not set, converting to 10^19 cm^-2 and using linear y-axis")
        yvals = inferred_molecular_column[sel & ok] / 1e19
        ylabel = f"N({icemol} ice) [10$^{{19}}$ cm$^{{-2}}$] using {color_filter1}-{color_filter2} color"
        if ylim is not None:
            ylim = (10.**ylim[0] / 1e19, 10.**ylim[1] / 1e19)

    # we do this many times redundantly because it seems not to work...
    ax.set_ylabel(ylabel)

    if ylim is None:
        ylim = (np.nanpercentile(yvals, 0.5), np.nanpercentile(yvals, 99.5))
        print(f"Computed ylimits automatically: {ylim}")
    else:
        try:
            assert np.isfinite(float(ylim[0]))
            assert np.isfinite(float(ylim[1]))
        except Exception as ex:
            print(ex)
            print(f"ylim was {ylim}")
            print(f"dtypes of ylime were {type(ylim[0])} and {type(ylim[1])}")
            raise ex

    assert ylim[1] > ylim[0]

    if hexbin:
        if scatter:
            raise ValueError("hexbin and scatter should not both be True")
        ax.hexbin(np.array(av[sel & ok]), yvals, mincnt=1, gridsize=100, extent=xlim + ylim, cmap='gray',
                  zorder=-20,
                  linewidths=0.1,
                  alpha=hexbin_alpha,
                  )

    if scatter:
        ax.scatter(np.array(av[sel & ok]),
                   yvals,
                   color='k',
                   marker='.', s=0.5, alpha=alpha)
        print(f"scatter plot of {icemol} ice column vs {avfilts[0]}-{avfilts[1]} color: {np.nanpercentile(yvals, 0.5):0.1e} to {np.nanpercentile(yvals, 99.5):0.1e}")

    # don't overlay contour on hexbin.  Sometimes we need contour.
    if contour and not hexbin:
        cx,cy,H,_,_,levels, cnt = mpl_plot_templates.adaptive_param_plot(np.array(av[sel & ok]),
                                            yvals,
                                            bins=np.array([np.linspace(xlim[0], xlim[1], nbins),
                                                           np.linspace(ylim[0], ylim[1], nbins)]),
                                            threshold=threshold,
                                            #linewidths=[0.5]*5,
                                            cmap=None,
                                            colors=[(0,0.5,0,0.5)]*50,
                                            marker='None',
                                            levels=nlevels,
                                            fill=False,
                                            axis=ax,
                                                    )
        artists, labels = cnt.legend_elements()
        # if cloudccat is None:
        #     pl.legend(artists, ['Brick'], **legend_kwargs)

        green_line = mlines.Line2D([], [], color=(0,0.5,0,0.5), marker='none',
                                    linestyle='-', label='Brick')
        artists_and_labels['Brick'] = green_line

    if sgrb2cat is not None:
        av_sgrb2 = calc_av(avfilts=avfilts, basetable=sgrb2cat, ext=ext, return_av=(xax == 'AV'))
        measured_color_sgrb2 = sgrb2cat[f'mag_ab_{color_filter1.lower()}'] - sgrb2cat[f'mag_ab_{color_filter2.lower()}']
        unextincted_color_sgrb2 = measured_color_sgrb2 + E_V_color * av_sgrb2
        inferred_molecular_column_sgrb2 = compute_molecular_column(unextincted_1m2=unextincted_color_sgrb2,
                                                                  dmag_tbl=dmag_tbl,
                                                                  icemol=icemol,
                                                                  filter1=color_filter1,
                                                                  filter2=color_filter2,
                                                                  verbose=debug)

        sgrb2_sel = inferred_molecular_column_sgrb2 < 10**19.8

        if logy:
            yvals_sgrb2 = np.log10(inferred_molecular_column_sgrb2[sgrb2_sel])
        else:
            yvals_sgrb2 = inferred_molecular_column_sgrb2[sgrb2_sel]

        if scatter:
            ax.scatter(np.array(av_sgrb2[sgrb2_sel]),
                       yvals_sgrb2,
                       marker='.', s=0.5, alpha=alpha, color='c')
        if contour:
            cx,cy,H,_,_,levels,cnt = mpl_plot_templates.adaptive_param_plot(np.array(av_sgrb2[sgrb2_sel]),
                                        yvals_sgrb2,
                                        bins=np.array([np.linspace(xlim[0], xlim[1], nbins),
                                                       np.linspace(ylim[0], ylim[1], nbins)]),
                                        threshold=threshold,
                                        cmap=None,
                                        colors=[(0,1,1,1)]*50,
                                        marker='None',
                                        levels=nlevels,
                                        fill=False,
                                        axis=ax,
                                        )
            cyan_line = mlines.Line2D([], [], color=(0,1,1,1), marker='none',
                                      linestyle='-', label='Sgr B2')
            artists_and_labels['Sgr B2'] = cyan_line
            #pl.legend(artists, ['Sgr B2'], **legend_kwargs)


    if cloudccat is not None:
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

        av_cloudc = calc_av(avfilts=avfilts, basetable=cloudccat, ext=ext, return_av=(xax == 'AV'))
        measured_color_cloudc = cloudccat[f'mag_ab_{color_filter1.lower()}'] - cloudccat[f'mag_ab_{color_filter2.lower()}']
        unextincted_color_cloudc = measured_color_cloudc + E_V_color * av_cloudc
        inferred_molecular_column_cloudc = compute_molecular_column(unextincted_1m2=unextincted_color_cloudc,
                                                                   dmag_tbl=dmag_tbl,
                                                                   icemol=icemol,
                                                                   filter1=color_filter1,
                                                                   filter2=color_filter2,
                                                                   verbose=debug)

        cloudc_sel &= inferred_molecular_column_cloudc < 10**19.8
        lactea_sel &= inferred_molecular_column_cloudc < 10**19.8

        if logy:
            yvals_cloudc = np.log10(inferred_molecular_column_cloudc[cloudc_sel])
            yvals_lactea = np.log10(inferred_molecular_column_cloudc[lactea_sel])
        else:
            yvals_cloudc = inferred_molecular_column_cloudc[cloudc_sel]
            yvals_lactea = inferred_molecular_column_cloudc[lactea_sel]

        if scatter:
            ax.scatter(np.array(av_cloudc[cloudc_sel]),
                       yvals_cloudc,
                       marker='.', s=0.5, alpha=alpha, color='b')
            ax.scatter(np.array(av_cloudc[lactea_sel]),
                       yvals_lactea,
                       marker='.', s=0.5, alpha=alpha, color='orange')
        if contour:
            cx,cy,H,_,_,levels,cnt = mpl_plot_templates.adaptive_param_plot(np.array(av_cloudc[cloudc_sel]),
                                        yvals_cloudc,
                                        bins=np.array([np.linspace(xlim[0], xlim[1], nbins),
                                                       np.linspace(ylim[0], ylim[1], nbins)]),
                                        threshold=threshold,
                                        cmap=None,
                                        colors=[(0,0,1,0.7)]*50, # blue
                                        marker='None',
                                        levels=nlevels,
                                        fill=False,
                                        axis=ax,
                                        )
            artists, labels = cnt.legend_elements()

            blue_line = mlines.Line2D([], [], color=(0,0,1,0.7), marker='none',
                                      linestyle='-', label='Cloud C & D')
            artists_and_labels['Cloud C & D'] = blue_line
            #print(cx, cy)
            #pl.legend(artists, ['Cloud C'], **legend_kwargs)
            cx,cy,H,_,_,levels,cnt = mpl_plot_templates.adaptive_param_plot(np.array(av_cloudc[lactea_sel]),
                                            yvals_lactea,
                                            bins=np.array([np.linspace(xlim[0], xlim[1], nbins),
                                                           np.linspace(ylim[0], ylim[1], nbins)]),
                                            threshold=threshold,
                                            cmap=None,
                                            colors=[(1,0.5,0,1.0)]*50,
                                            marker='None',
                                            levels=nlevels,
                                            fill=False,
                                            axis=ax,
                                            )
            artists, labels = cnt.legend_elements()

            orange_line = mlines.Line2D([], [], color=(1,0.5,0,1.0), marker='none',
                                      linestyle='-', label='3 kpc arm filament')
            artists_and_labels['3 kpc arm filament'] = orange_line
            #pl.legend(artists, ['3 kpc arm filament'], **legend_kwargs)
            #print(cx, cy)

        suffix += '_cloudc'
    else:
        suffix += '_with1182'


    if smithplot:
        NH2_smith, NCO_smith, NCO_poserr_smith, NCO_negerr_smith, AV_ice_smith = get_smithdata()
        toplot_smith = ((NCO_negerr_smith / NCO_smith) < 0.2) & ((NCO_poserr_smith / NCO_smith) < 0.2)
        if logy:
            ax.errorbar((AV_ice_smith + av_start)[toplot_smith],
                        np.log10(NCO_smith[toplot_smith]),
                        yerr=[np.abs((NCO_negerr_smith/NCO_smith)[toplot_smith]),
                              np.abs((NCO_poserr_smith/NCO_smith)[toplot_smith])],
                        color='k',
                        marker='o',
                        linewidth=0.5,
                        alpha=0.7,
                        zorder=3,
                        markersize=5, linestyle='none', label='Smith+25')
        else:
            ax.errorbar((AV_ice_smith + av_start)[toplot_smith],
                        NCO_smith[toplot_smith],
                        yerr=[NCO_negerr_smith[toplot_smith], NCO_poserr_smith[toplot_smith]],
                        color='k',
                        linewidth=0.5,
                        alpha=0.7,
                        zorder=3,
                        marker='o', markersize=5, linestyle='none', label='Smith+25')

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    #pl.plot([10, 35], [1e17, 1e20], 'k--', label='log N = 0.12 A$_V$ + 15.8');
    # by-eye fit
    # x1,y1 = 33,8e16
    # x2,y2 = 80,3e19
    # m = (np.log10(y2) - np.log10(y1)) / (x2 - x1)
    # b = np.log10(y1 / 10**(m * x1))
    # pl.plot([x1, x2], np.array([x1*m+b, x2*m+b]), 'k--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f}')
    #pl.plot([7, 23], 10**(np.array([7,23]) * m + b))
    #pl.legend(loc='lower right')
    ax.set_xlabel(f"A$_V$ from {avfilts[0]}-{avfilts[1]} (mag)")
    ax.set_ylabel(ylabel)

    #print(f"ylabel is {ax.get_ylabel()}")

    ax2 = ax.twiny()
    ax2.set_xlabel('N(H$_2$) [10$^{22}$ cm$^{-2}$]')
    if xax == 'AV':
        ax2.set_xlim((np.array(xlim) - av_start) * NHtoAV / 2 * 1e-22)
    elif xax == 'color':
        # A_V = [B-V] / E(B-V)
        ax2.set_xlim(np.array(xlim) / ev(avfilts, ext) * NHtoAV / 2 * 1e-22)
    else:
        raise ValueError(f"Unknown xax label choice: {xax}")
    ax2.set_ylabel(ax.get_ylabel())

    ax.set_ylabel(ylabel)

    #print(f"ylabel is {ax.get_ylabel()}.  ax2 ylabel is {ax2.get_ylabel()}")
    #print(f"ylabel is {ax.get_ylabel()}.  ax2 ylabel is {ax2.get_ylabel()}")

    if contour:
        suffix += '_contour'
    if logy:
        suffix += '_log'
    else:
        suffix += '_linear'

    title_to_save = title.replace(' ','_').replace("$", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace(",", "").replace(";", "")
    # this just gets overwritten, so skip it
    # outfn = f"{basepath}/paper_co/fig9s/N{icemol}_{title_to_save}_vs_{xax}_{color_filter1}-{color_filter2}_{avfilts[0]}-{avfilts[1]}{suffix}"
    # fig.savefig(f"{outfn}.pdf", dpi=150, bbox_inches='tight')
    # fig.savefig(f"{outfn}.png", dpi=250, bbox_inches='tight')
    # print(f"Completed {outfn}")

    #pl.plot([7, 23], np.log10([0.5e17, 7e17]), 'g', label='log N = 0.07 A$_V$ + 16.2 [BGW 2015]', linewidth=2)


    logN = int(np.log10(NHtoAV))
    xax_toplot = np.linspace(0.1, 100, 1000) + av_start
    if xax != 'AV':
        xax_toplot = xax_toplot * ev(avfilts, ext)

    if use_abundance:
        raise NotImplementedError("use_abundance is deprecated because I haven't tested it since making a lot of changes")
        NMolofAV = NHtoAV * np.linspace(0.1, 100, 1000) * abundance
        if logy:
            co_y = np.log10(NMolofAV)
        else:
            co_y = NMolofAV / 1e19
        co_av_line, = ax.plot(xax_toplot, co_y,
                label=f'100% of {atom} in {icemol} ice \nif N(H)={NHtoAV/10**logN}$\\times10^{{{logN}}}$ A$_V$\nand X({atom})=$10^{{{np.log10(abundance):0.1f}}}$\nand $E({avfilts[0]}-{avfilts[1]})={ev(avfilts, ext):0.2f}$\nand $A_{{V,fg}}={av_start}$', color='r', linestyle=':')
        artists_and_labels[co_av_line.get_label()] = co_av_line
        if show_25_percent:
            if logy:
                co_y_25 = np.log10(NMolofAV * 0.25)
            else:
                co_y_25 = NMolofAV * 0.25 / 1e19
            co_av_line_25, = ax.plot(xax_toplot, co_y_25,
                    label=f'25% of {atom} in {icemol} ice', color='r', linestyle='--', zorder=-10, alpha=0.5)

            artists_and_labels[co_av_line_25.get_label()] = co_av_line_25
    else:
        # if we're not using abundance, we're using fixed levels of {icemol}/H2 (so it's CO/H2, not C/H2 * 0.25)
        levels = [5e-4, 2.5e-4, 1e-4]
        NH2_of_AV = NHtoAV / 2. * np.linspace(0.1, 100, 1000)
        for level, linestyle in zip(levels, ('--', ':', '-.')):
            if logy:
                co_y = np.log10(NH2_of_AV * level)
            else:
                co_y = NH2_of_AV * level / 1e19
            level_leader = f'{level:0.1e}'[:3]
            co_av_line, = ax.plot(xax_toplot, co_y,
                label=f'{icemol}/H$_2 = {level_leader} \\times 10^{{{int(np.floor(np.log10(level)))}}}$',
                color='r', linestyle=linestyle, zorder=-10, alpha=0.5)

            artists_and_labels[co_av_line.get_label()] = co_av_line

    if xax == 'AV':
        ax.set_xlabel(f"A$_V$ from {avfilts[0]}-{avfilts[1]} (mag)")
    else:
        ax.set_xlabel(f"{avfilts[0]}-{avfilts[1]} (mag)")
    #pl.ylabel("N(CO) ice\nfrom Palumbo 2006 constants,\n4000K Phoenix atmosphere")
    ax.set_ylabel(ylabel)
    #print(f"ylabel is {ax.get_ylabel()}.  ax2 ylabel is {ax2.get_ylabel()}")
    # print(artists_and_labels)
    pl.legend(handles=list(artists_and_labels.values()),
              labels=list(artists_and_labels.keys()),
              **legend_kwargs)
    ax.set_title(title)

    # mask out low signal-to-noise stuff
    # if logy:
    #     ax.fill_between([0, 1e25], [16, 16], [18, 18], color='w', alpha=0.2, zorder=5)
    # else:
    #     ax.fill_between([0, 1e25], [1e16, 1e16], [1e18, 1e18], color='w', alpha=0.2, zorder=5)

    outfn2 = f"{basepath}/paper_co/fig9s/N{icemol}_{title_to_save}_vs_{xax}_{color_filter1}-{color_filter2}_{avfilts[0]}-{avfilts[1]}{suffix}"
    fig.savefig(f"{outfn2}.pdf", dpi=150, bbox_inches='tight')
    fig.savefig(f"{outfn2}.png", dpi=250, bbox_inches='tight')
    print(f"Completed {outfn2}")

    #print(f"ylabel is {ax.get_ylabel()}.  ax2 ylabel is {ax2.get_ylabel()}")

    if plot_brandt:
        plot_brandt_model(ax, molecule=icemol, nh_to_av=NHtoAV, av_start=av_start)
        outfn3 = f"{basepath}/paper_co/fig9s/N{icemol}_{title_to_save}_vs_{xax}_{color_filter1}-{color_filter2}_{avfilts[0]}-{avfilts[1]}_Brandt{suffix}"
        fig.savefig(f"{outfn3}.png", dpi=250, bbox_inches='tight')
        print(f"Completed {outfn3}")

    return av, inferred_molecular_column, ax


def plot_brandt_model(ax, nh_to_av=2.21e21, molecule='CO', av_start=0):
    column = fits.getdata(f'{basepath}/brandt_ice/brick.dust_column_density_cf.fits')
    if molecule == 'CO':
        molcol = np.load(f'{basepath}/brandt_ice/COIceMap_0.npy')
    elif molecule == 'H2O':
        molcol = np.load(f'{basepath}/brandt_ice/H2OIceMap_0.npy')
    else:
        raise ValueError(f"Unknown molecule: {molecule}")

    lims = ax.axis()
    if lims == (0.0, 1.0, 0.0, 1.0):
        lims = (0, 100, 15, 21)
    ok = np.isfinite(column) & np.isfinite(molcol) & (column>0) & (molcol>0)

    # multiply by 2 to go from H2->H
    av = column * 2 / nh_to_av + av_start

    nbins = 100
    hh, x_edges, y_edges = np.histogram2d(av[ok], np.log10(molcol[ok]),
                               bins=[np.linspace(lims[0], lims[1], nbins), np.linspace(lims[2], lims[3], nbins)])
    hh = hh.T
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    ax.contourf(x_centers, y_centers, hh, cmap='gray_r', zorder=-15)


def compare_freezeout_abundance_models():
    # oxygen abundance at r=0 is 12-9.4 = 2.6
    # solar oxygen abundance is 8.9
    # carbon abundance at r=0 is 8.7-12 = -3.3
    # solar carbon is 8.5 - 12 = -3.5
    # solar neighborhood is 8.2 - 12 = -3.8

    NHtoAV=2.21e21
    av_start=0

    color = np.linspace(0.0, 3, 1000)
    co_to_c = 0.5

    for abundance, linestyle in zip((10**(8.7-12), 10**(8.2-12)), ('-', '--')):
        for avfilts in (['F182M', 'F212N'],): # ['F182M', 'F410M'], ['F115W', 'F200W']):
            for ext, extname, plotcolor in zip((CT06_MWGC(), G23(Rv=2.5), G23(Rv=5.5), ), #F11_MWGC()),
                                           ('CT06', 'G23 $R_V=2.5$', 'G23 $R_V=5.5$', ), #'F11'),
                                           ('r', 'g', 'b', )): #'k')):
                av_wavelengths = [int(avf[1:-1])/100. * u.um for avf in avfilts]
                E_V = (ext(av_wavelengths[0]) - ext(av_wavelengths[1]))

                NMolofAV = NHtoAV * color / E_V * abundance * co_to_c
                logN = int(np.log10(NHtoAV))

                abundance_str = f"{abundance * co_to_c/10**(int(np.log10(abundance * co_to_c))-1):.1f}\\times10^{{{int(np.log10(abundance * co_to_c))-1}}}"

                pl.plot(color + av_start * E_V, np.log10(NMolofAV),
                        label=f'$X_{{CO}} = {co_to_c:0.1f} X_C = {abundance_str}$; {extname}',
                        color=plotcolor,
                        linestyle=linestyle)
    # for av_start, linestyle in zip((0, 15, 30), ('-', '--', ':')):
    #     av_wavelengths = [int(avf[1:-1])/100. * u.um for avf in avfilts]
    #     E_V = (ext(av_wavelengths[0]) - ext(av_wavelengths[1]))

    #     NMolofAV = NHtoAV * color / E_V * abundance
    #     logN = int(np.log10(NHtoAV))
    #     pl.plot(color + av_start * E_V, np.log10(NMolofAV),
    #             label=f'X=10$^{{{np.log10(abundance):.1f}}}$',
    #             linestyle=linestyle)
    pl.legend(loc='best')
    pl.xlabel('[F182M]-[F212N] (mag)')
    pl.ylabel('log N(CO) [cm$^{-2}$]')
    pl.savefig(f"{basepath}/paper_co/fig9s/freezeout_abundance_models.pdf", dpi=150, bbox_inches='tight')


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

    mol = 'COplusH2O'
    dmag_tbl = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
    dmag_tbl.add_index('composition')


    dmag_co = Table.read(f'{basepath}/tables/CO_ice_absorption_tables.ecsv')
    dmag_co.add_index('composition')
    dmag_co.add_index('temperature')
    dmag_co.add_index('mol_id')

    c_abundance = 10**(8.7-12)
    c_abundance = 5e-4 # equals 10^-3.3
    o_abundance = 10**-3.31
    o_abundance_gc = 10**(9.3-12)

    def makeplot_simpler(basetable=basetable, sel=ok2221, ok=ok2221,
                         dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:1)'],  **kwargs):
        return makeplot(basetable=basetable,
                        sel=sel, ok=ok,
                        dmag_tbl=dmag_tbl,
                        **kwargs)

    import sys
    sys.path.append('/home/savannahgramze/orange_link/adamginsburg/jwst/cloudc/lactea-filament/lactea-filament')
    import jwst_plots

    color_filter2 = 'F466N'
    for color_filter1 in ('F405N', 'F410M'):
        pl.close('all')

        # Moved to top for debug
        for contour in (False, True):
            for scatter, scatterlabel in zip((False, True, False), ('', '_scatter', '_hexbin')):
                for xax in ('AV', 'color'):
                    av, inferred_molecular_column, ax = makeplot_simpler(
                            avfilts=['F182M', 'F212N'],
                            icemol='CO',
                            abundance=c_abundance,
                            title='H2O:CO:CO2 (10:1:1)',
                            ext=CT06_MWGC(),
                            xax=xax,
                            plot_brandt=False,
                            scatter=scatter,
                            hexbin=scatterlabel == '_hexbin',
                            xlim=(-0.1, 2.5) if xax == 'color' else (0, 80),
                            ylim=(17, 19.5),
                            logy=True,
                            av_start=17,
                            nbins=46,
                            nlevels=2,
                            contour=contour,
                            smithplot=True,
                            suffix=f'_BrickCloudCandArm{scatterlabel}',
                            legend_kwargs={'loc': 'lower right',},# 'bbox_to_anchor': (1.2, 0,)},
                            cloudccat=jwst_plots.make_cat_use().catalog,
                            color_filter1=color_filter1,
                            color_filter2=color_filter2,
                            hexbin_alpha=0.25,
                            );
                    ax.cla()
                    pl.clf()
                    pl.close('all')


                    sgrb2cat = Table.read('/orange/adamginsburg/jwst/sgrb2/NB/crowdsource_nsky0_merged_photometry_tables_merged_11matches.fits')
                    sgrb2cat = sgrb2cat[(sgrb2cat['emag_ab_f212n'] < 0.05) &
                                        (sgrb2cat['emag_ab_f182m'] < 0.05) &
                                        (sgrb2cat['emag_ab_f410m'] < 0.05) &
                                        (sgrb2cat['emag_ab_f466n'] < 0.05)]
                    makeplot_simpler(
                            avfilts=['F182M', 'F212N'],
                            icemol='CO', abundance=c_abundance,
                            title='H2O:CO:CO2 (10:1:1)',
                            ext=CT06_MWGC(),
                            xax=xax,
                            plot_brandt=False,
                            scatter=scatter,
                            hexbin=scatterlabel == '_hexbin',
                            xlim=(-0.1, 2.5) if xax == 'color' else (0, 80),
                            ylim=(17, 19.5),
                            av_start=17,
                            logy=True,
                            nbins=46,
                            nlevels=2,
                            contour=contour,
                            legend_kwargs={'loc': 'lower right'}, #, 'bbox_to_anchor': (1.2, 0,)},
                            sgrb2cat=sgrb2cat,
                            color_filter1=color_filter1,
                            color_filter2=color_filter2,
                            suffix=f'SgrB2{scatterlabel}',
                            hexbin_alpha=0.5,
                            );
                    pl.clf()
                    pl.close('all')

        pl.close('all')

        # DEFAULT
        makeplot_simpler(
                ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)

        # CT06 doesn't apply to F115W
        makeplot_simpler(
                avfilts=['F115W', 'F200W'], ax=pl.figure().gca(), ext=G23(Rv=5.5), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
        makeplot_simpler(
                avfilts=['F115W', 'F212N'], ax=pl.figure().gca(), ext=G23(Rv=5.5), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)


        dmag_h2o = Table.read(f'{basepath}/tables/H2O_ice_absorption_tables.ecsv')
        #dmag_co2 = Table.read(f'{basepath}/tables/CO2_ice_absorption_tables.ecsv')
        #dmag_mine = Table.read(f'{basepath}/tables/H2O+CO_ice_absorption_tables.ecsv')

        dmag_h2o.add_index('composition')
        dmag_h2o.add_index('temperature')
        dmag_h2o.add_index('mol_id')

        makeplot_simpler(
                avfilts=['F182M', 'F212N'], ax=pl.figure().gca(),
                icemol='H2O', abundance=o_abundance_gc, #abundance=4.89e-4,
                atom='O',
                title='H2O',
                ylim=(16.5, 20),
                dmag_tbl=dmag_h2o.loc['H2O (1)'].loc['temperature', '25K'].loc['mol_id', 240],
                legend_kwargs={'loc': 'lower right'},
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                )


        makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                icemol='CO', abundance=c_abundance,
                title='H2O:CO (10:1)',
                dmag_tbl=dmag_tbl.loc['H2O:CO (10:1)'],
                ext=G23(Rv=5.5),
                plot_brandt=False,
                scatter=False,
                cloudccat=jwst_plots.make_cat_use().catalog,
                xlim=(-5, 60),
                ylim=(16.5, 20),
                av_start=10,
                legend_kwargs={'loc': 'lower right'},
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                )

        makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                icemol='CO', abundance=c_abundance,
                title='H2O:CO (10:1)',
                dmag_tbl=dmag_tbl.loc['H2O:CO (10:1)'],
                ext=G23(Rv=5.5),
                xax='color',
                plot_brandt=False,
                scatter=False,
                xlim=(-0.1, 3),
                ylim=(16.5, 20),
                av_start=10,
                legend_kwargs={'loc': 'lower right'},
                cloudccat=jwst_plots.make_cat_use().catalog,
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                )



        av, inferred_molecular_column, ax = makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                icemol='CO', abundance=c_abundance,
                title='CO',
                ylim=(16.5, np.log10(3e20)),
                legend_kwargs={'loc': 'lower right'},
                dmag_tbl=dmag_co.loc['mol_id', 64].loc['composition', 'CO'],
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                xax='color',
                )
        pl.close('all')

        # av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
        #         icemol='CO', abundance=c_abundance,
        #         title='CO:OCN (1:1)',
        #         ylim=(16.5, np.log10(2e19)),
        #         legend_kwargs={'loc': 'lower right'},
        #         dmag_tbl=dmag_tbl.loc['CO:OCN (1:1)'],
        #         color_filter1=color_filter1,
        #         color_filter2=color_filter2,
        #         )
        # pl.close('all')

        # av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
        #         icemol='H2O', abundance=o_abundance,
        #         atom='O',
        #         title='H2O:CO:OCN (1:1:1)',
        #         ylim=(16.5, np.log10(2e19)),
        #         legend_kwargs={'loc': 'lower right'},
        #         dmag_tbl=dmag_tbl.loc['H2O:CO:OCN (1:1:1)'],
        #         color_filter1=color_filter1,
        #         color_filter2=color_filter2,
        #         )


        av, inferred_molecular_column, ax = makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                icemol='CO', abundance=c_abundance,
                title='H2O:CO (10:1)',
                ylim=(16.5, np.log10(2e19)),
                legend_kwargs={'loc': 'lower right'},
                dmag_tbl=dmag_tbl.loc['H2O:CO (10:1)'],
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                )
        pl.close('all')
        av, inferred_molecular_column, ax = makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                icemol='H2O', abundance=o_abundance,
                atom='O',
                title='H2O:CO (10:1)',
                ylim=(16.5, np.log10(2e20)),
                legend_kwargs={'loc': 'lower right'},
                dmag_tbl=dmag_tbl.loc['H2O:CO (10:1)'],
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                )
        pl.close('all')

        av, inferred_molecular_column, ax = makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                icemol='CO', abundance=c_abundance,
                title='H2O:CO:CO2 (10:1:2)',
                ylim=(17.0, np.log10(2e19)),
                legend_kwargs={'loc': 'lower right'},
                dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:2)'],
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                )
        pl.close('all')

        av, inferred_molecular_column, ax = makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                icemol='H2O', abundance=o_abundance,
                atom='O',
                title='H2O:CO:CO2 (10:1:2)',
                ylim=(18.5, np.log10(2e20)),
                legend_kwargs={'loc': 'lower right'},
                dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:2)'],
                color_filter1=color_filter1,
                color_filter2=color_filter2,
                )
        pl.close('all')

        makeplot_simpler(
                avfilts=['F182M', 'F410M'], ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
        pl.close('all')
        makeplot_simpler(
                avfilts=['F187N', 'F212N'], ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
        pl.close('all')
        makeplot_simpler(
                avfilts=['F187N', 'F405N'], ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
        pl.close('all')
        makeplot_simpler(
                avfilts=['F182M', 'F200W'], ax=pl.figure().gca(), ylim=(17.0, 19.5), legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
        pl.close('all')

        av, inferred_molecular_column, ax = makeplot_simpler(
                avfilts=['F182M', 'F212N'],
                ax=pl.figure().gca(), ylim=(17.0, 19.5),
                xlim=(-1, 80),
                plot_brandt=False, legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
        pl.close('all')

        # for uncertainty estimation
        for hexbin, hexbinlabel in zip((False, True), ('', '_hexbin')):
            for molcomp in ('H2O:CO:CO2 (5:1:1)', 'H2O:CO:CO2 (10:1:1)', 'H2O:CO:CO2 (20:1:1)'):
                av, inferred_molecular_column, ax = makeplot_simpler(
                                                    avfilts=['F182M', 'F212N'],
                                                    ax=pl.figure().gca(), ylim=(17.0, 19.5),
                                                    xlim=(-1, 80),
                                                    dmag_tbl=dmag_tbl.loc[molcomp],
                                                    title=molcomp,
                                                    hexbin=hexbin,
                                                    av_start=17,
                                                    scatter=not hexbin,
                                                    suffix=f'{hexbinlabel}',
                                                    grid=True,
                                                    contour=False,
                                                    plot_brandt=False, legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)

                av, inferred_molecular_column, ax = makeplot_simpler(
                                                    avfilts=['F182M', 'F212N'],
                                                    ax=pl.figure().gca(), ylim=(17.0, 20.0),
                                                    xlim=(-1, 80),
                                                    dmag_tbl=dmag_tbl.loc[molcomp],
                                                    title=molcomp,
                                                    av_start=17,
                                                    hexbin=hexbin,
                                                    scatter=not hexbin,
                                                    suffix=f'{hexbinlabel}',
                                                    grid=True,
                                                    contour=False,
                                                    plot_brandt=False, legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2,
                                                    icemol='H2O', abundance=o_abundance,
                                                    atom='O',)
                pl.close('all')

            for av_foreground in (15, 17, 20, 25):
                av, inferred_molecular_column, ax = makeplot_simpler(
                                                    avfilts=['F182M', 'F212N'],
                                                    ax=pl.figure().gca(), ylim=(17.0, 19.5),
                                                    xlim=(-1, 80),
                                                    title=f'H2O:CO:CO2 (10:1:1); $A_{{V,fg}}={av_foreground}$',
                                                    av_start=av_foreground,
                                                    hexbin=hexbin,
                                                    scatter=not hexbin,
                                                    suffix=f'{hexbinlabel}',
                                                    grid=True,
                                                    contour=False,
                                                    plot_brandt=False, legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
                pl.close('all')

            for ext, extname in zip((CT06_MWGC(), G23(Rv=5.5), G23(Rv=3.1)), ('CT06', 'G23 $R_V=5.5$', 'G23 $R_V=3.1$')):
                av, inferred_molecular_column, ax = makeplot_simpler(
                                                    avfilts=['F182M', 'F212N'],
                                                    ax=pl.figure().gca(), ylim=(17.0, 19.5),
                                                    xlim=(-1, 80),
                                                    title=f'H2O:CO:CO2 (10:1:1) {extname}',
                                                    ext=ext,
                                                    hexbin=hexbin,
                                                    scatter=not hexbin,
                                                    av_start=17,
                                                    grid=True,
                                                    contour=False,
                                                    suffix=f'{hexbinlabel}',
                                                    plot_brandt=False, legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)
                pl.close('all')

            # Special case so the left edge matches
            ext = G23(Rv=5.5)
            extname = 'G23 $R_V=5.5$'
            av, inferred_molecular_column, ax = makeplot_simpler(
                                                avfilts=['F182M', 'F212N'],
                                                ax=pl.figure().gca(), ylim=(17.0, 19.5),
                                                xlim=(-1, 80),
                                                title=f'H2O:CO:CO2 (10:1:1) {extname}, $A_{{V,fg}}=12$',
                                                ext=ext,
                                                hexbin=hexbin,
                                                scatter=not hexbin,
                                                av_start=12,
                                                grid=True,
                                                contour=False,
                                                suffix=f'{hexbinlabel}',
                                                plot_brandt=False, legend_kwargs={'loc': 'lower right'}, color_filter1=color_filter1, color_filter2=color_filter2)

if __name__ == "__main__":
    main()