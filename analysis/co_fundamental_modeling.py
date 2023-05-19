# this takes a while to load the first time especially
from pyspeckitmodels.co import exomol_co_vibration
from pyspeckitmodels.sio import exomol_sio_vibration

from pyspeckitmodels.sio.exomol_sio_vibration import exomol_xsec as SiO_exomol_xsec
from pyspeckitmodels.co.exomol_co_vibration import exomol_xsec as CO_exomol_xsec, tau_of_N as CO_tau_of_N
from astropy import constants, units as u
from astroquery.svo_fps import SvoFps
import numpy as np
import pyspeckitmodels
from tqdm import tqdm

import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
pl.rcParams['figure.figsize'] = (10,8)
pl.rcParams['font.size'] = 16



def get_wltable(filtername, telescope='JWST', instrument='NIRCam'):
    return SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')


def fractional_absorption(temperature, column, linewidth, wavelength_table):

    xarr = wavelength_table['Wavelength'].quantity.to(u.um)
    xarr = np.linspace(xarr.min(), xarr.max(), int(1e4))
    trans = np.interp(xarr, wavelength_table['Wavelength'], wavelength_table['Transmission'])
    transmission_sum = (trans).sum()

    tau = pyspeckitmodels.co.exomol_co_vibration.tau_of_N(xarr.to(u.cm).value,
                                                column.to(u.cm**-2).value,
                                                temperature.to(u.K).value,
                                                width=linewidth.to(u.km/u.s).value,
                                                          progressbar=lambda x: x,
                                                         )
    absorbed_fraction = ((1-np.exp(-tau)) * trans).sum() / transmission_sum
    return absorbed_fraction


def make_temperature_linewidth_plot(filtername, column=1e18*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick'):
    wltable = get_wltable(filtername)
    temperatures = np.linspace(5, 150, 50)*u.K
    linewidths = np.linspace(1, 100, 24)*u.km/u.s
    grid = [[fractional_absorption(T, column, sig, wltable) for T in temperatures]
            for sig in tqdm(linewidths)]

    ax = pl.gca()
    im = ax.imshow(grid,
                   #extent=[np.log10(column.min().value), np.log10(column.max().value),
                   #        linewidths.min().value, linewidths.max().value],
                   origin='lower',
                   interpolation='bilinear',
                   vmin=0,
                   vmax=1,
                   cmap='gray')
    con = ax.contour(grid, levels=levels, colors=colors)
    pl.xlabel("Temperature [K]")
    pl.ylabel("Linewidth $\sigma$")
    pl.title(f"CO absorption at N(CO) = 10$^{{{np.log10(column.to(u.cm**-2).value):0.1f}}}$ cm$^{{-2}}$")


    sigma_labels = [1,10,20,50,100]
    ax.yaxis.set_ticks(np.interp(sigma_labels, linewidths.value, np.arange(len(linewidths)), right=np.nan),
                       labels=sigma_labels)
    tem_labels = np.arange(temperatures.min().value, temperatures.max().value, 20, dtype='int')
    ax.xaxis.set_ticks(np.interp(tem_labels, (temperatures.value), np.arange(len(temperatures)), right=np.nan),
                       labels=tem_labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=-2.3)
    cb = pl.colorbar(cax=cax, mappable=im)
    cb.set_label(f"Fractional Absorption in {filtername}")
    cb.ax.hlines(con.levels, *cb.ax.get_xlim(), colors=con.colors)
    ax.set_aspect(2)


    pl.savefig(f'{basepath}/paper_figures/CO_{filtername}_absorption_{column.value:0.1g}co_coldens.pdf', bbox_inches='tight')



def make_linewidth_column_plot(filtername, temperature=60*u.K, basepath='/orange/adamginsburg/jwst/brick'):
    wltable = get_wltable(filtername)
    linewidths = np.linspace(1, 100, 24)*u.km/u.s
    column = np.logspace(15,19,24)*u.cm**-2
    grid_sigcol = [[fractional_absorption(temperature, col, sig, wltable) for col in column]
                    for sig in tqdm(linewidths)]


    ax = pl.gca()
    im = ax.imshow(grid_sigcol,
                   #extent=[np.log10(column.min().value), np.log10(column.max().value),
                   #        linewidths.min().value, linewidths.max().value],
                   origin='lower',
                   interpolation='bilinear',
                   vmin=0,
                   vmax=1,
                   cmap='gray')
    con = ax.contour(grid_sigcol, levels=levels, colors=colors)
    pl.xlabel("CO Column Density [log cm$^{-2}$]")
    pl.ylabel("Linewidth $\sigma$")
    pl.title(f"CO absorption T={int(temperature.value):d} K")


    # a hideous way to format the axis
    # @pl.FuncFormatter
    # def log_scaler(x, pos):
    #     'The two args are the value and tick position'
    #     try:
    #         col = np.interp(x, np.arange(len(column)), column.value)
    #         return f'{np.log10(col):0.1f}'
    #     except Exception:
    #         return None
    # ax.xaxis.set_major_formatter(log_scaler)
    # @pl.FuncFormatter
    # def sig_scaler(x, pos):
    #     'The two args are the value and tick position'
    #     sig = np.interp(x, np.arange(len(linewidths)), linewidths.value)
    #     return f'{sig:0.1f}'
    # ax.yaxis.set_major_formatter(sig_scaler)
    sigma_labels = [1,10,20,50,100]
    ax.yaxis.set_ticks(np.interp(sigma_labels, linewidths.value, np.arange(len(linewidths)), right=np.nan),
                       labels=sigma_labels)
    col_labels = [15,16,17,18,19,]
    ax.xaxis.set_ticks(np.interp(col_labels, np.log10(column.value), np.arange(len(column)), right=np.nan),
                       labels=col_labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = pl.colorbar(cax=cax, mappable=im)
    cb.set_label(f"Fractional Absorption in {filtername}")
    cb.ax.hlines(con.levels, *cb.ax.get_xlim(), colors=con.colors)


    pl.savefig(f'{basepath}/paper_figures/CO_{filtername}_absorption_{int(temperature.value):d}K.pdf', bbox_inches='tight')


if __name__ == "__main__":


    for filtername in ('F466N', 'F470N', 'F480M', 'F444W'):
        make_linewidth_column_plot(filtername)
        make_linewidth_column_plot(filtername, temperature=20*u.K)
        make_temperature_linewidth_plot(filtername, column=1e17*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick')
        make_temperature_linewidth_plot(filtername, column=1e18*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick')
        make_temperature_linewidth_plot(filtername, column=1e19*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick')
