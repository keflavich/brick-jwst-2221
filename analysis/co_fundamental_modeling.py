# this takes a while to load the first time especially
from pyspeckitmodels.co import exomol_co_vibration
from pyspeckitmodels.sio import exomol_sio_vibration

from pyspeckitmodels.sio.exomol_sio_vibration import exomol_xsec as SiO_exomol_xsec
from pyspeckitmodels.co.exomol_co_vibration import exomol_xsec as CO_exomol_xsec, tau_of_N as CO_tau_of_N
from astropy import constants, units as u
from astroquery.svo_fps import SvoFps
import numpy as np
import pyspeckitmodels
from tqdm.autonotebook import tqdm


import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
pl.rcParams['figure.figsize'] = (10,8)
pl.rcParams['font.size'] = 16



def get_wltable(filtername, telescope='JWST', instrument='NIRCam'):
    return SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')


def fractional_absorption(temperature, column, linewidth, wavelength_table):

    xarr = wavelength_table['Wavelength'].quantity.to(u.um)
    chanwidth = 0.5 * linewidth/constants.c * xarr.mean()
    xarr = np.arange(xarr.min().to(u.um).value, xarr.max().to(u.um).value, (chanwidth).to(u.um).value) << u.um
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


def make_temperature_linewidth_plot(filtername, column=1e18*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick',
                                    levels=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9],
                                    colors=['w', 'y', 'g', 'c', 'r', 'b', 'purple', 'orange', 'maroon', 'springgreen'],
                                    telescope='JWST', instrument='NIRCam',
                                   ):
    wltable = get_wltable(filtername, telescope=telescope, instrument=instrument)
    temperatures = np.linspace(5, 150, 50)*u.K
    linewidths = np.linspace(1, 100, 24)*u.km/u.s
    grid = np.array([[fractional_absorption(T, column, sig, wltable) for T in temperatures]
                    for sig in tqdm(linewidths)])

    # y,x = linewidth, temperature
    assert grid.shape == (24, 50)

    pl.clf()
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



def make_linewidth_column_plot(filtername, temperature=60*u.K, basepath='/orange/adamginsburg/jwst/brick',
                               levels=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9],
                               colors=['w', 'y', 'g', 'c', 'r', 'b', 'purple', 'orange', 'maroon', 'springgreen'],
                               telescope='JWST', instrument='NIRCam',
                              ):
    wltable = get_wltable(filtername, telescope=telescope, instrument=instrument)
    linewidths = np.linspace(1, 100, 24)*u.km/u.s
    column = np.logspace(15,19,25)*u.cm**-2
    grid_sigcol = np.array([[fractional_absorption(temperature, col, sig, wltable) for col in column]
                    for sig in tqdm(linewidths)])

    # y,x = linewidth, column
    assert grid_sigcol.shape == (24, 25)

    pl.clf()
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


def fractional_absorption_ice(ice_column, center, width, ice_bandstrength,
                              filtername, telescope='JWST',
                              instrument='NIRCam', return_tau=False):

    wavelength_table = get_wltable(filtername, telescope=telescope, instrument=instrument)
    xarr = wavelength_table['Wavelength'].quantity.to(u.um)

    # ice widths are big
    chanwidth = 0.1 * width
    xarr = np.arange(xarr.min().to(u.um).value, xarr.max().to(u.um).value, (chanwidth).to(u.um).value) << u.um

    xarr_nu = xarr.to(u.cm**-1, u.spectral())
    dx = (xarr[1]-xarr[0])
    dx_icm = np.abs(xarr_nu[1] - xarr_nu[0])
    trans = np.interp(xarr, wavelength_table['Wavelength'], wavelength_table['Transmission'])
    transmission_sum = (trans).sum()

    center_nu = center.to(u.cm**-1, u.spectral())

    tau = np.zeros(xarr.size)
    wid_icm = (width / center) * center_nu
    line_profile = 1/((2*np.pi)**0.5 * wid_icm) * np.exp(-(xarr_nu-center_nu)**2/(2*wid_icm**2))
    #print(line_profile.unit, (line_profile * dx_icm).decompose().sum())

    # \int line_profile * dnu = 1

    tau = tau + (ice_column * ice_bandstrength * line_profile).decompose()
    assert tau.unit is u.dimensionless_unscaled
    if return_tau:
        print(ice_column, ice_bandstrength, "tau:", tau.sum(), "tau dnu sum:", (tau*dx_icm).sum(), "line prof:", (line_profile * dx_icm).sum(), "width:", width, "width icm:",wid_icm, dx_icm)
        return tau
    absorbed_fraction = (((1-np.exp(-tau)) * trans).sum() / transmission_sum).value
    return absorbed_fraction


def fracfluxplot(filtername, telescope='JWST', instrument='NIRCam', basepath='/orange/adamginsburg/jwst/brick',):

    # includes the CO ice bandstrengths, etc.
    from icemodels.gaussian_model_components import co_ice_wls, co_ice_widths, co_ice_bandstrength, ocn_center, ocn_width, ocn_bandstrength

    column = np.logspace(16, 21, 100)*u.cm**-2
    grid_co_in_co2_ice = [fractional_absorption_ice(col, co_ice_wls[0], co_ice_widths[0], co_ice_bandstrength, filtername=filtername, telescope=telescope, instrument=instrument) for col in column]
    grid_co_pure_ice = [fractional_absorption_ice(col, co_ice_wls[1], co_ice_widths[1], co_ice_bandstrength, filtername=filtername, telescope=telescope, instrument=instrument) for col in column]
    grid_co_methanol_ice = [fractional_absorption_ice(col, co_ice_wls[2], co_ice_widths[2], co_ice_bandstrength, filtername=filtername, telescope=telescope, instrument=instrument) for col in column]
    grid_ocn = [fractional_absorption_ice(col, ocn_center, ocn_width, ocn_bandstrength, filtername=filtername, telescope=telescope, instrument=instrument) for col in column]

    fig = pl.figure(figsize=(10, 8))
    ax = fig.gca()

    LCO, = pl.semilogx(column, grid_co_in_co2_ice, label='CO in CO$_2$')
    pl.semilogx(column, grid_co_pure_ice, label='Pure CO')
    pl.semilogx(column, grid_co_methanol_ice, label='CO in CH$_3$OH')
    LOCN, = pl.semilogx(column, grid_ocn, linestyle='--', label='OCN-')

    def col2mag(x):
        ret = -2.5*np.log10(1-np.array(x))
        ret[~np.isfinite(ret)] = 999
        return ret
    def mag2col(x):
        return 1-10**(-x/2.5)

    secax = ax.secondary_yaxis('right', functions=(col2mag, mag2col, ))
    secax.set_ylabel(f"Absorption in {filtername} filter [magnitudes]")
    secax.set_ticks([0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 1, 1.5, 2, 3, 4])

    pl.ylim(0.02, 1)# NOTE: conversion breaks if this is not 1!!
    pl.legend(loc='best')
    pl.xlabel("Column density of CO or OCN- in specified ice state")
    pl.ylabel(f"Absorption in {filtername} filter [fractional]")

    fig.savefig(f'{basepath}/paper_figures/{filtername}_IceAbsorptionvsColumn_linear.pdf', bbox_inches='tight');

    pl.arrow(2.7e18, 0, 0, 0.04, color=LCO.get_color(), label='N(CO)$_{max}$ Boogert+ 2022', head_width=2.7e17, head_length=0.02)
    pl.arrow(2.6e17, 0, 0, 0.04, color=LOCN.get_color(), linestyle='--', label='N(OCN)$_{max}$ Boogert+ 2022', head_width=2.6e16, head_length=0.02)
    pl.legend(loc='best')

    fig.savefig(f'{basepath}/paper_figures/{filtername}_IceAbsorptionvsColumn_linear_withBoogertMaxima.pdf', bbox_inches='tight');

    pl.semilogx(column, -2.5*np.log10(1-np.array(grid_co_in_co2_ice)), label='CO in CO$_2$')
    pl.semilogx(column, -2.5*np.log10(1-np.array(grid_co_pure_ice)), label='Pure CO')
    pl.semilogx(column, -2.5*np.log10(1-np.array(grid_co_methanol_ice)), label='CO in CH$_3$OH')
    pl.semilogx(column, -2.5*np.log10(1-np.array(grid_ocn)), label='OCN-')
    #pl.semilogx(column, grid_co_in_co2_ice, label='CO in CO$_2$')
    #pl.semilogx(column, grid_co_pure_ice, label='Pure CO')
    #pl.semilogx(column, grid_co_methanol_ice, label='CO in CH$_3$OH')
    #pl.semilogx(column, grid_ocn, label='OCN-')


    def col2mag(x):
        return -2.5*np.log10(1-np.array(x))
    def mag2col(x):
        return 1-10**(-x/2.5)

    ax = pl.gca()
    secax = ax.secondary_yaxis('right', functions=(mag2col, col2mag, ))
    secax.set_ylabel(f"Absorption in {filtername} filter [fractional]")

    pl.legend(loc='best')
    pl.xlabel("Column density of CO or OCN- in specified ice state")
    pl.ylabel(f"Absorption in {filtername} filter [magnitudes]")
    pl.ylim([0,5]);
    pl.savefig(f'{basepath}/paper_figures/IceAbsorptionvsColumn_magnitudes.pdf', bbox_inches='tight');

    pl.axvline(2.7e18, color='k', label='Largest N(CO) observed by Boogert+ 2022')
    pl.axvline(2.6e17, color='k', linestyle='--', label='Largest N(OCN) observed by Boogert+ 2022')
    pl.legend(loc='best')

    pl.savefig(f'{basepath}/paper_figures/IceAbsorptionvsColumn_magnitudes_withBoogertMaxima.pdf', bbox_inches='tight');


if __name__ == "__main__":

    filtername = 'I2'
    make_linewidth_column_plot(filtername, instrument='IRAC', telescope='Spitzer')
    make_linewidth_column_plot(filtername, temperature=20*u.K, instrument='IRAC', telescope='Spitzer')
    make_temperature_linewidth_plot(filtername, column=1e17*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick', instrument='IRAC', telescope='Spitzer')
    make_temperature_linewidth_plot(filtername, column=1e18*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick', instrument='IRAC', telescope='Spitzer')
    make_temperature_linewidth_plot(filtername, column=1e19*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick', instrument='IRAC', telescope='Spitzer')
    fracfluxplot(filtername)

    for filtername in ('F470N', 'F480M', 'F444W', 'F466N', ):
        make_linewidth_column_plot(filtername)
        make_linewidth_column_plot(filtername, temperature=20*u.K)
        make_temperature_linewidth_plot(filtername, column=1e17*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick')
        make_temperature_linewidth_plot(filtername, column=1e18*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick')
        make_temperature_linewidth_plot(filtername, column=1e19*u.cm**-2, basepath='/orange/adamginsburg/jwst/brick')
        fracfluxplot(filtername)
