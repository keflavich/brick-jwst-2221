"""
Overeplot absorbance profiles on JWST Filter transmission profiles
"""


import importlib as imp
import itertools
import icemodels
imp.reload(icemodels.core)
imp.reload(icemodels)
from icemodels.core import (absorbed_spectrum, absorbed_spectrum_Gaussians, convsum,
                            optical_constants_cache_dir,
                            download_all_ocdb,
                            retrieve_gerakines_co,
                            read_lida_file,
                            download_all_lida,
                            composition_to_molweight,
                            fluxes_in_filters, load_molecule_univap, load_molecule, load_molecule_ocdb, atmo_model, molecule_data, read_ocdb_file)
from icemodels.absorbance_in_filters import make_mixtable

from brick2221.analysis.analysis_setup import basepath

from astropy.table import Table

from astroquery.svo_fps import SvoFps
instrument = 'NIRCam'
telescope = 'JWST'
filt444 = 'F444W'
filt356 = 'F356W'
filt466 = 'F466N'
filt410 = 'F410M'
filt470 = 'F470N'
wavelength_table_466 = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt466}')
wavelength_table_410 = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt410}')
wavelength_table_444 = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt444}')
wavelength_table_356 = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt356}')

from astropy import units as u
import numpy as np

import pylab as pl
from cycler import cycler
import re


# Load mix bases
water_mastrapa = read_ocdb_file(f'{optical_constants_cache_dir}/240_H2O_(1)_25K_Mastrapa.txt') # h2otbs[('ocdb', 242, 25)] 242 is 50K....
co2_gerakines = read_ocdb_file(f'{optical_constants_cache_dir}/55_CO2_(1)_8K_Gerakines.txt') # co2tbs[('ocdb', 55, 8)]
#ethanol = read_lida_file(f'{optical_constants_cache_dir}/87_CH3CH2OH_1_30.0K.txt')
#methanol = read_lida_file(f'{optical_constants_cache_dir}/58_CH3OH_1_25.0K.txt')
ethanol = load_molecule_univap('ethanol')
methanol = load_molecule_univap('methanol')
ocn = read_lida_file(f'{optical_constants_cache_dir}/158_OCN-_1_12.0K.txt')
co_gerakines = gerakines = retrieve_gerakines_co()
nh3 = read_ocdb_file(f'{optical_constants_cache_dir}/273_NH3_(1)_40K_Roser.txt')
#nh3 = read_lida_file(f'{optical_constants_cache_dir}/116_NH3_1_27.0K.txt')
nh4p = read_lida_file(f'{optical_constants_cache_dir}/157_NH4+_1_12.0K.txt')
water_ammonia = read_ocdb_file(f'{optical_constants_cache_dir}/265_H2O:NH3_(4:1)_24K_Mukai.txt')
co_hudgins = read_ocdb_file(f'{optical_constants_cache_dir}/85_CO_(1)_10K_Hudgins.txt')
strong_icemix_hudgins = read_ocdb_file(f'{optical_constants_cache_dir}/119_H2O:CH3OH:CO:NH3_(100:50:1:1)_10K_Hudgins.txt')
#icemix_ehrenfreund = read_lida_file(f'{optical_constants_cache_dir}/35_H2O:CH3OH:CO2_(9:1:2)_10.0K.txt')


def _format_author_label(author):
    author = str(author).strip()
    if '+' in author:
        return author
    match = re.fullmatch(r'([A-Za-z]+)(\d{4})([A-Za-z]?)', author)
    if match:
        surname, year, suffix = match.groups()
        return f'{surname}+ {year}{suffix}'
    parts = [part.strip() for part in author.split(',') if part.strip()]
    if len(parts) > 1:
        return ', '.join(_format_author_label(part) for part in parts)
    if re.fullmatch(r'[A-Za-z]+', author):
        return f'{author}+'
    return author


def _format_species_label(species):
    species = str(species).strip()
    species = re.sub(r'\s*\(1\)\s*', ' ', species)
    species = re.sub(r'([A-Za-z])([0-9]+)', r'\1$_\2$', species)
    species = re.sub(r'\s+', ' ', species).strip()
    return species


def _format_temperature_label(temperature):
    temperature = str(temperature).strip()
    if temperature.lower().endswith('k'):
        temperature = temperature[:-1].strip()
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', temperature):
        temperature = f'{float(temperature):g}'
    return f'{temperature}K'


def _format_ratio_label(ratio):
    ratio = str(ratio).strip()
    # Replace spaces with colons in ratios like (4 1) -> (4:1)
    ratio = re.sub(r'\(\s*(\d+)\s+(\d+(?:\s+\d+)*)\s*\)', lambda m: f'({m.group(1)}:{m.group(2).replace(" ", ":")})', ratio)
    return ratio


def _opacity_label(tb):
    if 'author' in tb.meta:
        author = _format_author_label(tb.meta['author'])
        composition = _format_species_label(tb.meta['composition'])
        temperature = _format_temperature_label(tb.meta['temperature'])
        return f'{author} {composition} {temperature}'

    molecule = _format_species_label(tb.meta['molecule'])
    temperature = _format_temperature_label(tb.meta['temperature'])
    ratio = _format_ratio_label(tb.meta['ratio'])
    return f'{tb.meta["index"]} {molecule} {ratio} {temperature}'


def _opacity_label_from_meta(meta):
    if 'author' in meta:
        author = _format_author_label(meta['author'])
        composition = _format_species_label(meta['composition'])
        temperature = _format_temperature_label(meta['temperature'])
        return f'{author} {composition} {temperature}'

    molecule = _format_species_label(meta['molecule'])
    temperature = _format_temperature_label(meta['temperature'])
    ratio = _format_ratio_label(meta['ratio'])
    return f'{meta["index"]} {molecule} {ratio} {temperature}'


def plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co_hudgins, co2_gerakines, ethanol, methanol, ocn, water_ammonia),
                        colors=None,
                        ylim=(1e-21, 6e-18),
                        legend=True
                        ):
    for ii, tb in enumerate(opacity_tables):

        molwt = u.Quantity(composition_to_molweight(tb.meta['composition']), u.Da)

        kk = tb['k₁'] if 'k₁' in tb.colnames else tb['k']
        #opacity = ((kk.quantity * tb['Wavelength'].to(u.cm**-1, u.spectral()) * 4 * np.pi / (1*u.g/u.cm**3 / (molwt)))).to(u.cm**2)

        # calculated "tau" with unitless ice_column to get the same as calculated above
        opacity = absorbed_spectrum(xarr=tb['Wavelength'], ice_column=1, ice_model_table=tb, molecular_weight=molwt, return_tau=True).to(u.cm**2)

        pl.plot(tb['Wavelength'],
                opacity,
            label=_opacity_label(tb),
                linestyle='-',
                color=colors[ii] if colors is not None else None,
                )
        # DEBUG if colors is not None:
        # DEBUG     print(f"table {ii} plotted with color {colors[ii]} [{tb.meta['composition']}].  colors={colors}")
    if legend:
        pl.legend(loc='lower left', bbox_to_anchor=(0, 1, 0, 0))
    pl.xlabel("Wavelength ($\\mu$m)")
    pl.ylabel("$\\kappa_{eff}$ [$\\tau = \\kappa_{eff} * N(ice)$]");
    pl.semilogy();
    pl.ylim(ylim);

def plot_filters(filternames=['F466N', 'F410M'], ymax=5e-18,
                 linestyles=['-', ':']):
    linestyle_cycle = itertools.cycle(linestyles)

    ax = pl.gca()
    transmission_ax = ax.twinx()
    tmax = 0

    for filtername, linestyle in zip(filternames, linestyle_cycle):
        wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')
        xarr = wavelength_table['Wavelength'].quantity.to(u.um)
        transmission = wavelength_table['Transmission']
        transmission_ax.plot(
            xarr,
            transmission,
            color='k',
            linewidth=2,
            alpha=0.5,
            #zorder=-5,
            linestyle=linestyle,
        )
        tmax = max(tmax, wavelength_table['Transmission'].max())

    transmission_ax.set_ylim(0, tmax * 1.05)
    transmission_ax.set_ylabel('Transmission')

    return ax, transmission_ax, tmax

def plot_mixed_opacity(opacity_tables={'CO': co_gerakines,
                                       'H2O': water_mastrapa,
                                       'CO2': co2_gerakines,
                                       'CH3CH2OH': ethanol,
                                       'CH3OH': methanol,
                                       'OCN': ocn,
                                       #'NH4+': nh4p,
                                       'NH3': nh3, },
                        mixture={'CO': 1},
                        colors=None,
                        normalize_to_molecule=False,
                        ylim=(1e-21, 6e-18),
                        legend=True,
                        **kwargs):

    authors = {mol: tb.meta['author'] for mol, tb in opacity_tables.items()}

    grid = np.linspace(2.5*u.um, 5.2*u.um, 20000)
    composition = ':'.join(mixture.keys()) + " (" + ":".join([str(val) for val in mixture.values()]) + ")"
    print(f"composition: {composition}")
    tb = make_mixtable(composition, moltbls=opacity_tables, grid=grid, density=1*u.g/u.cm**3, temperature=25*u.K,
                       authors=', '.join([authors[mol] for mol in composition.split(' ')[0].split(':')]),)

    molwt = u.Quantity(composition_to_molweight(tb.meta['composition']), u.Da)
    opacity = absorbed_spectrum(xarr=tb['Wavelength'], ice_column=1, ice_model_table=tb, molecular_weight=molwt, return_tau=True).to(u.cm**2)

    #kk = tb['k₁'] if 'k₁' in tb.colnames else tb['k']
    #opacity = ((kk.quantity * tb['Wavelength'].to(u.cm**-1, u.spectral()) * 4 * np.pi / (1*u.g/u.cm**3 / (molwt)))).to(u.cm**2)

    if normalize_to_molecule:
        total = sum(mixture.values())
        molval = mixture[normalize_to_molecule]
        molfrac_mol = molval / total
        opacity = opacity * molfrac_mol

    pl.plot(tb['Wavelength'],
            opacity,
            label=_opacity_label_from_meta(tb.meta),
            linestyle='-',
            color=colors[0] if colors is not None else None,
            **kwargs,
            )

    if legend:
        pl.legend(loc='lower left', bbox_to_anchor=(0, 1, 0, 0))
    pl.xlabel("Wavelength ($\\mu$m)")
    pl.ylabel("$\\kappa_{eff}$ [$\\tau = \\kappa_{eff} * N(ice)$]");
    pl.semilogy();
    pl.ylim(ylim);

    return tb


if __name__ == "__main__":

    default_colors = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
    if default_colors[1] != '#ff7f0e':
        print("DANGER: default colors broke.  Setting them back to normal.")
        pl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    pl.rcParams['figure.figsize'] = (8.5, 4)

    pl.figure(figsize=(4.25, 4))
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa))
    plot_filters()
    pl.xlim(4.55, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466.pdf', dpi=150, bbox_inches='tight')

    pl.figure(figsize=(8.5, 4))
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co_hudgins))
    plot_filters()
    pl.xlim(4.55, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_withhudgins.pdf', dpi=150, bbox_inches='tight')

    # BROKEN for no reason!?!?!?!?!?!?!?!?
    default_colors = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
    default_colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    print(f"default_colors: {default_colors}")
    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, ocn),
                       colors=[default_colors[ii] for ii  in [0,1,3]]
    )
    plot_filters()
    pl.xlim(4.55, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_withocn.pdf', dpi=150, bbox_inches='tight')



    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn,  ))
    plot_filters()
    pl.xlim(3.71, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_and_f410.pdf', dpi=150, bbox_inches='tight')

    pl.figure(figsize=(8.5, 4))
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn,  ))
    ax, transmission_ax, tmax = plot_filters(['F466N', 'F410M', 'F405N'])
    transmission_ax.text(4.66, tmax * 1.01, 'F466N', ha='center')
    transmission_ax.text(4.10, tmax * 1.01, 'F410M', ha='center')
    transmission_ax.text(4.05, tmax * 1.01, 'F405N', ha='center')
    pl.xlim(3.71, 4.75);
    ax.set_ylim(1e-21, 1e-17)
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_f410_f405.pdf', dpi=150, bbox_inches='tight')

    # New figure: F466N, F410M, F405N with all valid pure-water-ice opacity tables.
    # Phase classification (amorphous / crystalline / liquid) drives linestyle;
    # author drives color. Tables with wavelength coverage that does not span
    # [4.0, 4.7] um are skipped (e.g. Mukai 23K, which begins at 4.17 um).
    import glob
    import os as _os

    # (filename glob pattern under optical_constants_cache_dir, phase, author_key)
    # phase rules (from literature):
    #   - Mastrapa et al. 2008/2009: T <= 100 K amorphous, T >= 110 K crystalline
    #   - Hudgins et al. 1993: T <= 100 K amorphous, T >= 120 K crystalline
    #   - Kitta, Leger, Mukai (cold deposit): amorphous
    #   - Bertie 1969 (100 K, annealed): crystalline Ih
    #   - Curtis 2005, Rajaram 2010 (>=106 K, annealed): crystalline
    #   - Clapp 1995 (190 K): crystalline (cubic/Ih)
    #   - Zhang 2013 (243 K): liquid
    def _classify_water(author, temperature):
        author = str(author).strip()
        tstr = str(temperature).strip()
        if tstr.lower().endswith('k'):
            tstr = tstr[:-1].strip()
        T = float(tstr)
        if author == 'Mastrapa':
            return 'amorphous' if T < 110 else 'crystalline'
        if author == 'Hudgins':
            return 'amorphous' if T <= 100 else 'crystalline'
        if author in ('Kitta', 'Léger', 'Leger', 'Mukai'):
            return 'amorphous'
        if author in ('Bertie', 'Curtis', 'Rajaram', 'Clapp'):
            return 'crystalline'
        if author == 'Zhang':
            return 'liquid'
        return 'unknown'

    phase_linestyle = {'amorphous': '-', 'crystalline': '--', 'liquid': ':', 'unknown': '-.'}
    author_color = {
        'Mastrapa':    '#1f77b4',
        'Hudgins':     '#ff7f0e',
        'Bertie':      '#2ca02c',
        'Clapp':       '#d62728',
        'Kitta':       '#9467bd',
        'Léger':       '#8c564b',
        'Leger':       '#8c564b',
        'Mukai':       '#e377c2',
        'Curtis':      '#7f7f7f',
        'Rajaram':     '#bcbd22',
        'Zhang':       '#17becf',
    }

    water_files = sorted(glob.glob(f'{optical_constants_cache_dir}/*_H2O_(1)_*K_*.txt'))
    water_entries = []
    for fn in water_files:
        try:
            tb_w = read_ocdb_file(fn)
        except Exception as ex:
            print(f"Skipping {fn}: read error {ex}")
            continue
        wl = np.asarray(tb_w['Wavelength'], dtype=float)
        # Require coverage of both 4.05 and 4.66 um
        if not (np.isfinite(wl).any() and wl.min() <= 4.0 and wl.max() >= 4.7):
            print(f"Skipping {_os.path.basename(fn)}: wl range {wl.min():.2f}-{wl.max():.2f} um")
            continue
        author = tb_w.meta.get('author', '')
        temperature = tb_w.meta.get('temperature', np.nan)
        phase = _classify_water(author, temperature)
        tstr = str(temperature).strip()
        if tstr.lower().endswith('k'):
            tstr = tstr[:-1].strip()
        try:
            T_num = float(tstr)
        except ValueError:
            T_num = np.nan
        water_entries.append((tb_w, author, T_num, phase))

    # Group by (author, phase). For multi-T groups, plot a min/max envelope
    # (fill_between) plus the median curve to keep the legend short. Single-T
    # groups plot a single curve. Color = author, linestyle = phase.
    from collections import defaultdict
    from matplotlib.lines import Line2D

    grid = np.linspace(3.7, 4.8, 1100)  # um, plot range

    grouped = defaultdict(list)
    for tb_w, author, T, phase in water_entries:
        molwt = u.Quantity(composition_to_molweight(tb_w.meta['composition']), u.Da)
        opacity = absorbed_spectrum(xarr=tb_w['Wavelength'], ice_column=1,
                                    ice_model_table=tb_w, molecular_weight=molwt,
                                    return_tau=True).to(u.cm**2).value
        wl = np.asarray(tb_w['Wavelength'], dtype=float)
        so = np.argsort(wl)
        wl = wl[so]; opacity = opacity[so]
        op_on_grid = np.interp(grid, wl, opacity, left=np.nan, right=np.nan)
        grouped[(author, phase)].append((T, op_on_grid))

    pl.figure(figsize=(9, 5))
    phase_order = {'amorphous': 0, 'crystalline': 1, 'liquid': 2, 'unknown': 3}
    legend_handles = []
    for (author, phase), entries in sorted(grouped.items(),
                                           key=lambda kv: (phase_order[kv[0][1]], kv[0][0])):
        Ts = sorted(e[0] for e in entries)
        stack = np.array([e[1] for e in entries])
        color = author_color.get(author, None)
        ls = phase_linestyle.get(phase, '-')
        if len(entries) >= 2:
            lo = np.nanmin(stack, axis=0)
            hi = np.nanmax(stack, axis=0)
            med = np.nanmedian(stack, axis=0)
            pl.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
            pl.plot(grid, med, color=color, linestyle=ls, alpha=0.95, linewidth=1.4)
            label = f"{author} {phase} ({len(entries)} T: {Ts[0]:g}–{Ts[-1]:g} K)"
        else:
            pl.plot(grid, stack[0], color=color, linestyle=ls, alpha=0.95, linewidth=1.4)
            label = f"{author} {phase} ({Ts[0]:g} K)"
        legend_handles.append(Line2D([0], [0], color=color, linestyle=ls,
                                     linewidth=1.4, label=label))

    pl.xlabel("Wavelength ($\\mu$m)")
    pl.ylabel("$\\kappa_{eff}$ [$\\tau = \\kappa_{eff} * N(ice)$]")
    pl.semilogy()
    pl.ylim(1e-21, 1e-17)
    ax, transmission_ax, tmax = plot_filters(['F466N', 'F410M', 'F405N'])
    transmission_ax.text(4.66, tmax * 1.01, 'F466N', ha='center')
    transmission_ax.text(4.10, tmax * 1.01, 'F410M', ha='center')
    transmission_ax.text(4.05, tmax * 1.01, 'F405N', ha='center')
    pl.xlim(3.71, 4.75)
    ax.set_ylim(1e-21, 1e-17)
    # Phase legend (linestyle key)
    phase_handles = [Line2D([0], [0], color='k', linestyle=ls, label=ph)
                     for ph, ls in phase_linestyle.items()
                     if any(e[3] == ph for e in water_entries)]
    leg1 = ax.legend(handles=phase_handles, loc='upper left', fontsize=8, title='phase')
    ax.add_artist(leg1)
    # Group legend (author × phase)
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.08, 1.0), fontsize=8, frameon=True,
              title='group (shaded = T range)')
    pl.title("Pure H$_2$O ice opacities — bands span deposit T range")
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_f410_f405_water_versions.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,),
                        colors=[default_colors[ii] for ii  in [0,1,2]]
                        )
    #plot_filters()
    pl.xlim(1.11, 5.10);
    pl.ylim(1e-22, 6e-18);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_full_range.pdf', dpi=150, bbox_inches='tight')

    pl.figure(figsize=(8.5, 4))
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn, methanol, ethanol, water_ammonia))
    plot_filters(filternames=['F356W', 'F444W',])# 'F466N', 'F410M'])
    pl.text(3.56, 6e-18, 'F356W', ha='center')
    pl.text(4.44, 6e-18, 'F444W', ha='center')
    pl.xlim(3.00, 5.05);
    pl.ylim(1e-21, 1e-17)
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f356_and_f444.pdf', dpi=150, bbox_inches='tight')


    # special: merge figure 3+4 for paper
    pl.figure(figsize=(8.5, 6))
    # First subplot
    pl.subplot(2,1,1)
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn,  ))
    ax1, transmission_ax1, tmax1 = plot_filters(['F466N', 'F410M', 'F405N'])
    transmission_ax1.text(4.66, tmax1 * 1.01, 'F466N', ha='center')
    transmission_ax1.text(4.15, tmax1 * 1.01, 'F410M', ha='center')
    transmission_ax1.text(4.05, tmax1 * 1.01, 'F405N', ha='center')
    pl.xlim(3.71, 4.75)
    ax1.set_ylim(1e-21, 1.2e-17)
    transmission_ax1.set_ylim(0, tmax1 * 1.05)
    # Second subplot
    pl.subplot(2,1,2)
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn, methanol, ethanol, water_ammonia), legend=False)
    ax2, transmission_ax2, tmax2 = plot_filters(filternames=['F356W', 'F444W',])
    transmission_ax2.text(3.56, tmax2 * 1.01, 'F356W', ha='center')
    transmission_ax2.text(4.44, tmax2 * 1.01, 'F444W', ha='center')
    pl.xlim(3.00, 5.05)
    ax2.set_ylim(1e-21, 1.2e-17)
    transmission_ax2.set_ylim(0, tmax2 * 1.10)
    handles, labels = pl.gca().get_legend_handles_labels()
    pl.subplot(2,1,1)
    leg = pl.legend(handles=handles,
        labels=labels,
        loc='lower left',
        bbox_to_anchor=(0, 1, 0, 0),
        ncol=2,
        mode=None,
    )
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_figure3plus4merge.pdf', dpi=150, bbox_inches='tight')



    ocn_mix1 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(1:1:1).ecsv')
    ocn_mix2 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(2:1:0.1).ecsv')
    ocn_mix3 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(1:1:0.02).ecsv')
    ocn_mix4 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(2:1:0.5).ecsv')
    ocn_mix5 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/CO:OCN_(1:1).ecsv')

    pl.figure()
    plot_opacity_tables(opacity_tables=(ocn_mix1, ocn_mix2, ocn_mix3, ocn_mix4, ocn_mix5,))
    plot_filters()
    pl.xlim(3.71, 4.75);

    pl.figure()
    plot_opacity_tables(opacity_tables=(water_mastrapa, ethanol, water_ammonia))
    ax, transmission_ax, tmax = plot_filters(filternames=['F277W', 'F323N', 'F360M', 'F480M'])
    transmission_ax.text(2.77, tmax * 1.01, 'F277W', ha='center')
    transmission_ax.text(3.23, tmax * 1.01, 'F323N', ha='center')
    transmission_ax.text(3.60, tmax * 1.01, 'F360M', ha='center')
    transmission_ax.text(4.80, tmax * 1.01, 'F480M', ha='center')
    ax.set_ylim(1e-21, 1e-17)
    pl.xlim(2.00, 5.20);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f277_f323_f360_f480.pdf', dpi=150, bbox_inches='tight')


    pl.figure()
    plot_opacity_tables(opacity_tables=(water_mastrapa, ethanol, water_ammonia, co2_gerakines))
    ax, transmission_ax, tmax = plot_filters(filternames=['F410M', 'F430M', 'F460M', 'F480M'])
    transmission_ax.text(4.10, tmax * 1.01, 'F410M', ha='center')
    transmission_ax.text(4.30, tmax * 1.01, 'F430M', ha='center')
    transmission_ax.text(4.60, tmax * 1.01, 'F460M', ha='center')
    transmission_ax.text(4.80, tmax * 1.01, 'F480M', ha='center')
    ax.set_ylim(1e-21, 1e-17)
    pl.xlim(3.80, 5.20);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f410_f430_f460_f480.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(water_mastrapa, ethanol, water_ammonia))
    ax, transmission_ax, tmax = plot_filters(filternames=['F250M', 'F300M', 'F335M', 'F360M'])
    transmission_ax.text(2.5, tmax * 1.01, 'F250M', ha='center')
    transmission_ax.text(3.0, tmax * 1.01, 'F300M', ha='center')
    transmission_ax.text(3.35, tmax * 1.01, 'F335M', ha='center')
    transmission_ax.text(3.6, tmax * 1.01, 'F360M', ha='center')
    ax.set_ylim(1e-21, 1e-17)
    pl.xlim(2.30, 3.90);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f250_f300_f335_f360.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(water_mastrapa, ethanol, water_ammonia))
    ax, transmission_ax, tmax = plot_filters(filternames=['F277W', 'F356W', 'F444W'])
    transmission_ax.text(2.77, tmax * 1.01, 'F277W', ha='center')
    transmission_ax.text(3.56, tmax * 1.01, 'F356W', ha='center')
    transmission_ax.text(4.44, tmax * 1.01, 'F444W', ha='center')
    ax.set_ylim(1e-21, 1e-17)
    pl.xlim(2.20, 5.20);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f277_f356_f444.pdf', dpi=150, bbox_inches='tight')


    pl.close('all')
    # compare my mixture to real mixture
    pl.figure()
    plot_mixed_opacity(mixture={'H2O': 100, 'CH3OH': 50, 'CO': 1, 'NH3': 1},)
    plot_opacity_tables(opacity_tables=(strong_icemix_hudgins,))
    pl.xlim(2.71, 5.25);
    pl.ylim(1e-22, 1e-18)
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_longwavelength_compare_mix_to_realmix.pdf', dpi=150, bbox_inches='tight')

    pl.close('all')
    # compare the ingredients my mixture to real mixture
    pl.figure(figsize=(8.5, 4))
    plot_mixed_opacity(mixture={'H2O': 100, 'CH3OH': 50, 'CO': 1, 'NH3': 1},)
    plot_opacity_tables(opacity_tables=(strong_icemix_hudgins,))
    plot_opacity_tables(opacity_tables=(water_mastrapa,))
    plot_opacity_tables(opacity_tables=(methanol,))
    plot_opacity_tables(opacity_tables=(co_gerakines,))
    plot_opacity_tables(opacity_tables=(nh3,))
    pl.xlim(2.71, 5.25);
    pl.ylim(1e-22, 1e-17)
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_longwavelength_compare_mix_components_to_realmix.pdf', dpi=150, bbox_inches='tight')

    pl.close('all')
    # compare my mixture to real mixture
    # not k-measured pl.figure()
    # not k-measured plot_mixed_opacity(mixture={'H2O': 9, 'CH3OH': 1, 'CO2': 2},)
    # not k-measured plot_opacity_tables(opacity_tables=(icemix_ehrenfreund,))
    # not k-measured pl.xlim(2.71, 5.25);
    # not k-measured pl.ylim(1e-22, 1e-18)
    # not k-measured pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_longwavelength_compare_mix_to_realmix_ehrenfreund.pdf', dpi=150, bbox_inches='tight')

    # pl.clf()
    # tb = plot_mixed_opacity(mixture={'H2O': 100, 'CH3OH': 50, 'CO': 1, 'NH3': 1},)
    # pl.ylim(1e-35, 1e-10)
    # pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/debug.pdf', dpi=150, bbox_inches='tight')


    pl.figure()
    plot_mixed_opacity(mixture={'H2O': 10, 'CO': 1, 'CO2': 1, 'CH3OH': 1, }, normalize_to_molecule='CO')
    plot_mixed_opacity(mixture={'H2O': 10, 'CO': 1, 'CO2': 1}, normalize_to_molecule='CO', linewidth=0.5)
    plot_mixed_opacity(mixture={'H2O': 5,  'CO': 1, 'CO2': 1}, normalize_to_molecule='CO', linewidth=0.5)
    plot_mixed_opacity(mixture={'H2O': 20, 'CO': 1, 'CO2': 1}, normalize_to_molecule='CO', linewidth=0.5)
    pl.ylabel("$\\kappa_{eff}$ [$\\tau = \\kappa_{eff} * N(CO_{ice})$]");
    pl.ylim(1e-22, 1e-19)
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_longwavelength_compare_mixtures_normalized.pdf', dpi=150, bbox_inches='tight')