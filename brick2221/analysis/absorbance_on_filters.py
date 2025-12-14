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
#nh4p = read_lida_file(f'{optical_constants_cache_dir}/157_NH4+_1_12.0K.txt')
water_ammonia = read_ocdb_file(f'{optical_constants_cache_dir}/265_H2O:NH3_(4:1)_24K_Mukai.txt')
co_hudgins = read_ocdb_file(f'{optical_constants_cache_dir}/85_CO_(1)_10K_Hudgins.txt')
strong_icemix_hudgins = read_ocdb_file(f'{optical_constants_cache_dir}/119_H2O:CH3OH:CO:NH3_(100:50:1:1)_10K_Hudgins.txt')
#icemix_ehrenfreund = read_lida_file(f'{optical_constants_cache_dir}/35_H2O:CH3OH:CO2_(9:1:2)_10.0K.txt')


def plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co_hudgins, co2_gerakines, ethanol, methanol, ocn, water_ammonia),
                        colors=None,
                        ylim=(1e-21, 6e-18)):
    for ii, tb in enumerate(opacity_tables):

        molwt = u.Quantity(composition_to_molweight(tb.meta['composition']), u.Da)

        kk = tb['k₁'] if 'k₁' in tb.colnames else tb['k']
        #opacity = ((kk.quantity * tb['Wavelength'].to(u.cm**-1, u.spectral()) * 4 * np.pi / (1*u.g/u.cm**3 / (molwt)))).to(u.cm**2)

        # calculated "tau" with unitless ice_column to get the same as calculated above
        opacity = absorbed_spectrum(xarr=tb['Wavelength'], ice_column=1, ice_model_table=tb, molecular_weight=molwt, return_tau=True).to(u.cm**2)

        pl.plot(tb['Wavelength'],
                opacity,
                label=f'{tb.meta["author"]} {tb.meta["composition"]} {tb.meta["temperature"]}'
                        if 'author' in tb.meta else
                    f'{tb.meta["index"]} {tb.meta["molecule"]} {tb.meta["ratio"]} {tb.meta["temperature"]}',
                linestyle='-',
                color=colors[ii] if colors is not None else None,
                )
        # DEBUG if colors is not None:
        # DEBUG     print(f"table {ii} plotted with color {colors[ii]} [{tb.meta['composition']}].  colors={colors}")
    pl.legend(loc='lower left', bbox_to_anchor=(0, 1, 0, 0))
    pl.xlabel("Wavelength ($\\mu$m)")
    pl.ylabel("$\\kappa_{eff}$ [$\\tau = \\kappa_{eff} * N(ice)$]");
    pl.semilogy();
    pl.ylim(ylim);

def plot_filters(filternames=['F466N', 'F410M'], ymax=5e-18,
                 linestyles=['-', ':']):
    linestyle_cycle = itertools.cycle(linestyles)
    
    for filtername, linestyle in zip(filternames, linestyle_cycle):
        wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')
        xarr = wavelength_table['Wavelength'].quantity.to(u.um)
        pl.plot(xarr, wavelength_table['Transmission']/wavelength_table['Transmission'].max() * ymax,
                color='k', linewidth=3, alpha=0.5, zorder=-5, linestyle=linestyle)

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
            label=f'{tb.meta["author"]} {tb.meta["composition"]} {tb.meta["temperature"]}'
                    if 'author' in tb.meta else
                f'{tb.meta["index"]} {tb.meta["molecule"]} {tb.meta["ratio"]} {tb.meta["temperature"]}',
            linestyle='-',
            color=colors[ii] if colors is not None else None,
            **kwargs,
            )

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

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa))
    plot_filters()
    pl.xlim(4.55, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
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

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn,  ))
    plot_filters(['F466N', 'F410M', 'F405N'])
    pl.xlim(3.71, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_f410_f405.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,),
                        colors=[default_colors[ii] for ii  in [0,1,2]]
                        )
    #plot_filters()
    pl.xlim(1.11, 5.10);
    pl.ylim(1e-22, 6e-18);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_full_range.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn, methanol, ethanol, water_ammonia))
    plot_filters(filternames=['F356W', 'F444W',])# 'F466N', 'F410M'])
    pl.xlim(3.00, 5.05);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f356_and_f444.pdf', dpi=150, bbox_inches='tight')

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
    plot_filters(filternames=['F277W', 'F323N', 'F360M', 'F480M'])
    pl.xlim(2.00, 5.20);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f277_f323_f360_f480.pdf', dpi=150, bbox_inches='tight')


    pl.figure()
    plot_opacity_tables(opacity_tables=(water_mastrapa, ethanol, water_ammonia, co2_gerakines))
    plot_filters(filternames=['F410M', 'F430M', 'F460M', 'F480M'])
    pl.text(4.10, 6e-18, 'F410M', ha='center')
    pl.text(4.30, 6e-18, 'F430M', ha='center')
    pl.text(4.60, 6e-18, 'F460M', ha='center')
    pl.text(4.80, 6e-18, 'F480M', ha='center')
    pl.ylim(1e-21, 1e-17)
    pl.xlim(3.80, 5.20);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f410_f430_f460_f480.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(water_mastrapa, ethanol, water_ammonia))
    plot_filters(filternames=['F250M', 'F300M', 'F335M', 'F360M'])
    pl.text(2.5, 6e-18, 'F250M', ha='center')
    pl.text(3.0, 6e-18, 'F300M', ha='center')
    pl.text(3.35, 6e-18, 'F335M', ha='center')
    pl.text(3.6, 6e-18, 'F360M', ha='center')
    pl.ylim(1e-21, 1e-17)
    pl.xlim(2.30, 3.90);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f250_f300_f335_f360.pdf', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(water_mastrapa, ethanol, water_ammonia))
    plot_filters(filternames=['F277W', 'F356W', 'F444W'])
    pl.text(2.77, 6e-18, 'F277W', ha='center')
    pl.text(3.56, 6e-18, 'F356W', ha='center')
    pl.text(4.44, 6e-18, 'F444W', ha='center')
    pl.ylim(1e-21, 1e-17)
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