"""
Overeplot absorbance profiles on JWST Filter transmission profiles
"""


import importlib as imp
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
                            fluxes_in_filters, load_molecule, load_molecule_ocdb, atmo_model, molecule_data, read_ocdb_file)

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
ethanol = read_lida_file(f'{optical_constants_cache_dir}/87_CH3CH2OH_1_30.0K.txt')
methanol = read_lida_file(f'{optical_constants_cache_dir}/58_CH3OH_1_25.0K.txt')
ocn = read_lida_file(f'{optical_constants_cache_dir}/158_OCN-_1_12.0K.txt')
co_gerakines = gerakines = retrieve_gerakines_co()
#nh3 = read_ocdb_file(f'{optical_constants_cache_dir}/273_NH3_(1)_40K_Roser.txt')
#nh3 = read_lida_file(f'{optical_constants_cache_dir}/116_NH3_1_27.0K.txt')
nh4p = read_lida_file(f'{optical_constants_cache_dir}/157_NH4+_1_12.0K.txt')
water_ammonia = read_ocdb_file(f'{optical_constants_cache_dir}/265_H2O:NH3_(4:1)_24K_Mukai.txt')
co_hudgins = read_ocdb_file(f'{optical_constants_cache_dir}/85_CO_(1)_10K_Hudgins.txt')


def plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co_hudgins, co2_gerakines, ethanol, methanol, ocn, nh4p, water_ammonia),
                        colors=None,
                        ylim=(1e-21, 6e-18)):
    for ii, tb in enumerate(opacity_tables):

        molwt = u.Quantity(composition_to_molweight(tb.meta['composition']), u.Da)

        kk = tb['k₁'] if 'k₁' in tb.colnames else tb['k']
        opacity = ((kk.quantity * tb['Wavelength'].to(u.cm**-1, u.spectral()) * 4 * np.pi / (1*u.g/u.cm**3 / (molwt)))).to(u.cm**2)
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

def plot_filters(filternames=['F466N', 'F410M'], ymax=5e-18):
    for filtername in filternames:
        wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')
        xarr = wavelength_table['Wavelength'].quantity.to(u.um)
        pl.plot(xarr, wavelength_table['Transmission']/wavelength_table['Transmission'].max() * ymax,
                color='k', linewidth=3, alpha=0.5, zorder=-5)


if __name__ == "__main__":

    default_colors = list(pl.rcParams['axes.prop_cycle'].by_key()['color'])
    if default_colors[1] != '#ff7f0e':
        print("DANGER: default colors broke.  Setting them back to normal.")
        pl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa))
    plot_filters()
    pl.xlim(4.55, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466.png', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co_hudgins))
    plot_filters()
    pl.xlim(4.55, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_withhudgins.png', dpi=150, bbox_inches='tight')

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
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_withocn.png', dpi=150, bbox_inches='tight')



    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn,  ))
    plot_filters()
    pl.xlim(3.71, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_and_f410.png', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn,  ))
    plot_filters(['F466N', 'F410M', 'F405N'])
    pl.xlim(3.71, 4.75);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f466_f410_f405.png', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,),
                        colors=[default_colors[ii] for ii  in [0,1,2]]
                        )
    #plot_filters()
    pl.xlim(1.11, 5.10);
    pl.ylim(1e-22, 6e-18);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_full_range.png', dpi=150, bbox_inches='tight')

    pl.figure()
    plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines,  ocn, methanol, ethanol, water_ammonia))
    plot_filters(filternames=['F356W', 'F444W',])# 'F466N', 'F410M'])
    pl.xlim(3.00, 5.05);
    pl.savefig('/orange/adamginsburg/ice/colors_of_ices_overleaf/figures/opacities_on_f356_and_f444.png', dpi=150, bbox_inches='tight')

    ocn_mix1 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(1:1:1).ecsv')
    ocn_mix2 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(2:1:0.1).ecsv')
    ocn_mix3 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(1:1:0.02).ecsv')
    ocn_mix4 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(2:1:0.5).ecsv')
    ocn_mix5 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/CO:OCN_(1:1).ecsv')

    pl.figure()
    plot_opacity_tables(opacity_tables=(ocn_mix1, ocn_mix2, ocn_mix3, ocn_mix4, ocn_mix5,))
    plot_filters()
    pl.xlim(3.71, 4.75);
