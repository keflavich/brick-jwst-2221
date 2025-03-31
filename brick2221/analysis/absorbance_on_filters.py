"""
Absorbance on Filters
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


# Load mix bases
water_mastrapa = read_ocdb_file(f'{optical_constants_cache_dir}/240_H2O_(1)_25K_Mastrapa.txt') # h2otbs[('ocdb', 242, 25)] 242 is 50K....
co2_gerakines = read_ocdb_file(f'{optical_constants_cache_dir}/55_CO2_(1)_8K_Gerakines.txt') # co2tbs[('ocdb', 55, 8)]
ethanol = read_lida_file(f'{optical_constants_cache_dir}/87_CH3CH2OH_1_30.0K.txt')
methanol = read_lida_file(f'{optical_constants_cache_dir}/58_CH3OH_1_25.0K.txt')
ocn = read_lida_file(f'{optical_constants_cache_dir}/158_OCN-_1_12.0K.txt')
co_gerakines = gerakines = retrieve_gerakines_co()


def plot_opacity_tables(opacity_tables=(co_gerakines, water_mastrapa, co2_gerakines, ethanol, methanol, ocn)):
    for tb in opacity_tables:

        molwt = u.Quantity(composition_to_molweight(tb.meta['composition']), u.Da)
        
        kk = tb['k₁'] if 'k₁' in tb.colnames else tb['k']
        opacity = ((kk.quantity * tb['Wavelength'].to(u.cm**-1, u.spectral()) * 4 * np.pi / (1*u.g/u.cm**3 / (molwt)))).to(u.cm**2)
        pl.plot(tb['Wavelength'], 
                opacity,
                label=f'{tb.meta["author"]} {tb.meta["composition"]} {tb.meta["temperature"]}'
                        if 'author' in tb.meta else
                    f'{tb.meta["index"]} {tb.meta["molecule"]} {tb.meta["ratio"]} {tb.meta["temperature"]}',
                linestyle='-')
    pl.legend(loc='lower left', bbox_to_anchor=(0, 1, 0, 0))
    pl.xlabel("Wavelength ($\\mu$m)")
    pl.ylabel("$\\kappa_{eff}$ [$\\tau = \\kappa_{eff} * N(ice)$]");
    pl.semilogy();
    pl.ylim(1e-21, 6e-18);

def plot_filters(filternames=['F466N', 'F410M'], ymax=6e-18):
    for filtername in filternames:
        wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filtername}')
        xarr = wavelength_table['Wavelength'].quantity.to(u.um)
        pl.plot(xarr, wavelength_table['Transmission']/wavelength_table['Transmission'].max() * ymax,
                color='k', linewidth=3, alpha=0.5, zorder=-5)


if __name__ == "__main__":
    pl.figure()
    plot_opacity_tables()
    plot_filters()
    pl.xlim(3.71, 4.75);

    pl.figure()
    plot_opacity_tables()
    plot_filters(filternames=['F356W', 'F444W', 'F466N', 'F410M'])
    pl.xlim(3.00, 5.05);

    ocn_mix1 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(1:1:1).ecsv')
    ocn_mix2 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(2:1:0.1).ecsv')
    ocn_mix3 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(1:1:0.02).ecsv')
    ocn_mix4 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/H2O:CO:OCN_(2:1:0.5).ecsv')
    ocn_mix5 = Table.read('/orange/adamginsburg/repos/icemodels/icemodels/data/mymixes/CO:OCN_(1:1).ecsv')

    pl.figure()
    plot_opacity_tables(opacity_tables=(ocn_mix1, ocn_mix2, ocn_mix3, ocn_mix4, ocn_mix5))
    plot_filters()
    pl.xlim(3.71, 4.75);