from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import mpl_plot_templates
from molmass import Formula
import re

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath
from brick2221.analysis import plot_tools

from icemodels.core import composition_to_molweight

from dust_extinction.averages import CT06_MWGC, G21_MWAvg

if 'basetable_merged1182_daophot' not in globals():
    basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
    result = load_table(basetable_merged1182_daophot, ww=ww)
    ok2221 = result['ok2221']
    ok1182 = result['ok1182']
    globals().update(result)
    basetable = basetable_merged1182_daophot
    print("Loaded merged1182_daophot_basic_indivexp")


sel = ok = ok2221

dmag_h2o = Table.read(f'{basepath}/tables/H2O_ice_absorption_tables.ecsv')
dmag_co2 = Table.read(f'{basepath}/tables/CO2_ice_absorption_tables.ecsv')
dmag_co = Table.read(f'{basepath}/tables/CO_ice_absorption_tables.ecsv')
dmag_mine = Table.read(f'{basepath}/tables/H2O+CO_ice_absorption_tables.ecsv')
dmag_mine.add_index('mol_id')
dmag_mine.add_index('composition')
dmag_h2o.add_index('composition')
dmag_h2o.add_index('temperature')

color1= ['F182M', 'F212N']
color2= ['F410M', 'F466N']

def plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -2.5, 1]):
    plot_tools.ccd(basetable, ax=pl.gca(), color1=[x.lower() for x in color1], color2=[x.lower() for x in color2],
    s=1, sel=False,
                exclude=~ok2221)

    def wavelength_of_filter(filtername):
        return u.Quantity(int(filtername[1:-1])/100, u.um).to(u.um, u.spectral())

    E_V_color1 = (CT06_MWGC()(wavelength_of_filter(color1[0])) - CT06_MWGC()(wavelength_of_filter(color1[1])))
    E_V_color2 = (CT06_MWGC()(wavelength_of_filter(color2[0])) - CT06_MWGC()(wavelength_of_filter(color2[1])))
        
    abundance = 1e-4
    nh2_to_av = 2.21e21 #* u.cm**-2 / u.mag
    #nh2_to_av = 1.8e22 #* u.cm**-2 / u.mag # try the more extreme version...

    dcol = 2
    for mol_id in np.unique(dmag_mine['mol_id']):
        tb = dmag_mine.loc[mol_id]
        comp = np.unique(tb['composition'])[0]

        sel = tb['column'] < 5e19

        a_color1 = tb['column'][sel] / abundance / nh2_to_av * E_V_color1
        a_color2 = tb['column'][sel] / abundance / nh2_to_av * E_V_color2

        c1 = (tb[color1[0]][sel] if color1[0] in tb.colnames else 0) - (tb[color1[1]][sel] if color1[1] in tb.colnames else 0) + a_color1
        c2 = (tb[color2[0]][sel] if color2[0] in tb.colnames else 0) - (tb[color2[1]][sel] if color2[1] in tb.colnames else 0) + a_color2

        L, = pl.plot(c1, c2, label=comp, )
        #pl.scatter(tb['F410M'][sel][::dcol] - tb['F466N'][sel][::dcol], tb['F356W'][sel][::dcol] - tb['F444W'][sel][::dcol],
        #           s=(np.log10(tb['column'][sel][::dcol])-14.9)*20, c=L.get_color())
    pl.axis(axlims);

    return a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2

if __name__ == "__main__":
    color1= ['F182M', 'F212N']
    color2= ['F410M', 'F466N']
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2 = plot_ccd_with_icemodels(color1, color2)

    color1= ['F182M', 'F212N']
    color2= ['F212N', 'F466N']
    pl.figure()
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2 = plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -0.5, 4])