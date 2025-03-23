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

from cycler import cycler

pl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'], ) * cycler(linestyle=['-', '--', ':', '-.'])

if 'basetable_merged1182_daophot' not in globals():
    basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
    result = load_table(basetable_merged1182_daophot, ww=ww)
    ok2221 = result['ok2221']
    ok1182 = result['ok1182']
    globals().update(result)
    basetable = basetable_merged1182_daophot
    print("Loaded merged1182_daophot_basic_indivexp")


sel = ok = ok2221

dmag_co2 = Table.read(f'{basepath}/tables/CO2_ice_absorption_tables.ecsv')
dmag_co2.add_index('mol_id')
dmag_co2.add_index('composition')
dmag_co2.add_index('temperature')
dmag_co2.add_index('database')
dmag_co = Table.read(f'{basepath}/tables/CO_ice_absorption_tables.ecsv')
dmag_co.add_index('mol_id')
dmag_co.add_index('composition')
dmag_co.add_index('temperature')
dmag_co.add_index('database')
dmag_mine = Table.read(f'{basepath}/tables/H2O+CO_ice_absorption_tables.ecsv')
dmag_mine.add_index('mol_id')
dmag_mine.add_index('composition')
dmag_mine.add_index('temperature')
dmag_mine.add_index('database')
dmag_h2o = Table.read(f'{basepath}/tables/H2O_ice_absorption_tables.ecsv')
dmag_h2o.add_index('mol_id')
dmag_h2o.add_index('composition')
dmag_h2o.add_index('temperature')
dmag_h2o.add_index('database')

x = np.linspace(1.24*u.um, 5*u.um, 1000)
pp_ct06 = np.polyfit(x, CT06_MWGC()(x), 7)

def ext(x, model=CT06_MWGC()): 
    if x > 1/model.x_range[1]*u.um and x < 1/model.x_range[0]*u.um:
        return model(x)
    else:
        return np.polyval(pp_ct06, x.value)


def plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -2.5, 1], nh2_to_av=2.21e21, abundance=1e-4,
                            molids=np.unique(dmag_mine['mol_id']),
                            av_start=20,
                            max_column=5e20,
                            icemol='CO',
                            ext=ext,
                            dmag_tbl=dmag_mine,
                            temperature_id=0,
                            ):
    """
    """
    plot_tools.ccd(basetable, ax=pl.gca(), color1=[x.lower() for x in color1],
                   color2=[x.lower() for x in color2], s=1, sel=False,
                   ext=ext,
                   extvec_scale=30,
                   head_width=0.1,
                   exclude=~ok2221)

    def wavelength_of_filter(filtername):
        return u.Quantity(int(filtername[1:-1])/100, u.um).to(u.um, u.spectral())

    E_V_color1 = (ext(wavelength_of_filter(color1[0])) - ext(wavelength_of_filter(color1[1])))
    E_V_color2 = (ext(wavelength_of_filter(color2[0])) - ext(wavelength_of_filter(color2[1])))
        
    dcol = 2
    for mol_id in molids:
        if isinstance(mol_id, tuple):
            mol_id, database = mol_id
            tb = dmag_tbl.loc[mol_id].loc['database', database]
        else:
            tb = dmag_tbl.loc[mol_id]
        comp = np.unique(tb['composition'])[0]
        temp = np.unique(tb['temperature'])[temperature_id]
        tb = tb.loc['temperature', temp]

        sel = tb['column'] < max_column

        molwt = u.Quantity(composition_to_molweight(comp), u.Da)
        if len(comp.split(" ")) == 2:
            mols, comps = comp.split(" ")
            comps = list(map(float, re.split("[: ]", comps.strip("()"))))
            mols = re.split("[: ]", mols)
        elif len(comp.split(" (")) == 1:
            mols = [comp.split()[0]]
            comps = [1]
        else:
            mols, comps = comp.split(" (")
            comps = list(map(float, re.split("[: ]", comps.strip(")"))))
            mols = re.split("[: ]", mols)
        if icemol in mols:
            mol_massfrac = comps[mols.index(icemol)] / sum(comps)
        else:
            continue

        mol_wt_tgtmol = Formula(icemol).mass * u.Da

        col = tb['column'][sel] * molwt * mol_massfrac / (mol_wt_tgtmol)
        # print(f'comp={comp}, mol_massfrac={mol_massfrac}, mol_wt_tgtmol={mol_wt_tgtmol}, molwt={molwt}, abundance={abundance}, col[0]={col[0]:0.1e}, col[-1]={col[-1]:0.1e}')

        a_color1 = col / abundance / nh2_to_av * E_V_color1 + av_start * E_V_color1
        a_color2 = col / abundance / nh2_to_av * E_V_color2 + av_start * E_V_color2

        # DEBUG. print(f'color {color1[0]} in colnames: {color1[0] in tb.colnames}.  color {color1[1]} in colnames: {color1[1] in tb.colnames}.  color {color2[0]} in colnames: {color2[0] in tb.colnames}.  color {color2[1]} in colnames: {color2[1] in tb.colnames}.')
        c1 = (tb[color1[0]][sel] if color1[0] in tb.colnames else 0) - (tb[color1[1]][sel] if color1[1] in tb.colnames else 0) + a_color1
        c2 = (tb[color2[0]][sel] if color2[0] in tb.colnames else 0) - (tb[color2[1]][sel] if color2[1] in tb.colnames else 0) + a_color2

        L, = pl.plot(c1, c2, label=comp, )
        #pl.scatter(tb['F410M'][sel][::dcol] - tb['F466N'][sel][::dcol], tb['F356W'][sel][::dcol] - tb['F444W'][sel][::dcol],
        #           s=(np.log10(tb['column'][sel][::dcol])-14.9)*20, c=L.get_color())
    pl.axis(axlims);

    return a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb

if __name__ == "__main__":
    color1= ['F182M', 'F212N']
    color2= ['F410M', 'F466N']
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, molids=np.arange(8))
    pl.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0, 0, 0))

    color1= ['F182M', 'F212N']
    color2= ['F212N', 'F466N']
    pl.figure()
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -0.5, 4], molids=np.arange(8))
    pl.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0, 0, 0))