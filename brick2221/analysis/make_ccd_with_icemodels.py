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
from brick2221.analysis.make_icecolumn_fig9 import molscomps
from icemodels.core import composition_to_molweight

from dust_extinction.averages import CT06_MWGC, G21_MWAvg

from cycler import cycler
from tqdm.auto import tqdm

from brick2221.analysis.iceage_fluxes import iceage_flxd, iceage_mags

pl.rcParams['axes.prop_cycle'] = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'], ) * cycler(linestyle=['-', '--', ':', '-.'])

if 'basetable_merged1182_daophot' not in globals():
    basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
    result = load_table(basetable_merged1182_daophot, ww=ww)
    ok2221 = result['ok2221']
    ok1182 = result['ok1182'][ok2221]
    globals().update(result)
    basetable = basetable_merged1182_daophot[ok2221]
    # there are several bad data points in F182M that are brighter than 15.5 mag
    print("Loaded merged1182_daophot_basic_indivexp")

sel = ok = oksep_noJ[ok2221] & ~bad[ok2221] & (basetable['mag_ab_f182m'] > 15.5)


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
dmag_all = Table.read(f'{basepath}/tables/combined_ice_absorption_tables.ecsv')
dmag_all.add_index('mol_id')
dmag_all.add_index('composition')
dmag_all.add_index('temperature')
dmag_all.add_index('database')

x = np.linspace(1.24*u.um, 5*u.um, 1000)
pp_ct06 = np.polyfit(x, CT06_MWGC()(x), 7)

def ext(x, model=CT06_MWGC()): 
    if x > 1/model.x_range[1]*u.um and x < 1/model.x_range[0]*u.um:
        return model(x)
    else:
        return np.polyval(pp_ct06, x.value)

oxygen_abundance = 10**(9.3-12)
carbon_abundance = 10**(8.7-12)

percent_ice = 20

def plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -2.5, 1],
                            nh2_to_av=2.21e21,
                            abundance=(percent_ice/100)*carbon_abundance,
                            molids=np.unique(dmag_mine['mol_id']),
                            molcomps=None,
                            av_start=20,
                            max_column=2e20,
                            icemol='CO',
                            icemol2=None,
                            icemol2_col=None,
                            icemol2_abund=None,
                            ext=ext,
                            dmag_tbl=dmag_all,
                            temperature_id=0,
                            exclude=~ok,
                            iceage=True,
                            pure_ice_no_dust=False,
                            ):
    """
    """
    plot_tools.ccd(basetable, ax=pl.gca(), color1=[x.lower() for x in color1],
                   color2=[x.lower() for x in color2], s=1, sel=False,
                   ext=ext,
                   extvec_scale=30,
                   head_width=0.1,
                   exclude=exclude)

    if iceage:
        # av_iceage = 60
        # ext_iceage = G21_MWAvg()
        # av_iceage1 = ext_iceage(wavelength_of_filter(color1[0])) - ext_iceage(wavelength_of_filter(color1[1]))
        # av_iceage2 = ext_iceage(wavelength_of_filter(color2[0])) - ext_iceage(wavelength_of_filter(color2[1]))
        c1iceage = iceage_mags['JWST/NIRCam.'+color1[0]] - iceage_mags['JWST/NIRCam.'+color1[1]]
        c2iceage = iceage_mags['JWST/NIRCam.'+color2[0]] - iceage_mags['JWST/NIRCam.'+color2[1]]
        pl.scatter(c1iceage,
                   c2iceage,
                   s=25, c='r', marker='x')

    def wavelength_of_filter(filtername):
        return u.Quantity(int(filtername[1:-1])/100, u.um).to(u.um, u.spectral())

    E_V_color1 = (ext(wavelength_of_filter(color1[0])) - ext(wavelength_of_filter(color1[1])))
    E_V_color2 = (ext(wavelength_of_filter(color2[0])) - ext(wavelength_of_filter(color2[1])))
        
    if molcomps is not None:
        molids = np.unique(dmag_tbl.loc['composition', molcomps]['mol_id'])
        
    dcol = 2
    for mol_id in tqdm(molids):
        if isinstance(mol_id, tuple):
            mol_id, database = mol_id
            tb = dmag_tbl.loc[mol_id].loc['database', database]
        else:
            tb = dmag_tbl.loc[mol_id]
        comp = np.unique(tb['composition'])[0]
        temp = np.unique(tb['temperature'])[temperature_id]
        tb = tb.loc['temperature', temp]

        sel = tb['column'] < max_column

        try:
            molwt = u.Quantity(composition_to_molweight(comp), u.Da)
            mols, comps = molscomps(comp)
        except ValueError as ex:
            print(f'Error converting composition {comp} to molwt: {ex}')
            continue
        if icemol in mols:
            mol_frac = comps[mols.index(icemol)] / sum(comps)
        else:
            continue

        #mol_wt_tgtmol = Formula(icemol).mass * u.Da

        col = tb['column'][sel] * mol_frac #molwt * mol_massfrac / (mol_wt_tgtmol)
        # print(f'comp={comp}, mol_massfrac={mol_massfrac}, mol_wt_tgtmol={mol_wt_tgtmol}, molwt={molwt}, abundance={abundance}, col[0]={col[0]:0.1e}, col[-1]={col[-1]:0.1e}')

        h2col = col / abundance
        a_color1 = h2col / nh2_to_av * E_V_color1 + av_start * E_V_color1
        a_color2 = h2col / nh2_to_av * E_V_color2 + av_start * E_V_color2

        # DEBUG. print(f'color {color1[0]} in colnames: {color1[0] in tb.colnames}.  color {color1[1]} in colnames: {color1[1] in tb.colnames}.  color {color2[0]} in colnames: {color2[0] in tb.colnames}.  color {color2[1]} in colnames: {color2[1] in tb.colnames}.')
        c1 = (tb[color1[0]][sel] if color1[0] in tb.colnames else 0) - (tb[color1[1]][sel] if color1[1] in tb.colnames else 0) + a_color1 * (not pure_ice_no_dust)
        c2 = (tb[color2[0]][sel] if color2[0] in tb.colnames else 0) - (tb[color2[1]][sel] if color2[1] in tb.colnames else 0) + a_color2 * (not pure_ice_no_dust)

        #pl.scatter(tb['F410M'][sel][::dcol] - tb['F466N'][sel][::dcol], tb['F356W'][sel][::dcol] - tb['F444W'][sel][::dcol],
        #           s=(np.log10(tb['column'][sel][::dcol])-14.9)*20, c=L.get_color())

        if icemol2 is not None and icemol2 in mols and icemol2_col is not None:
            mol_frac2 = comps[mols.index(icemol2)] / sum(comps)
            # mol_wt_tgtmol2 = Formula(icemol2).mass * u.Da
            ind_icemol2 = np.argmin(np.abs(tb['column'][sel] * mol_frac2 - icemol2_col))
            #print(f'icemol2={icemol2}, icemol2_col={icemol2_col}, ind_icemol2={ind_icemol2} c1[ind_icemol2]={c1[ind_icemol2]}, c2[ind_icemol2]={c2[ind_icemol2]}')
            L, = pl.plot(c1, c2, label=f'{comp} (X$_{{{icemol2}}}$ = {icemol2_col / h2col[ind_icemol2]:0.1e})', )
            #L, = pl.plot(c1[ind_icemol2], c2[ind_icemol2], 'o', color=L.get_color())
        else:
            L, = pl.plot(c1, c2, label=comp, )

    pl.axis(axlims);

    return a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb

if __name__ == "__main__":

    color1= ['F182M', 'F212N']
    color2= ['F410M', 'F466N']
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, molcomps=['H2O:CO (0.5:1)',
 'H2O:CO (1:1)',
 'H2O:CO (3:1)',
 'H2O:CO (5:1)',
 'H2O:CO (7:1)',
 'H2O:CO (10:1)',
 'H2O:CO (15:1)',
 'H2O:CO (20:1)'],
                                                                                          dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                          axlims=[-0.1, 2.5, -2, 0.75],
                                                                                          icemol2='H2O', icemol2_col=1e19, abundance=(percent_ice/100)*carbon_abundance, icemol2_abund=(percent_ice/100)*oxygen_abundance, max_column=2e20)
    pl.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0, 0, 0))
    pl.title(f"{percent_ice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10^{{20}}$ cm$^{{-2}}$")# , dots show N(H$_2$O)=$10^{19}$ cm$^{-2}$")

    color1= ['F182M', 'F212N']
    color2= ['F212N', 'F466N']
    pl.figure()
    a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2, axlims=[-1, 4, -0.5, 4], molcomps=['H2O:CO (0.5:1)',
 'H2O:CO (1:1)',
 'H2O:CO (3:1)',
 'H2O:CO (5:1)',
 'H2O:CO (7:1)',
 'H2O:CO (10:1)',
 'H2O:CO (15:1)',
 'H2O:CO (20:1)'],
                                                                                          dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                          icemol2='H2O', icemol2_col=1e19, abundance=(percent_ice/100)*carbon_abundance, icemol2_abund=(percent_ice/100)*oxygen_abundance)
    pl.title(f"{percent_ice}% of CO in ice, N(CO)$_{{max}}$=$2\\times10^{{20}}$ cm$^{{-2}}$")# , dots show N(H$_2$O)=$10^{19}$ cm$^{-2}$")
    pl.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0, 0, 0))

    
    
    # Search for models that can explain the wide-band filters
    percent = 25

    for color1, color2, lims in ((['F182M', 'F212N'], ['F410M', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F182M', 'F212N'], ['F405N', 'F466N'], (0, 3, -1.5, 1.0)),
                                 (['F115W', 'F200W'], ['F356W', 'F444W'], (0, 20, -0.5, 1.5)),
                                 (['F356W', 'F410M'], ['F410M', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F356W', 'F405N'], ['F405N', 'F444W'], (-0.5, 2, -0.5, 0.5)),
                                 (['F356W', 'F466N'], ['F466N', 'F444W'], (-0.75, 1, -0.5, 1.5)),
                                 (['F182M', 'F212N'], ['F212N', 'F466N'], (0, 3, -0.1, 2.5)),
                                ):
        pl.figure();
        a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb = plot_ccd_with_icemodels(color1, color2,
                                                                                              molcomps=['H2O:CO (0.5:1)',
 'H2O:CO (1:1)',
 'H2O:CO (3:1)',
 'H2O:CO (5:1)',
 'H2O:CO (7:1)',
 'H2O:CO (10:1)',
 'H2O:CO:CO2 (1:1:10)',
 'H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)',
 'H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)',
 'H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)',
 'H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)',
 'H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)'],
                                                                                              dmag_tbl=dmag_all.loc['database', 'mymix'],
                                                                                              abundance=(percent/100.)*carbon_abundance,
                                                                                              max_column=2e20)
        pl.legend(loc='upper left', bbox_to_anchor=(1,1,0,0))
        pl.title(f"{percent}% of CO in ice");
        pl.axis(lims);
        pl.savefig(f'{basepath}/figures/CCD_with_icemodel_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}.png', bbox_inches='tight', dpi=150)