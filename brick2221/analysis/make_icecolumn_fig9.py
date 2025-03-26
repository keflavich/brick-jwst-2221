"""
Created 2025-03-19, long after publication
"""
from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import mpl_plot_templates
from molmass import Formula
import re
from astropy.io import fits

from brick2221.analysis.analysis_setup import filternames
from brick2221.analysis.selections import load_table
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from brick2221.analysis.analysis_setup import basepath

from icemodels.core import composition_to_molweight

from dust_extinction.averages import CT06_MWGC, G21_MWAvg


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
dmag_tbl = Table.read(f'{basepath}/tables/H2O+CO_ice_absorption_tables.ecsv')
dmag_tbl.add_index('composition')


def calc_av(avfilts=['F182M', 'F410M'], basetable=basetable, ext=CT06_MWGC()):
    av_wavelengths = [int(avf[1:-1])/100. * u.um for avf in avfilts]
    try:
        E_V = (ext(av_wavelengths[0]) - ext(av_wavelengths[1]))
    except ValueError:
        E_V = (G21_MWAvg()(av_wavelengths[0]) - ext(av_wavelengths[1]))

    av = (basetable[f'mag_ab_{avfilts[0].lower()}'] - basetable[f'mag_ab_{avfilts[1].lower()}']) / E_V

    return av


def molscomps(comp):
    if len(comp.split(" ")) == 2:
        mols, comps = comp.split(" ")
        comps = list(map(float, re.split("[: ]", comps.strip("()"))))
        mols = re.split("[: ]", mols)
    elif len(comp.split(" (")) == 1:
        mols = [comp]
        comps = [1]
    else:
        mols, comps = comp.split(" (")
        comps = list(map(float, re.split("[: ]", comps.strip(")"))))
        mols = re.split("[: ]", mols)

    return mols, comps

    
def compute_molecular_column(unextincted_466m410, av, dmag_tbl, basetable, ext, icemol='CO'):
    dmags466 = dmag_tbl['F466N']
    dmags410 = dmag_tbl['F410M']

    comp = np.unique(dmag_tbl['composition'])[0]
    molwt = u.Quantity(composition_to_molweight(comp), u.Da)
    mols, comps = molscomps(comp)
    mol_frac = comps[mols.index(icemol)] / sum(comps)

    #mol_wt_tgtmol = Formula(icemol).mass * u.Da
    #print(f'icemol={icemol}, molwt={molwt}, mol_wt_tgtmol={mol_wt_tgtmol}, comps={comps}, mols={mols}, massfrac={mol_massfrac}')

    cols = dmag_tbl['column'] * mol_frac #molwt * mol_massfrac / (mol_wt_tgtmol)

    dmag_466m410 = np.array(dmags466) - np.array(dmags410) 
    inferred_molecular_column = np.interp(unextincted_466m410, dmag_466m410[cols<1e21], cols[cols<1e21])

    return inferred_molecular_column


def makeplot(avfilts=['F182M', 'F410M'], 
             ax=None, sel=ok2221, ok=ok2221, alpha=0.5,
             icemol='CO',
             abundance=10**(8.7-12), # roughly extrapolated from Smartt 2001A%26A...367...86S
             title='H2O:CO (3:1)',
             dmag_tbl=dmag_tbl.loc['H2O:CO (3:1)'],
             plot_brandt=True,
             NtoAV=2.21e21,
             av_start=20,
             ):

    if ax is None:
        ax = pl.gca()

    ext = CT06_MWGC()
    av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext)

    E_V_410_466 = (ext(4.10*u.um) - ext(4.66*u.um))

    unextincted_466m410 = measured_466m410 + E_V_410_466 * av


    inferred_molecular_column = compute_molecular_column(unextincted_466m410=unextincted_466m410, av=av, dmag_tbl=dmag_tbl, basetable=basetable, ext=ext, icemol=icemol)

    fig = pl.gcf()
    pl.scatter(np.array(av[sel & ok]),
               np.log10(inferred_molecular_column[sel & ok]),
               marker='.', s=0.5, alpha=alpha)
    _,_,H,_,_,levels = mpl_plot_templates.adaptive_param_plot(np.array(av[sel & ok]),
                                    np.log10(inferred_molecular_column[sel & ok]),
                                    bins=50,
                                    threshold=15,
                                    #linewidths=[0.5]*5,
                                    cmap='Spectral_r',
                                    marker=',',
                                            )
    
    #pl.semilogy(av182b[selb], inferred_co_column_av182410b[selb], marker=',', linestyle='none')
    pl.xlim(-5, 105)
    pl.ylim(np.log10(2e15), np.log10(5e20))
    #pl.plot([10, 35], [1e17, 1e20], 'k--', label='log N = 0.12 A$_V$ + 15.8');
    # by-eye fit
    # x1,y1 = 33,8e16
    # x2,y2 = 80,3e19
    # m = (np.log10(y2) - np.log10(y1)) / (x2 - x1)
    # b = np.log10(y1 / 10**(m * x1))
    # pl.plot([x1, x2], np.array([x1*m+b, x2*m+b]), 'k--', label=f'log N = {m:0.2f} A$_V$ + {b:0.1f}')
    #pl.plot([7, 23], 10**(np.array([7,23]) * m + b))
    #pl.legend(loc='lower right')
    pl.xlabel(f"A$_V$ from {avfilts[0]}-{avfilts[1]} (mag)")
    pl.ylabel(f"log N({icemol} ice) [cm$^{{-2}}$]")
    pl.savefig(f"{basepath}/paper_co/figures/N{icemol}_vs_AV_{avfilts[0]}-{avfilts[1]}_contour_with1182.pdf", dpi=150, bbox_inches='tight')
    pl.savefig(f"{basepath}/paper_co/figures/N{icemol}_vs_AV_{avfilts[0]}-{avfilts[1]}_contour_with1182.png", dpi=250, bbox_inches='tight')
    
    #pl.plot([7, 23], np.log10([0.5e17, 7e17]), 'g', label='log N = 0.07 A$_V$ + 16.2 [BGW 2015]', linewidth=2)
    
    NMolofAV = NtoAV * np.linspace(0.1, 100, 1000) * 1e-4
    pl.plot(np.linspace(0.1, 100, 1000), np.log10(NMolofAV),
            label=f'100% of {icemol} in ice if N(H$_2$)=2.2$\\times10^{{21}}$ A$_V$', color='r', linestyle=':')
    
    pl.xlabel(f"A$_V$ from {avfilts[0]}-{avfilts[1]} (mag)")
    #pl.ylabel("N(CO) ice\nfrom Palumbo 2006 constants,\n4000K Phoenix atmosphere")
    pl.ylabel(f"log N({icemol} ice) [cm$^{{-2}}$]")
    pl.legend(loc='upper left')
    pl.title(title)
    pl.savefig(f"{basepath}/paper_co/figures/N{icemol}_vs_AV_{avfilts[0]}-{avfilts[1]}_contour_with1182.pdf", dpi=150, bbox_inches='tight')
    pl.savefig(f"{basepath}/paper_co/figures/N{icemol}_vs_AV_{avfilts[0]}-{avfilts[1]}_contour_with1182.png", dpi=250, bbox_inches='tight')

    if plot_brandt:
        plot_brandt_model(ax, molecule=icemol, nh_to_av=NtoAV, av_start=av_start)
        fig.savefig(f"{basepath}/paper_co/figures/N{icemol}_vs_AV_{avfilts[0]}-{avfilts[1]}_contour_with1182_Brandt.png", dpi=250, bbox_inches='tight')

    return av, inferred_molecular_column, ax

    
def plot_brandt_model(ax, nh_to_av=2.21e21*2, molecule='CO', av_start=0):
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

    av = column / nh_to_av + av_start

    nbins = 100
    hh, x_edges, y_edges = np.histogram2d(av[ok], np.log10(molcol[ok]),
                               bins=[np.linspace(lims[0], lims[1], nbins), np.linspace(lims[2], lims[3], nbins)])
    hh = hh.T
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    ax.contourf(x_centers, y_centers, hh, cmap='gray_r', zorder=-10)

    

def main():


    dmag_co = Table.read(f'{basepath}/tables/CO_ice_absorption_tables.ecsv')
    dmag_co.add_index('composition')
    dmag_co.add_index('temperature')
    dmag_co.add_index('mol_id')

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221,
             icemol='CO', abundance=1e-4,
             title='CO',
             dmag_tbl=dmag_co.loc['mol_id', 64].loc['composition', 'CO'])

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=1e-4,
             title='H2O:CO:CO2 (5:1:0.5)',
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (5:1:0.5)'])

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=10**-3.31,
             title='H2O:CO:CO2 (5:1:0.5)',
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (5:1:0.5)'])


    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=1e-4,
             title='H2O:CO (10:1)',
             dmag_tbl=dmag_tbl.loc['H2O:CO (10:1)'])
    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=10**-3.31,
             title='H2O:CO (10:1)',
             dmag_tbl=dmag_tbl.loc['H2O:CO (10:1)'])

    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=1e-4,
             title='H2O:CO:CO2 (10:1:2)',
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:2)'])
    av, inferred_molecular_column, ax = makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=10**-3.31,
             title='H2O:CO:CO2 (10:1:2)',
             dmag_tbl=dmag_tbl.loc['H2O:CO:CO2 (10:1:2)'])


    makeplot(avfilts=['F182M', 'F410M'], sel=ok2221, ok=ok2221, ax=pl.figure().gca())
    makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca())
    makeplot(avfilts=['F187N', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca())
    makeplot(avfilts=['F187N', 'F405N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca())
    makeplot(avfilts=['F182M', 'F200W'], sel=ok2221, ok=ok2221, ax=pl.figure().gca())
    makeplot(avfilts=['F115W', 'F200W'], sel=ok2221, ok=ok2221, ax=pl.figure().gca())
    makeplot(avfilts=['F115W', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca())

    dmag_h2o = Table.read(f'{basepath}/tables/H2O_ice_absorption_tables.ecsv')
    #dmag_co2 = Table.read(f'{basepath}/tables/CO2_ice_absorption_tables.ecsv')
    #dmag_mine = Table.read(f'{basepath}/tables/H2O+CO_ice_absorption_tables.ecsv')

    dmag_h2o.add_index('composition')
    dmag_h2o.add_index('temperature')
    dmag_h2o.add_index('mol_id')

    makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='H2O', abundance=10**(9.3-12), #abundance=4.89e-4,
             title='H2O',
             dmag_tbl=dmag_h2o.loc['H2O (1)'].loc['temperature', '25K'].loc['mol_id', 240])

             
    dmag_tbl_this = dmag_co.loc['mol_id', 36].loc['composition', 'CO CO2 (100 70)']
    makeplot(avfilts=['F182M', 'F212N'], sel=ok2221, ok=ok2221, ax=pl.figure().gca(),
             icemol='CO', abundance=1e-4,
             title=dmag_tbl_this['composition'][0],
             dmag_tbl=dmag_tbl_this)


if __name__ == "__main__":
    main()