from astropy.table import Table
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
import matplotlib as mpl
# from molmass import Formula
# from icemodels.core import composition_to_molweight
from dust_extinction.averages import CT06_MWGC  # , G21_MWAvg
import os
from icemodels.core import molscomps

# pl.rcParams['axes.prop_cycle']
propcycle = pl.cycler(
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
) * pl.cycler(linestyle=['-', '--', ':', '-.'])

x = np.linspace(1.24*u.um, 5*u.um, 1000)
# this allows extrapolation from CT06, which empirically looks OK - but should be used with caution!
pp_ct06 = np.polyfit(x, CT06_MWGC()(x), 7)


def compute_molecular_column(unextincted_1m2, dmag_tbl, icemol='CO', filter1='F410M', filter2='F466N',
                             maxcol=1e21, verbose=True):
    dmags1 = dmag_tbl[filter1]
    dmags2 = dmag_tbl[filter2]

    assert len(np.unique(dmag_tbl['composition'])) == 1, "dmag_tbl must have only one composition"
    comp = np.unique(dmag_tbl['composition'])[0]
    # molwt = u.Quantity(composition_to_molweight(comp), u.Da)
    mols, comps = molscomps(comp)
    mol_frac = comps[mols.index(icemol)] / sum(comps)

    cols_of_icemol = dmag_tbl['column'] * mol_frac  # molwt * mol_massfrac / (mol_wt_tgtmol)

    dmag_1m2 = np.array(dmags1) - np.array(dmags2)

    if verbose:
        print(f"min(dmag1) = {np.nanmin(dmags1)}, max(dmag1) = {np.nanmax(dmags1)}")
        print(f"min(unextincted_1m2) = {np.nanmin(unextincted_1m2)}, max(unextincted_1m2) = {np.nanmax(unextincted_1m2)}")

    sortorder = np.argsort(dmag_1m2)
    inferred_molecular_column = np.interp(unextincted_1m2,
                                          xp=dmag_1m2[sortorder][cols_of_icemol < maxcol],
                                          fp=cols_of_icemol[sortorder][cols_of_icemol < maxcol])

    return inferred_molecular_column


def compute_dmag_from_column(cols_of_icemol_observed, dmag_tbl, icemol='CO', filter1='F410M', filter2='F466N',
                             maxcol=1e21, verbose=True):

    # nan values in the dmag table mean zero effect on color
    dmags1 = np.nan_to_num(dmag_tbl[filter1])
    dmags2 = np.nan_to_num(dmag_tbl[filter2])

    assert len(np.unique(dmag_tbl['composition'])) == 1, "dmag_tbl must have only one composition"
    comp = np.unique(dmag_tbl['composition'])[0]
    # molwt = u.Quantity(composition_to_molweight(comp), u.Da)
    mols, comps = molscomps(comp)
    mol_frac = comps[mols.index(icemol)] / sum(comps)
    cols_of_icemol_theory = dmag_tbl['column'] * mol_frac

    dmag_1m2 = np.array(dmags1) - np.array(dmags2)

    sortorder = np.argsort(cols_of_icemol_theory)
    dmag_of_icemol = np.interp(cols_of_icemol_observed,
                               xp=cols_of_icemol_theory[sortorder][cols_of_icemol_theory < maxcol],
                               fp=dmag_1m2[sortorder][cols_of_icemol_theory < maxcol],
                               )

    if verbose:
        print(f"min(dmag1) = {np.nanmin(dmags1)}, max(dmag1) = {np.nanmax(dmags1)}")
        print(f"min(cols_of_icemol_theory) = {np.nanmin(cols_of_icemol_theory)}, max(cols_of_icemol_theory) = {np.nanmax(cols_of_icemol_theory)}")
        print(f"min(dmag_of_icemol) = {np.nanmin(dmag_of_icemol)}, max(dmag_of_icemol) = {np.nanmax(dmag_of_icemol)}")

    return dmag_of_icemol


def ext(x, model=CT06_MWGC()):
    if (x > 1/model.x_range[1]*u.um and
            x < 1/model.x_range[0]*u.um):
        return model(x)
    else:
        return np.polyval(pp_ct06, x.value)


@mpl.rc_context({'axes.prop_cycle': propcycle})
def plot_ccd_icemodels(color1, color2, dmag_tbl, molcomps=None, molids=None,
                       axlims=[-1, 4, -2.5, 1], nh_to_av=2.21e21,
                       abundance_wrt_h2=2e-5, av_start=20, max_column=2e20,
                       max_h2_column=None,
                       icemol='CO', icemol2=None, icemol2_col=None,
                       icemol2_abund=None, ext=ext, temperature_id=0,
                       label_author=False, label_temperature=False,
                       column_to_plot_point=None, pure_ice_no_dust=False,
                       verbose=False,
                       **kwargs):
    """
    Plot only the model tracks for given color combinations and ice compositions.

    abundance is with respect to H2.
    """
    def wavelength_of_filter(filtername):
        return u.Quantity(int(filtername[1:-1])/100, u.um).to(
            u.um, u.spectral())

    E_V_color1 = (ext(wavelength_of_filter(color1[0])) -
                  ext(wavelength_of_filter(color1[1])))
    E_V_color2 = (ext(wavelength_of_filter(color2[0])) -
                  ext(wavelength_of_filter(color2[1])))

    if molcomps is not None:
        if isinstance(molcomps[0][1], tuple):
            molids = [np.unique(dmag_tbl
                                .loc['author', author]
                                .loc['composition', mc]
                                .loc['temperature', float(tem)]['mol_id'])
                      for (author, (mc, tem)) in molcomps]
            molcomps = [xx[1] for xx in molcomps]
        else:
            molids = [np.unique(dmag_tbl
                                .loc['composition', mc]
                                .loc['temperature', float(tem)]['mol_id'])
                      for mc, tem in molcomps]
    else:
        molcomps = np.unique(dmag_tbl.loc['mol_id', molids]['composition'])

    assert len(molcomps) == len(molids)
    assert len(molcomps) > 0

    if max_h2_column is not None:
        if max_column is not None:
            raise ValueError("max_column and max_h2_column cannot both be set")
        max_column = max_h2_column * abundance_wrt_h2

    dcol = 2
    for mol_id, (molcomp, temperature) in (zip(molids, molcomps)):
        if isinstance(mol_id, tuple):
            mol_id, database = mol_id
            tb = dmag_tbl.loc['mol_id', mol_id].loc['database', database].loc['composition', molcomp]
        else:
            tb = dmag_tbl.loc['mol_id', mol_id].loc['composition', molcomp]
        comp = np.unique(tb['composition'])[0]
        temp = np.unique(tb['temperature'])[temperature_id]
        author = np.unique(tb['author'])[0]
        tb = tb.loc['temperature', float(temp)]

        try:
            # molwt = u.Quantity(composition_to_molweight(comp), u.Da)
            from icemodels.core import molscomps
            mols, comps = molscomps(comp)
        except Exception as ex:
            print(f'Error converting composition {comp} to molwt: {ex}')
            continue
        if icemol in mols:
            mol_frac = comps[mols.index(icemol)] / sum(comps)
        else:
            print(f"icemol {icemol} not in {mols} for {comp}.  tb.meta={tb.meta}")
            continue

        icemol_col = np.geomspace(1e17, max_column, 50)
        sel = icemol_col <= max_column
        h2col = icemol_col / abundance_wrt_h2

        dmag_of_icemol_color1 = compute_dmag_from_column(icemol_col, tb, icemol=icemol, maxcol=max_column, filter1=color1[0], filter2=color1[1], verbose=verbose)
        dmag_of_icemol_color2 = compute_dmag_from_column(icemol_col, tb, icemol=icemol, maxcol=max_column, filter1=color2[0], filter2=color2[1], verbose=verbose)

        # a_colors are the extinction colors
        a_color1 = h2col * 2 / nh_to_av * E_V_color1 + av_start * E_V_color1
        a_color2 = h2col * 2 / nh_to_av * E_V_color2 + av_start * E_V_color2

        # nan_to_num used here because nans are returned if there is no overlap with the filter
        # and in that case, the effective color (dmag) is really zero
        c1 = dmag_of_icemol_color1 + a_color1 * (not pure_ice_no_dust)
        c2 = dmag_of_icemol_color2 + a_color2 * (not pure_ice_no_dust)
        # c1 = ((np.nan_to_num(tb[color1[0]][sel]) if color1[0] in tb.colnames else 0) -
        #       (np.nan_to_num(tb[color1[1]][sel]) if color1[1] in tb.colnames else 0) +
        #       a_color1 * (not pure_ice_no_dust))
        # c2 = ((np.nan_to_num(tb[color2[0]][sel]) if color2[0] in tb.colnames else 0) -
        #       (np.nan_to_num(tb[color2[1]][sel]) if color2[1] in tb.colnames else 0) +
        #       a_color2 * (not pure_ice_no_dust))
        assert not np.any(np.isnan(c1))
        assert not np.any(np.isnan(c2))

        if icemol2 is not None and icemol2 in mols and icemol2_col is not None:
            raise NotImplementedError("icemol2 not implemented correctly / I don't know what I was going for")
            mol_frac2 = comps[mols.index(icemol2)] / sum(comps)
            ind_icemol2 = np.argmin(np.abs(tb['column'][sel] * mol_frac2 - icemol2_col))
            L, = pl.plot(c1, c2, label=f'{comp} (X$_{{{icemol2}}}$ = {icemol2_col / h2col[ind_icemol2]:0.1e})', **kwargs)
        else:
            label = comp
            if label_author:
                label = label + f' {author}'
            if label_temperature:
                label = label + f' {temp}'
            L, = pl.plot(c1, c2, label=label, **kwargs)

        if column_to_plot_point is not None:
            sel2 = np.argmin(np.abs(tb['column'] - column_to_plot_point))
            pl.plot(c1[sel2], c2[sel2], 'o', color='black', markersize=5)

    pl.axis(axlims)
    pl.xlabel(f"{color1[0]} - {color1[1]}")
    pl.ylabel(f"{color2[0]} - {color2[1]}")
    return a_color1, a_color2, c1, c2, sel, E_V_color1, E_V_color2, tb


# Constants for abundances and percent ice
carbon_abundance = 10**(8.7-12)  # = 1e-3.3 = 5e-4
oxygen_abundance = 10**(9.3-12)
percent_ice = 25  # can be changed per plot if needed

# Example plot configurations
example_plots = [
    # Simple CO/H2O mixes
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F410M', 'F466N'],
        'axlims': (0, 3, -1.5, 1.0),
        'molcomps': [
            ('H2O:CO (0.5:1)', 25.0),
            ('H2O:CO (1:1)', 25.0),
            ('H2O:CO (3:1)', 25.0),
            ('H2O:CO (5:1)', 25.0),
            ('H2O:CO (7:1)', 25.0),
            ('H2O:CO (10:1)', 25.0),
            ('H2O:CO (15:1)', 25.0),
            ('H2O:CO (20:1)', 25.0),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F410M-F466N_nodata.png',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
        'axlims': (0, 3, -1.5, 1.0),
        'molcomps': [
            ('H2O:CO (0.5:1)', 25.0),
            ('H2O:CO (1:1)', 25.0),
            ('H2O:CO (3:1)', 25.0),
            ('H2O:CO (5:1)', 25.0),
            ('H2O:CO (7:1)', 25.0),
            ('H2O:CO (10:1)', 25.0),
            ('H2O:CO (15:1)', 25.0),
            ('H2O:CO (20:1)', 25.0),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_nodata.png',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
        'axlims': (0, 4, -1.5, 1.0),
        'molcomps': [
            ('H2O:CO:CO2 (1:1:1)', 25.0),
            ('H2O:CO:CO2 (3:1:1)', 25.0),
            ('H2O:CO:CO2 (5:1:1)', 25.0),
            ('H2O:CO:CO2 (10:1:1)', 25.0),
            ('H2O:CO:CO2 (15:1:1)', 25.0),
            ('H2O:CO:CO2 (20:1:1)', 25.0),
            ('H2O:CO2:CO 72:25:2.7', -999),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 5e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_H2OCOCO2_nodata.png',
    },
    {
        # This one is totally pointless - it's just a vertical line
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.3, 0.3, -2.5, 1.0),
        'molcomps': [
            ('H2O:CO:CO2 (1:1:1)', 25.0),
            ('H2O:CO:CO2 (3:1:1)', 25.0),
            ('H2O:CO:CO2 (5:1:1)', 25.0),
            ('H2O:CO:CO2 (10:1:1)', 25.0),
            ('H2O:CO:CO2 (15:1:1)', 25.0),
            ('H2O:CO:CO2 (20:1:1)', 25.0),
            ('H2O:CO2:CO 72:25:2.7', -999),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'pure_ice_no_dust': True,
        'column_to_plot_point': 1e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_H2OCOCO2_pureicenodust_nodata.png',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F466N'],
        'axlims': (-0.3, 0.3, -2.5, 1.0),
        'molcomps': [
            ('H2O:CO2:CO 72:25:2.7', -999),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': 5.4e-6,
        'max_column': 2e20,
        'pure_ice_no_dust': True,
        'column_to_plot_point': 1e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F466N_H2OCOCO2_pureicenodust_nodata_kp5.png',
    },
    # CO/H2O/CO2/CH3OH/CH3CH2OH mixes
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F466N', 'F480M'],
        'axlims': (-0.2, 10, -1, 2.5),
        'molcomps': [
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:1:0.1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:0.1:0.1:0.1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.1:1:0.1:1:0.1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:1:0.1:0.1:1)', 25.0),
            ('H2O:CO:CO2:CH3OH:CH3CH2OH (0.01:0.1:0.1:0.1:1)', 25.0),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e20,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F466N-F480M_mixes_nodata.png',
    },
    # OCN mixes
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F410M', 'F466N'],
        'axlims': (0, 3, -1.5, 1.0),
        'molcomps': [
            ('CO:OCN (1:1)', 25.0),
            ('H2O:CO:OCN (1:1:1)', 25.0),
            ('H2O:CO:OCN (1:1:0.02)', 25.0),
            ('H2O:CO:OCN (2:1:0.1)', 25.0),
            ('H2O:CO:OCN (2:1:0.5)', 25.0),
        ],
        'icemol': 'CO',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 5e19,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 5e19 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F410M-F466N_OCNmixes_nodata.png',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F410M'],
        'axlims': (-0.1, 2.5, -0.4, 0.15),
        'molcomps': [
            ('Hudgins', ('CO2 (1)', 70)),
            ('Gerakines', ('CO2 (1)', 70)),
            ('Hudgins', ('CO2 (1)', 10)),
            ('Ehrenfreund', ('CO2 (1)', 10)),
            ('Hudgins', ('CO2 (1)', 30)),
            ('Hudgins', ('CO2 (1)', 50)),
            ('Ehrenfreund', ('CO2 (1)', 50)),
            ('Gerakines', ('CO2 (1)', 8)),
            # ('Mastrapa 2024, Gerakines 2020, etc', ('H2O:CO:CO2 (1:1:1)', 25.0)),
            # ('Mastrapa 2024, Gerakines 2020, etc', ('H2O:CO:CO2:CH3OH (1:1:1:1)', 25.0)),
            # ('Mastrapa 2024, Gerakines 2020, etc', ('H2O:CO:CO2:CH3OH:CH3CH2OH (1:1:1:1:1)', 25.0)),
        ],
        'icemol': 'CO2',
        'abundance_wrt_h2': (percent_ice/100.)*carbon_abundance,
        'max_column': 2e19,
        'av_start': 0,
        'column_to_plot_point': 1e18,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 2e19 cm$^{{-2}}$, $N(\\bullet)=1e18 \\mathrm{{cm}}^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F410M_CO2only_nodata.png',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F410M'],
        'axlims': (-0.1, 2.5, -0.2, 0.15),
        'molcomps': [
            # ('Curtis', ('H2O (1)', '146K')),
            ('Bertie', ('H2O (1)', 100)),
            ('Mastrapa', ('H2O (1)', 100)),
            ('Kitta', ('H2O (1)', 23)),
            ('Mastrapa', ('H2O (1)', 50)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 10)),
            ('Léger', ('H2O (1)', 77)),
            ('Mastrapa', ('H2O (1)', 20)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': (percent_ice/100.)*oxygen_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 1e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F410M_H2Oonly_nodata.png',
    },
    {
        'color1': ['F182M', 'F212N'],
        'color2': ['F405N', 'F410M'],
        'axlims': (-0.1, 2.5, -0.2, 0.15),
        'molcomps': [
            # ('Curtis', ('H2O (1)', '146K')),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:0.6:1)", 180.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:1:1)", 80.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (9:1:2)", 30.0)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 10)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': (percent_ice/100.)*oxygen_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 1e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F182M-F212N_F405N-F410M_H2OandMethanolonly_nodata.png',
    },
    {
        'color1': ['F200W', 'F356W'],
        'color2': ['F356W', 'F444W'],
        'axlims': (-1, 4, -0.5, 1.5),
        'molcomps': [
            # ('Curtis', ('H2O (1)', '146K')),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:0.6:1)", 180.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (1:1:1)", 80.0)),
            ('Ehrenfreund et al.', ("H2O:CH3OH:CO2 (9:1:2)", 30.0)),
            ('Hudgins', ('H2O (1)', 80)),
            ('Hudgins', ('H2O (1)', 10)),
        ],
        'icemol': 'H2O',
        'abundance_wrt_h2': (percent_ice/100.)*oxygen_abundance,
        'max_column': 1e20,
        'av_start': 0,
        'label_author': True,
        'label_temperature': True,
        'title': f"{percent_ice}% of C in ice, $N_{{max}}$ = 1e20 cm$^{{-2}}$",
        'filename': 'CCD_icemodel_F200W-F356W_F356W-F444W_H2OandMethanolonly_nodata.png',
    }
    # Add more plot configs as needed...
]

if __name__ == "__main__":
    """
    The "main" example is intended to be run in the Brick 2221 project's directory.
    """

    with mpl.rc_context({'axes.prop_cycle': propcycle}):
        savefig_path = '/orange/adamginsburg/jwst/brick/figures/'

        basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        dmag_tbl = dmag_all = Table.read(os.path.join(basepath, 'icemodels', 'data', 'combined_ice_absorption_tables.ecsv'))
        dmag_all.add_index('mol_id')
        dmag_all.add_index('composition')
        dmag_all.add_index('temperature')
        dmag_all.add_index('database')
        dmag_tbl.add_index('author')

        for plot_cfg in example_plots:
            if dmag_tbl is None:
                raise ValueError("dmag_tbl not loaded. Please load your model table in the __main__ block.")
            pl.figure()
            plot_ccd_icemodels(
                color1=plot_cfg['color1'],
                color2=plot_cfg['color2'],
                dmag_tbl=dmag_tbl,
                molcomps=plot_cfg['molcomps'],
                axlims=plot_cfg['axlims'],
                abundance_wrt_h2=plot_cfg['abundance_wrt_h2'],
                max_column=plot_cfg['max_column'],
                icemol=plot_cfg['icemol'],
                label_author=plot_cfg.get('label_author', False),
                label_temperature=plot_cfg.get('label_temperature', False),
                av_start=plot_cfg.get('av_start', 0),
                column_to_plot_point=plot_cfg.get('column_to_plot_point', None),
                pure_ice_no_dust=plot_cfg.get('pure_ice_no_dust', False),
                **plot_cfg.get('kwargs', {})
            )
            pl.legend(loc='upper left', bbox_to_anchor=(1, 1, 0, 0))
            pl.title(plot_cfg['title'])
            pl.savefig(os.path.join(savefig_path, plot_cfg['filename']),
                       bbox_inches='tight', dpi=150)
            pl.close()
