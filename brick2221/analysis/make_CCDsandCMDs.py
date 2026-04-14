import pylab as pl
import itertools

import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.visualization import simple_norm

import analysis_setup
import selections
import plot_tools
from plot_tools import plot_extvec_ccd, ccd, ccds, cmds, cmds_withiso, ccds_withiso

from dust_extinction.averages import RRP89_MWGC, CT06_MWGC, F11_MWGC
from dust_extinction.parameter_averages import CCM89

from analysis_setup import filternames, basepath, img_nostars as img
from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from selections import load_table, make_downselected_table_20250721, make_downselected_table_20251211

from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
#basetable_merged1182_indivexp = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
# basetable = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20250721.fits')
#result = selections_main(basetable_merged1182_indivexp, ww=ww)
#result = load_table(basetable_merged1182_indivexp, ww=ww)
# result = load_table(basetable, ww=ww)

# This version was used in the "Colors of Ices" paper
#basetable = make_downselected_table_20250721()

# This version should, in theory, supersede the above
# I'm leaving it commented out right now b/c the Colors of Ices paper is under review
basetable = make_downselected_table_20251211()
result = load_table(basetable, ww=ww)

pl.rcParams['figure.dpi'] = 150

globals().update(result)
#basetable = basetable_merged1182_indivexp
#print("Loaded merged1182_daophot_indivexp", flush=True)
print("Loaded merged1182_daophot_indivexp_ok2221or1182 from 2025-07-21", flush=True)

print()
print("merged-1182", flush=True)
all_good = result['all_good']
any_good = result['any_good']
any_saturated = result['any_saturated']
long_good = result['long_good']
bad = result['bad']
exclude = result['exclude']
two_stars_in_same_pixel = result['two_stars_in_same_pixel'] if 'two_stars_in_same_pixel' in result else False
globals().update({key+"_mr": val for key, val in result.items()})

colors=[('f410m', 'f466n'),
        ('f405n', 'f410m'),
        ('f405n', 'f466n'),
        ('f187n', 'f182m', ),
        ('f182m', 'f410m'),
        ('f182m', 'f212n', ),
        ('f187n', 'f405n'),
        ('f187n', 'f212n'),
        #('f212n', '410m405'),
        ('f212n', 'f410m'),
        #('182m187', '410m405'),
        ('f356w', 'f444w'),
        ('f200w', 'f444w'),
        ('f356w', 'f410m'),
        ('f356w', 'f405n'),
        ('f410m', 'f444w'),
        ('f405n', 'f444w'),
        ('f405n', 'f466n'),
        ('f444w', 'f466n'),
        ('f200w', 'f356w'),
        ('f200w', 'f212n'),
        ('f182m', 'f200w'),
        ('f115w', 'f182m'),
        ('f115w', 'f212n'),
        ('f115w', 'f200w'),
    ]

axlim_dict = {
    ('f405n', 'f410m'): (-0.6, 0.05),
    ('f356w', 'f444w'): (-0.3, 3),
    ('f356w', 'f410m'): (-0.1, 3.5),
    ('f356w', 'f405n'): (-0.5, 1.5),
    ('f212n', 'f410m'): (-0.1, 5.0),
    ('f182m', 'f200w'): (-0.1, 1.1),
    ('f182m', 'f212n'): (-0.1, 3.0),
    ('f187n', 'f182m'): (-0.5, 0.2),
    ('f187n', 'f212n'): (-0.2, 2.0),
    ('f200w', 'f356w'): (0, 3),
    ('f410m', 'f444w'): (-0.5, 0.5),
    ('f410m', 'f466n'): (-2, 0.5),
    ('f200w', 'f212n'): (-0.1, 1.0),
    ('f200w', 'f444w'): (-0.1, 5.0),
    ('f182m', 'f410m'): (-0.3, 5),
}
for key, val in list(axlim_dict.items()):
    axlim_dict[(key[1], key[0])] = (val[1], val[0])
axlim_dict['f356w'] = (19, 13)
axlim_dict['f410m'] = (19, 13)
axlim_dict['f405n'] = (19, 13)
axlim_dict['f444w'] = (19, 13)
axlim_dict['f115w'] = (26, 18)
axlim_dict['f182m'] = (22, 14)
axlim_dict['f187n'] = (20, 14)
axlim_dict['f200w'] = (22, 14)
axlim_dict['f212n'] = (20, 14)

axlim_dict_lowext = {
    ('f356w', 'f405n'): (-0.2, 0.7),
    ('f356w', 'f410m'): (-0.2, 0.7),
    ('f356w', 'f410m'): (-0.1, 1.0),
    ('f356w', 'f444w'): (-0.3, 1),
    ('f405n', 'f466n'): (-0.5, 0.5),
    ('f405n', 'f444w'): (-0.5, 0.5),
    ('f410m', 'f466n'): (-0.5, 0.5),
    ('f444w', 'f466n'): (-0.5, 0.5),
    ('f410m', 'f466n'): (-0.5, 0.5),
}
axlim_dict_dereddened = {
    ('f356w', 'f405n'): (-0.2, 0.7),
    ('f356w', 'f410m'): (-0.2, 0.7),
    ('f356w', 'f410m'): (-0.1, 0.6),
    ('f356w', 'f444w'): (-0.3, 0.6),
    ('f405n', 'f466n'): (-0.5, 0.5),
    ('f405n', 'f444w'): (-0.5, 0.5),
    ('f410m', 'f466n'): (-0.5, 0.5),
    ('f444w', 'f466n'): (-0.5, 0.5),
    ('f200w', 'f444w'): (-0.7, 0.7),
    'f356w': (19, 12),
    'f410m': (19, 12),
    'f405n': (19, 12),
    'f444w': (19, 12),
    'f115w': (26, 17),
    'f182m': (20, 11),
    'f187n': (19, 11),
    'f200w': (19, 11),
    'f212n': (19, 11),
}

fig = pl.figure()
combos = list(itertools.combinations(colors, 2))
extvec_scale = 100
ext = CT06_MWGC()
extvec_scale = 100
rasterized = True

#sel = all_good & (~two_stars_in_same_pixel)
# TEMP plot
bluesel = basetable['mag_ab_f405n'] - basetable['mag_ab_f410m'] < -0.2
#print(f"Found {sel.sum()} stars with blue405n410m")
#suffix = '_blue405n410m'
suffix = ''


import warnings
warnings.simplefilter('ignore')


for color1 in colors:

    fig.clf()
    axlims = (-2,5,26,15) if 'f115w' in color1 else (-2,5,22,12)
    if tuple(color1) in axlim_dict:
        axlims = axlim_dict[tuple(color1)] + axlims[2:]
    if color1[0] in axlim_dict:
        print(f"color1[0]={color1[0]} in axlim_dict, replacing {axlims[:2]} with {axlim_dict[(color1[0])]}")
        axlims = axlims[:2] + axlim_dict[color1[0]]
    cmds(basetable, colors=[color1],
            sel=bluesel,
            alpha=0.02,
            alpha_sel=0.02,
            fig=fig,
            exclude=exclude,
            ext=ext if 'f115w' not in color1 else None,
            max_uncertainty=0.05,
            axlims=axlims,
            xlim_percentiles=(0.1, 99.),
            rasterized=rasterized, 
            extvec_scale=extvec_scale,)
    fig.savefig(f'{basepath}/ccds_cmds/cmd_{color1[0]}-{color1[1]}_{color1[0]}{suffix}.png')
    fig.savefig(f'{basepath}/ccds_cmds/cmd_{color1[0]}-{color1[1]}_{color1[0]}{suffix}.pdf')


    if color1[0][3] != 'm' and color1[1][3] != 'm':
        fig.clf()
        axlims = (-2,5,26,13) if 'f115w' in color1 else (-2,5,22,13)
        if tuple(color1) in axlim_dict:
            axlims = axlim_dict[tuple(color1)] + axlims[2:]
        if color1[0] in axlim_dict:
            axlims = axlims[:2] + axlim_dict[color1[0]]
        cmds_withiso(basetable, colors=[color1],
                sel=False,
                alpha_k=0.05,
                #alpha_sel=0.02,
                hexbin=False,
                fig=fig,
                exclude=exclude,
                #max_uncertainty=0.05,
                axlims=axlims,
                #xlim_percentiles=(0.1, 99.),
                rasterized=rasterized,
                markersize=2,
                ext=ext if 'f115w' not in color1 else None,
                extinction_scaling_av=extvec_scale,
                iso=(10,),
                distance_modulus=14.5,
                phase_transition_markers=True,
                #log_g_min=2.0,
                phase_include=[0, 2, 3, 4, 5],
                # extvec_scale=extvec_scale,
                )
        fig.savefig(f'{basepath}/ccds_cmds/cmd_iso_{color1[0]}-{color1[1]}_{color1[0]}{suffix}.png')
        fig.savefig(f'{basepath}/ccds_cmds/cmd_iso_{color1[0]}-{color1[1]}_{color1[0]}{suffix}.pdf')


        if tuple(color1) in axlim_dict_lowext:
            axlims = axlim_dict_lowext[tuple(color1)] + axlims[2:]
        if color1[0] in axlim_dict_lowext:
            axlims = axlims[:2] + axlim_dict_lowext[color1[0]]

        fig.clf()
        extcolor = basetable['mag_ab_f182m'] - basetable['mag_ab_f212n']
        extcolor2 = basetable['mag_ab_f200w'] - basetable['mag_ab_f444w']
        lowext_sel = ((extcolor > 0.5) & (extcolor < 0.8)) | ((extcolor2 > 1) & (extcolor2 < 2))
        cmds_withiso(basetable[lowext_sel], colors=[color1],
                sel=False,
                alpha_k=0.05,
                #alpha_sel=0.02,
                hexbin=False,
                fig=fig,
                exclude=exclude[lowext_sel],
                #max_uncertainty=0.05,
                axlims=axlims,
                #xlim_percentiles=(0.1, 99.),
                rasterized=rasterized,
                markersize=2,
                ext=ext if 'f115w' not in color1 else None,
                extinction_scaling_av=extvec_scale,
                iso=(10,),
                distance_modulus=14.5,
                phase_transition_markers=True,
                #log_g_min=2.0,
                phase_include=[0, 2, 3, 4, 5],
                # extvec_scale=extvec_scale,
                )
        fig.savefig(f'{basepath}/ccds_cmds/cmd_iso_{color1[0]}-{color1[1]}_{color1[0]}{suffix}_lowext.png')
        fig.savefig(f'{basepath}/ccds_cmds/cmd_iso_{color1[0]}-{color1[1]}_{color1[0]}{suffix}_lowext.pdf')


        if tuple(color1) in axlim_dict_dereddened:
            axlims = axlim_dict_dereddened[tuple(color1)] + axlims[2:]
        if color1[0] in axlim_dict_dereddened:
            axlims = axlims[:2] + axlim_dict_dereddened[color1[0]]

        fig.clf()
        av = extcolor / (ext(1.82*u.um) - ext(2.12*u.um))
        cmds_withiso(basetable[lowext_sel], colors=[color1],
                sel=False,
                alpha_k=0.05,
                #alpha_sel=0.02,
                hexbin=False,
                fig=fig,
                exclude=exclude[lowext_sel],
                #max_uncertainty=0.05,
                axlims=axlims,
                #xlim_percentiles=(0.1, 99.),
                rasterized=rasterized,
                markersize=4,
                ext=ext if 'f115w' not in color1 else None,
                extinction_scaling_av=extvec_scale,
                iso=(10,),
                distance_modulus=14.5,
                phase_transition_markers=True,
                #log_g_min=2.0,
                phase_include=[0, 2, 3, 4, 5],
                A_V=av[lowext_sel],
                # extvec_scale=extvec_scale,
                )
        fig.savefig(f'{basepath}/ccds_cmds/cmd_iso_{color1[0]}-{color1[1]}_{color1[0]}{suffix}_lowext_dereddened.png')
        fig.savefig(f'{basepath}/ccds_cmds/cmd_iso_{color1[0]}-{color1[1]}_{color1[0]}{suffix}_lowext_dereddened.pdf')

    pl.close('all')

for ii, (color1, color2) in enumerate(combos):
    print(f'{ii} out of {len(combos)}: {color1}, {color2}', flush=True)

    fig.clf()
    ax = fig.gca()
    axlims = (-1, 10, -1, 10) if 'f115w' in color1 or 'f115w' in color2 else (-1, 5, -1, 5)
    if tuple(color1) in axlim_dict:
        axlims = axlim_dict[tuple(color1)] + axlims[2:]
    if tuple(color2) in axlim_dict:
        axlims = axlims[:2] + axlim_dict[tuple(color2)]
    ccd(basetable, ax=ax, color1=color1, color2=color2,
        axlims=axlims,
        sel=bluesel,
        alpha=0.02,
        alpha_sel=0.02,
        exclude=exclude,
        max_uncertainty=0.05,
        rasterized=rasterized,
        ext=ext if 'f115w' not in color1 and 'f115w' not in color2 else None,
        extvec_scale=extvec_scale,)
    fig.savefig(f'{basepath}/ccds_cmds/ccd_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}{suffix}.png')
    fig.savefig(f'{basepath}/ccds_cmds/ccd_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}{suffix}.pdf')

    pl.close('all')