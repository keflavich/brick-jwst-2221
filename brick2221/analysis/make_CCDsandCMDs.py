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
from selections import main as selections_main

from analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
basetable_merged1182_daophot_bgsub_blur = Table.read(f'{basepath}/catalogs/basic_merged_photometry_tables_merged_bgsub_blur.fits')
result = selections_main(basetable_merged1182_daophot_bgsub_blur, ww=ww)
globals().update(result)
basetable = basetable_merged1182_daophot_bgsub_blur
print("Loaded merged1182_daophot_basic_bgsub_blur")

print()
print("merged-1182")
all_good = result['all_good']
any_good = result['any_good']
any_saturated = result['any_saturated']
long_good = result['long_good']
bad = result['bad']
exclude = result['exclude']
two_stars_in_same_pixel = result['two_stars_in_same_pixel']
globals().update({key+"_mr": val for key, val in result.items()})

colors=[('f410m', 'f466n'),
        ('f405n', 'f410m'),
        ('f405n', 'f466n'),
        ('f187n', 'f182m', ),
        ('f182m', 'f410m'),
        ('f182m', 'f212n', ),
        ('f187n', 'f405n'),
        ('f187n', 'f212n'),
        ('f212n', '410m405'),
        ('f212n', 'f410m'),
        ('182m187', '410m405'),
        ('f356w', 'f444w'),
        ('f356w', 'f410m'),
        ('f410m', 'f444w'),
        ('f405n', 'f444w'),
        ('f444w', 'f466n'),
        ('f200w', 'f356w'),
        ('f200w', 'f212n'),
        ('f182m', 'f200w'),
        ('f115w', 'f182m'),
        ('f115w', 'f212n'),
        ('f115w', 'f200w'),
    ]

fig = pl.figure()
combos = list(itertools.combinations(colors, 2))
extvec_scale = 100
ext = CT06_MWGC()
extvec_scale = 100
rasterized = True

sel = all_good & (~two_stars_in_same_pixel)

for ii, (color1, color2) in enumerate(combos):
    try:
        fig.clf()
        ax = fig.gca()
        ccd(basetable, ax=ax, color1=color1, color2=color2,
            axlims=(-1, 10, -1, 10) if 'f115w' in color1 or 'f115w' in color2 else (-1, 5, -1, 5),
            sel=sel,
            alpha=0.02,
            alpha_sel=0.02,
            exclude=exclude,
            max_uncertainty=0.05,
            rasterized=rasterized, ext=ext, extvec_scale=extvec_scale,)
        fig.savefig(f'{basepath}/ccds_cmds/ccd_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}.png')
        fig.savefig(f'{basepath}/ccds_cmds/ccd_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}.pdf')
    except Exception as ex:
        print(ex)

    try:
        fig.clf()
        cmds(basetable, colors=[color1],
             sel=sel,
             alpha=0.02,
             alpha_sel=0.02,
             fig=fig,
             exclude=exclude,
             max_uncertainty=0.05,
             axlims=(-2,5,26,15) if 'f115w' in color1 else (-2,5,22,12),
             xlim_percentiles=(0.1, 99.),
             rasterized=rasterized, ext=ext, extvec_scale=extvec_scale,)
        fig.savefig(f'{basepath}/ccds_cmds/cmd_{color1[0]}-{color1[1]}_{color1[0]}.png')
        fig.savefig(f'{basepath}/ccds_cmds/cmd_{color1[0]}-{color1[1]}_{color1[0]}.pdf')
    except Exception as ex:
        print(ex)

    pl.close('all')