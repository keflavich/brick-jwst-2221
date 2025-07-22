import numpy as np
from dust_extinction.averages import CT06_MWGC
import matplotlib.pyplot as pl
from brick2221.analysis import plot_tools
from brick2221.analysis.analysis_setup import basepath
from brick2221.analysis.make_icecolumn_fig9 import calc_av
from astropy import units as u

def checkicystarsincommon_ccds(basetable):
    fig = pl.figure(figsize=(12, 12))
    ax = pl.subplot(2, 2, 1)

    ext = CT06_MWGC()
    avfilts = ['F182M', 'F212N']
    color_filts = ['F405N', 'F466N']
    filter1_wav = int(color_filts[0][1:-1])/100. * u.um
    filter2_wav = int(color_filts[1][1:-1])/100. * u.um
    E_V_color = (ext(filter2_wav) - ext(filter1_wav))
    av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext, return_av=True)
    measured_color_405_466 = basetable[f'mag_ab_{color_filts[0].lower()}'] - basetable[f'mag_ab_{color_filts[1].lower()}']
    unextincted_color_405_466 = measured_color_405_466 + E_V_color * av
    #sel = basetable['mag_ab_f356w'] - basetable['mag_ab_f444w'] > 0.4

    avfilts = ['F200W', 'F444W']
    color_filts = ['F356W', 'F444W']
    filter1_wav = int(color_filts[0][1:-1])/100. * u.um
    filter2_wav = int(color_filts[1][1:-1])/100. * u.um
    E_V_color = (ext(filter2_wav) - ext(filter1_wav))
    av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext, return_av=True)
    measured_color_356_444 = basetable[f'mag_ab_{color_filts[0].lower()}'] - basetable[f'mag_ab_{color_filts[1].lower()}']
    unextincted_color_356_444 = measured_color_356_444 + E_V_color * av
    #sel = basetable['mag_ab_f356w'] - basetable['mag_ab_f444w'] > 0.4


    sel_356_444 = unextincted_color_356_444 > 0.4
    color1 = ['F182M', 'F212N']
    color2 = ['F405N', 'F466N']

    print(f"Number of masked mags for 356-444 selected = F182m: {np.sum((basetable['mag_ab_f182m'][sel_356_444].mask))}")
    print(f"Number of masked mags for 356-444 selected = F212n: {np.sum((basetable['mag_ab_f212n'][sel_356_444].mask))}")
    print(f"Number of masked mags for 356-444 selected = F405N: {np.sum((basetable['mag_ab_f405n'][sel_356_444].mask))}")
    print(f"Number of masked mags for 356-444 selected = F466N: {np.sum((basetable['mag_ab_f466n'][sel_356_444].mask))}")

    plot_tools.ccd(basetable, ax=ax, color1=[x.lower() for x in color1],
                color2=[x.lower() for x in color2],
                sel=sel_356_444,
                markersize=2,
                ext=ext,
                extvec_scale=30,
                head_width=0.1,
                axlims=[-0.01, 3, -2.5, 0.5],
                alpha=0.25,
                hexbin=True,
                alpha_sel=0.25,
    )
    ax.set_title('(F356W - F444W > 0.4)')

    ax2 = pl.subplot(2, 2, 2)
    color1 = ['F200w', 'F444W']
    color2 = ['F356W', 'F444W']

    sel_405_466 = unextincted_color_405_466 < -0.4

    plot_tools.ccd(basetable, ax=ax2, color1=[x.lower() for x in color1],
                color2=[x.lower() for x in color2],
                sel=sel_405_466,
                markersize=2,
                ext=CT06_MWGC(),
                extvec_scale=30,
                head_width=0.1,
                axlims=[-0.01, 5, -0.5, 2.5],
                alpha=0.25,
                hexbin=True,
                alpha_sel=0.25,
    )
    ax2.set_title('(F405N - F466N < -0.4)')

    # intentionally swapped 3,4 so that wide-bands are on the right
    ax3 = pl.subplot(2, 2, 4)

    plot_tools.cmd(basetable=basetable, ax=ax3,
                   f1='f356w',
                   f2='f444w',
                   include=(~sel_356_444) & ((basetable['mag_ab_f405n'].mask) | (basetable['mag_ab_f466n'].mask)),
                   sel=None,
                   hexbin_cmap='Greens',
                   axlims=[-1, 3, 22, 12],
                   alpha=0.25,
                   #hexbin=True,
                   color='g',
                   zorder=5,
                   markersize=1,
    )
    plot_tools.cmd(basetable=basetable, ax=ax3,
                   f1='f356w',
                   f2='f444w',
                   sel=sel_405_466 & ~sel_356_444,
                   markersize=2,
                   ext=CT06_MWGC(),
                   extvec_scale=30,
                   head_width=0.1,
                   axlims=[-1, 3, 22, 12],
                   alpha=0.25,
                   hexbin=True,
                   alpha_sel=0.25,
                   zorder=1,
                   sel_zorder=20,
    )
    ax3.set_title('(F405N - F466N < -0.4) and not (F356W - F444W > 0.4)')

    ax4 = pl.subplot(2, 2, 3)

    plot_tools.cmd(basetable=basetable, ax=ax4,
                   f1='f405n',
                   f2='f466n',
                   include=(~sel_405_466) & ((basetable['mag_ab_f444w'].mask) | (basetable['mag_ab_f356w'].mask)),
                   sel=None,
                   hexbin_cmap='Greens',
                   axlims=[-1, 3, 22, 12],
                   alpha=0.25,
                   #hexbin=True,
                   color='g',
                   zorder=5,
                   markersize=1,
    )
    plot_tools.cmd(basetable=basetable, ax=ax4,
                   f1='f405n',
                   f2='f466n',
                   sel=(~sel_405_466) & sel_356_444,
                   markersize=2,
                   ext=CT06_MWGC(),
                   extvec_scale=30,
                   head_width=0.1,
                   axlims=[-1.5, 1, 20, 11],
                   alpha=0.25,
                   hexbin=True,
                   alpha_sel=0.5,
                   zorder=1,
                   sel_zorder=20,
    )
    ax4.set_title('(F356W - F444W > 0.4) and not (F405N - F466N < -0.4)')

    pl.savefig(f'{basepath}/figures/checkicystarsincommon_ccds.png', dpi=200, bbox_inches='tight')

    pl.figure(figsize=(12, 6))
    for skycoord in [basetable['skycoord_f410m'], basetable['skycoord_f200w']]:
        pl.scatter(
                skycoord.dec.deg[sel_405_466 | sel_356_444],
                skycoord.ra.deg[sel_405_466 | sel_356_444],
                c='k', s=2)
        pl.scatter(
                skycoord.dec.deg[sel_405_466 & sel_356_444],
                skycoord.ra.deg[sel_405_466 & sel_356_444],
                c='g', s=4)
        pl.scatter(
                skycoord.dec.deg[~(sel_405_466) & sel_356_444],
                skycoord.ra.deg[~(sel_405_466) & sel_356_444],
                #c=av[~(sel_405_466) & sel_200_444],
                c='r',
                marker='x',
                #cmap='Reds_r',
                alpha=0.5, s=2)
        pl.scatter(
                skycoord.dec.deg[(sel_405_466) & ~sel_356_444],
                skycoord.ra.deg[(sel_405_466) & ~sel_356_444],
                #c=av[(sel_405_466) & ~sel_200_444],
                c='b',
                marker='x',
                #cmap='Blues_r',
                alpha=0.5, s=2)
    pl.gca().set_aspect('equal')
    pl.savefig(f'{basepath}/figures/checkicystarsincommon_coords.png', dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    from brick2221.analysis.catalog_on_RGB import load_table
    basetable = load_table()

    # this masks out all F444W sources, I don't know why.
    # from brick2221.analysis.analysis_setup import field_edge_regions, ww410_merged
    # edge_sources = field_edge_regions.contains(basetable['skycoord_f410m'], ww410_merged)
    # edge_sources &= ~basetable['skycoord_f410m'].mask
    # basetable = basetable[~edge_sources]

    checkicystarsincommon_ccds(basetable)

    # sanity check
    sel = basetable['mag_ab_f356w'] > 18.5
    pl.clf()
    pl.scatter(basetable['mag_ab_f405n'][sel] - basetable['mag_ab_f466n'][sel], basetable['mag_ab_f405n'][sel], c='k', s=1)
    pl.xlabel('F405N - F466N')
    pl.ylabel('F405N')
    pl.savefig(f'{basepath}/figures/checkicystarsincommon_ccds_sanity.png', dpi=200, bbox_inches='tight')