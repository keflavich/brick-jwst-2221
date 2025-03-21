import numpy as np
from astropy.table import Table
import os
from astropy import units as u
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from astropy.visualization import simple_norm
from brick2221.analysis.plot_tools import plot_extvec_ccd
import glob
import pylab as pl
basepath = '/orange/adamginsburg/jwst/brick/'

from dust_extinction.averages import CT06_MWGC, G21_MWAvg


def make_av_comparison(basetable, color1, color2, color3, color4, color5, color6, color7, axlims1=(0.0, 2, -0.5, 4), axlims2=(0,3,-2.5,2), suffix=''):
    pl.figure(figsize=(20,10))
    ax = pl.subplot(2,1,1, adjustable='box', aspect=0.88)

    crds = basetable['skycoord_f410m']
    sel = basetable['sep_f405n'].quantity < 0.05*u.arcsec
    sel &= basetable['sep_f212n'].quantity < 0.05*u.arcsec
    #sel &= (~any_saturated) & (any_good)
    if 'nmatch_good_f410m' in basetable.colnames:
        sel &= basetable['nmatch_good_f410m'] > 2
        sel &= basetable['nmatch_good_f212n'] > 2

    if 'qfit_f410m' in basetable.colnames:
        # add qfit cuts - can be generous by requiring multiple bands
        sel &= basetable['qfit_f410m'] < 0.4
        sel &= basetable['qfit_f405n'] < 0.4
        sel &= basetable['qfit_f212n'] < 0.4
        sel466 = basetable['qfit_f466n'] < 0.4
    elif 'qf_f410m' in basetable.colnames:
        sel &= basetable['qf_f410m'] > 0.9
        sel &= basetable['qf_f405n'] > 0.9
        sel &= basetable['qf_f212n'] > 0.9
        sel466 = basetable['qf_f466n'] > 0.9

    #sel &= basetable['mag_ab_f410m'] < 18.5
    xx,yy = ww410.world_to_pixel(crds[sel])

    colorby = basetable[f'mag_ab_{color5[0]}'] - basetable[f'mag_ab_{color5[1]}']

    colornorm = simple_norm(colorby[sel], stretch='linear', vmin=-0.5, vmax=4)
    cmap = 'YlOrRd'

    scat = ax.scatter(
                    crds.dec[sel],
                    crds.ra[sel],
                    c=colorby[sel],
                    s=0.5,
                    alpha=0.75,
                    norm=colornorm,
                    cmap=cmap)

    pl.draw()
    colors = scat.get_facecolors()
    scat.set_edgecolors(colors)
    scat.set_facecolors('none')
    ax.axis()
    ax.set_ylabel("Right Ascension [ICRS]")
    ax.set_xlabel("Declination [ICRS]")

    ax2 = pl.subplot(2,3,5)
    sc = ax2.scatter((basetable[f'mag_ab_{color1[0]}'] - basetable[f'mag_ab_{color1[1]}'])[sel],
                    (basetable[f'mag_ab_{color2[0]}'] - basetable[f'mag_ab_{color2[1]}'])[sel],
                    s=0.5, alpha=0.75,
                    marker=',',
                    c=colorby[sel],
                    norm=colornorm, cmap=cmap
                    )
    # A_V=30 roughly corresponds to 1 mag color excess in 182-212
    # 1/(CT06_MWGC()(1.82*u.um) - CT06_MWGC()(2.12*u.um))
    plot_extvec_ccd(ax2, color1, color2, start=(1, 0,), color='k', extvec_scale=30, head_width=0.1)
    ax2.set_xlabel(f"[{color1[0].upper()}] - [{color1[1].upper()}]")
    ax2.set_ylabel(f"[{color2[0].upper()}] - [{color2[1].upper()}]");
    ax2.axis(axlims1);
    ax2.text(2, 1.5, "A$_V = 30$", ha='center')


    ax3 = pl.subplot(2,3,6)
    sc = ax3.scatter((basetable[f'mag_ab_{color3[0]}'] - basetable[f'mag_ab_{color3[1]}'])[sel],
                (basetable[f'mag_ab_{color4[0]}'] - basetable[f'mag_ab_{color4[1]}'])[sel],
                s=0.5, alpha=0.75,
                c=colorby[sel],
                norm=colornorm, cmap=cmap
            )
    plot_extvec_ccd(ax3, color3, color4, start=(0.5,1,), color='k', extvec_scale=30, head_width=0.1)


    ax3.set_xlabel(f"[{color3[0].upper()}] - [{color3[1].upper()}]")
    ax3.set_ylabel(f"[{color4[0].upper()}] - [{color4[1].upper()}]")
    ax3.axis(axlims2);
    ax3.text(1, 1.25, "A$_V = 30$", ha='center')

    pl.suptitle(os.path.splitext(os.path.basename(fn))[0])
    cb = pl.colorbar(mappable=sc, ax=pl.gcf().axes)
    cb.set_label(f"[{color5[0].upper()}] - [{color5[1].upper()}]")

    print(f"Selected {sel.sum()} stars for the colorcolor diagram plot.  sel466 has {sel466.sum()} stars. name={os.path.basename(fn)}")
    newname = os.path.basename(fn).replace('.fits', f'{suffix}_colorcolorcolor.png')
    pl.savefig(f"{basepath}/figures/{newname}", dpi=150, bbox_inches='tight')

    pl.figure(figsize=(15,5))
    for ii, color in enumerate([color1, color2, color3, color4, color5]):
        ax = pl.subplot(1,5,ii+1)
        pl.hist((basetable[f'mag_ab_{color[0]}'] - basetable[f'mag_ab_{color[1]}'])[sel], bins=np.linspace(axlims1[0], axlims1[1], 100))
        pl.hist((basetable[f'mag_ab_{color[0]}'] - basetable[f'mag_ab_{color[1]}'])[sel466], bins=np.linspace(axlims1[0], axlims1[1], 100), histtype='step', color='k')
        pl.xlabel(f"[{color[0].upper()}] - [{color[1].upper()}]")
    pl.tight_layout()
    pl.suptitle(os.path.splitext(os.path.basename(fn))[0])
    histname = os.path.basename(fn).replace('.fits', f'{suffix}_hist.png')
    pl.savefig(f"{basepath}/figures/{histname}", dpi=150, bbox_inches='tight')

    pl.figure()
    #colors = set(map(tuple, [color1, color2, color3, color4, color5, color6, color7, color8]))
    colors = [('f187n', 'f212n'), ('f187n', 'f405n'), ('f187n', 'f410m'), ('f187n', 'f466n'),
              ('f182m', 'f212n'), ('f182m', 'f405n'), ('f182m', 'f410m'), ('f182m', 'f466n'),
              ('f212n', 'f405n'), ('f212n', 'f410m'), ('f212n', 'f466n'),
              ('f410m', 'f466n'), ('f405n', 'f466n'),
    ]

    for ii, color in enumerate(colors):
        av_wavelengths = [int(avf[1:-1])/100. * u.um for avf in color]
        av = (basetable[f'mag_ab_{color[0]}'] - basetable[f'mag_ab_{color[1]}']) / (CT06_MWGC()(av_wavelengths[0]) - CT06_MWGC()(av_wavelengths[1]))
        av1 = np.array(av[sel])
        av1 = av1[np.isfinite(av1)]
        linestyle = '-'
        ax = pl.subplot(2, 1, 1)
        if 'f410m' in color:
            linestyle = '--'
        if 'f187n' in color:
            linestyle = ':'
        if 'f466n' in color:
            ax = pl.subplot(2, 1, 2)
        if len(av1) > 0:
            ax.hist(av1, bins=np.linspace(0 + np.random.randn(), 100 + np.random.randn(), 100),
                    label=f"[{color[0].upper()}] - [{color[1].upper()}]", histtype='step', linestyle=linestyle,
                    alpha=0.5 if linestyle == '-' else 1,
                    )


    color = ('f182m', 'f212n')
    av_wavelengths = [int(avf[1:-1])/100. * u.um for avf in color]
    av = (basetable[f'mag_ab_{color[0]}'] - basetable[f'mag_ab_{color[1]}']) / (CT06_MWGC()(av_wavelengths[0]) - CT06_MWGC()(av_wavelengths[1]))
    av1 = np.array(av[sel])
    av1 = av1[np.isfinite(av1)]
    pl.subplot(2, 1, 2).hist(av1, bins=np.linspace(0 + np.random.randn(), 100 + np.random.randn(), 100), label=f"[{color[0].upper()}] - [{color[1].upper()}]", color='k', alpha=0.25, zorder=-5)
        # av2 = np.array(av[sel466])
        # av2 = av2[np.isfinite(av2)]
        # if len(av2) > 0:
        #     ax.hist(av2, bins=np.linspace(0 + np.random.randn(), 100 + np.random.randn(), 100), label=f"[{color[0].upper()}] - [{color[1].upper()}]", alpha=0.25, zorder=-5)
    pl.subplot(2, 1, 1).set_xticks([])
    pl.subplot(2, 1, 1).set_xlim(-1, 101)
    pl.title(os.path.splitext(os.path.basename(fn))[0])
    pl.legend(loc='best')
    pl.subplot(2, 1, 2).set_xlabel("AV")
    pl.subplot(2, 1, 2).set_xlim(-1, 101)
    pl.legend(loc='best')
    pl.tight_layout()
    histname = os.path.basename(fn).replace('.fits', f'{suffix}_hist_av.png')
    pl.savefig(f"{basepath}/figures/{histname}", dpi=150, bbox_inches='tight')

if __name__ == "__main__":

    for fn in glob.glob(f'{basepath}/catalogs/basic*merged*fits') + glob.glob(f'{basepath}/catalogs/crowd*merged*fits') + glob.glob(f"{basepath}/catalogs/iter*merged*fits"):
        basetable = Table.read(fn)
        make_av_comparison(basetable, color1=['f182m', 'f212n'], color2=['f212n', 'f410m'], color3=['f182m', 'f212n'], color4=['f410m', 'f466n'], color5=['f187n', 'f405n'], color6=['f212n', 'f466n'], color7=['f182m', 'f410m'],  suffix='_410466_color187405')
        make_av_comparison(basetable, color1=['f182m', 'f212n'], color2=['f212n', 'f405n'], color3=['f182m', 'f212n'], color4=['f410m', 'f466n'], color5=['f182m', 'f410m'], color6=['f212n', 'f466n'], color7=['f187n', 'f405n'],  suffix='_410466')
        make_av_comparison(basetable, color1=['f182m', 'f212n'], color2=['f212n', 'f405n'], color3=['f182m', 'f212n'], color4=['f212n', 'f466n'], color5=['f182m', 'f410m'], color6=['f410m', 'f466n'], color7=['f187n', 'f405n'],  suffix='_212466', axlims2=(-0.1, 2.5, -0.1, 2.5), axlims1=(-0.1, 2.5, -0.1, 3.5))
        pl.close('all')