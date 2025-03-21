from astropy.table import Table
import os
from astropy import units as u
from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
from astropy.visualization import simple_norm
from brick2221.analysis.plot_tools import plot_extvec_ccd
import glob
import pylab as pl
basepath = '/orange/adamginsburg/jwst/brick/'



def make_av_comparison(basetable, color1, color2, color3, color4, color5, axlims1=(0.0, 2, -0.5, 4), axlims2=(0,3,-2.5,2), suffix=''):
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
        sel &= basetable['qfit_f410m'] < 0.5
        sel &= basetable['qfit_f405n'] < 0.5
        sel &= basetable['qfit_f212n'] < 0.5

    #sel &= basetable['mag_ab_f410m'] < 18.5
    xx,yy = ww410.world_to_pixel(crds[sel])

    colorby = basetable[f'mag_ab_{color5[0]}'] - basetable[f'mag_ab_{color5[1]}']

    colornorm = simple_norm(colorby[sel], stretch='linear', min_cut=-0.5, max_cut=4)
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

    print(f"Selected {sel.sum()} stars for the colorcolor diagram plot")
    newname = os.path.basename(fn).replace('.fits', f'{suffix}_colorcolorcolor.png')
    pl.savefig(f"{basepath}/figures/{newname}", dpi=150, bbox_inches='tight')

if __name__ == "__main__":

    for fn in glob.glob(f'{basepath}/catalogs/basic*merged*fits') + glob.glob(f"{basepath}/catalogs/iter*merged*fits") + glob.glob(f'{basepath}/catalogs/crowd*merged*fits'):
        basetable = Table.read(fn)
        make_av_comparison(basetable, color1=['f182m', 'f212n'], color2=['f212n', 'f405n'], color3=['f182m', 'f212n'], color4=['f410m', 'f466n'], color5=['f182m', 'f410m'], suffix='_410466')
        make_av_comparison(basetable, color1=['f182m', 'f212n'], color2=['f212n', 'f405n'], color3=['f182m', 'f212n'], color4=['f212n', 'f466n'], color5=['f182m', 'f410m'], suffix='_212466', axlims2=(-0.1, 2.5, -0.1, 2.5), axlims1=(-0.1, 2.5, -0.1, 3.5))
        pl.close('all')