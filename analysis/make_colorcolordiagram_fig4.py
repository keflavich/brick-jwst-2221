import pylab as pl

import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.visualization import simple_norm

import analysis_setup
import selections
import plot_tools
from plot_tools import plot_extvec_ccd

from analysis_setup import filternames, basepath, img_nostars as img
from analysis_setup import fh_merged_reproject as fh, ww410_merged_reproject as ww410, ww410_merged_reproject as ww
from selections import main, basetable_merged_reproject, basetable_merged_reproject  as basetable

print()
print("merged-reproject")
result = main(basetable_merged_reproject, ww=ww)
all_good = result['all_good']
any_good = result['any_good']
any_saturated = result['any_saturated']
long_good = result['long_good']
bad = result['bad']
globals().update({key+"_mr": val for key, val in result.items()})

pl.figure(figsize=(20,10))
ax = pl.subplot(2,1,1, adjustable='box', aspect=0.88)

crds = basetable['skycoord_f410m']
sel = basetable['sep_f405n'].quantity < 0.1*u.arcsec
sel &= (~any_saturated) & (any_good) & long_good & ~bad
sel &= basetable['mag_ab_f410m'] < 15.4
xx,yy = ww410.world_to_pixel(crds[sel])

colorby = basetable['mag_ab_f187n'] - basetable['mag_ab_f405n']

colornorm = simple_norm(colorby[sel], stretch='linear', min_cut=0.5, max_cut=4.4)
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
sc = ax2.scatter((basetable['mag_ab_f182m'] - basetable['mag_ab_f212n'])[sel],
                 (basetable['mag_ab_f212n'] - basetable['mag_ab_f410m'])[sel],
                 s=0.5, alpha=0.75,
                 marker=',',
                 c=colorby[sel],
                 norm=colornorm, cmap=cmap
                )
# A_V=30 roughly corresponds to 1 mag color excess in 182-212
# 1/(CT06_MWGC()(1.82*u.um) - CT06_MWGC()(2.12*u.um))
plot_extvec_ccd(ax2, ('f182m', 'f212n'), ('f212n', 'f410m'), start=(1.5,-0.2,), color='k', extvec_scale=30, head_width=0.1)
ax2.set_xlabel("[F182M] - [F212N]")
ax2.set_ylabel("[F212N] - [F410M]");
ax2.axis((0.0, 3.4, -1.2, 4.3));
ax2.text(2, -0.2, "A$_V = 30$", ha='center')


ax3 = pl.subplot(2,3,6)
sc = ax3.scatter((basetable['mag_ab_f182m'] - basetable['mag_ab_f212n'])[sel],
            (basetable['mag_ab_f410m'] - basetable['mag_ab_f466n'])[sel],
            s=0.5, alpha=0.75,
            c=colorby[sel],
            norm=colornorm, cmap=cmap
           )
plot_extvec_ccd(ax3, ('f182m', 'f212n'), ('f410m', 'f466n'), start=(0.5,1,), color='k', extvec_scale=30, head_width=0.1)


ax3.set_ylabel("[F410M] - [F466N]")
ax3.set_xlabel("[F182M] - [F212N]")
ax3.axis((0, 3.4, -2.5, 2));
ax3.text(1, 1.25, "A$_V = 30$", ha='center')

cb = pl.colorbar(mappable=sc, ax=pl.gcf().axes)
cb.set_label("[F187N] - [F405N]")

pl.savefig(f"{basepath}/paper_co/figures/ColorColorDiagrams_WithSourceMap_ALL.pdf", dpi=150, bbox_inches='tight')
pl.savefig(f"{basepath}/paper_co/figures/ColorColorDiagrams_WithSourceMap_ALL.png", dpi=150, bbox_inches='tight')
print(f"Selected {sel.sum()} stars for the colorcolor digram plot")
