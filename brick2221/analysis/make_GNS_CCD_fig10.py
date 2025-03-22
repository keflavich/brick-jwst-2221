"""
2025-03-20 note: these figures look like they're missing some cross-matches in stripey areas in all three catalogs.  That is really discouraging.

"""
import numpy as np
import importlib as imp
from brick2221.analysis import plot_tools
imp.reload(plot_tools)
from brick2221.analysis.plot_tools import plot_extvec_ccd
from astropy.table import Table
from astropy import units as u
import pylab as pl
from brick2221.analysis.analysis_setup import ww410_merged as ww410
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.table import Table
from astropy import table
from astropy import units as u
from astroquery.vizier import Vizier

from dust_extinction.averages import CT06_MWGC, G21_MWAvg

from brick2221.analysis.analysis_setup import fh_merged as fh
from brick2221.analysis.analysis_setup import ww410_merged as ww
from brick2221.analysis.analysis_setup import ww410_merged as ww410
from brick2221.analysis.selections import load_table

import regions
import os
import glob

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')

paranal = SvoFps.get_filter_list('Paranal')
paranal.add_index('filterID')

e_HK = CT06_MWGC()(paranal.loc['Paranal/VISTA.H']['WavelengthEff']*u.AA) - CT06_MWGC()(paranal.loc['Paranal/VISTA.Ks']['WavelengthEff']*u.AA)
e_182212 =  CT06_MWGC()(jfilts.loc['JWST/NIRCam.F182M']['WavelengthEff']*u.AA) - CT06_MWGC()(jfilts.loc['JWST/NIRCam.F212N']['WavelengthEff']*u.AA)

if 'basepath' not in locals():
    basepath = '/orange/adamginsburg/jwst/brick/'

for catname, shortname in [('crowdsource_nsky0_merged_indivexp_photometry_tables_merged_qualcuts.fits', 'crowdsource_indivexp'),
                           ('crowdsource_nsky0_merged-reproject_photometry_tables_merged_20231003.fits', 'crowdsource_20231003'),
                           ('basic_merged_indivexp_photometry_tables_merged.fits', 'dao_basic_indivexp')]:
    basetable = Table.read(f'{basepath}/catalogs/{catname}')
    result = load_table(basetable, ww=ww)
    globals().update(result)
    assert 'ok2221' in result
    assert 'ok2221' in locals()

    fov = regions.Regions.read(f'{basepath}/regions_/brick_fov_2221and1182.reg')
    coord = fov[0].center
    height = fov[0].height
    width = fov[0].width
    height = width = max((width, height))

    galnuc2021 = Vizier(row_limit=-1).query_region(coordinates=coord, width=width, height=height, catalog=['J/A+A/653/A133'])[0]
    galnuc2021_crds = SkyCoord(galnuc2021['RAJ2000'], galnuc2021['DEJ2000'], frame='fk5')

    # Crossmatch galnuc w/"best"
    threshold = 0.3*u.arcsec
    idx, sidx, sep, sep3d = galnuc2021_crds.search_around_sky(basetable['skycoord_ref'], threshold)
    idx2, sidx2, sep2, sep3d2 = basetable['skycoord_ref'].search_around_sky(galnuc2021_crds, threshold)
    idxmatch, sepmatch, _ = coordinates.match_coordinates_sky(galnuc2021_crds, basetable['skycoord_ref'][idx2])

    galnuc_merged = table.hstack([galnuc2021[(sepmatch < threshold)],
                                  basetable[idx2][idxmatch[sepmatch < threshold]]])
    galnuc_merged.write(f'{basepath}/catalogs/GALACTICNUCLEUS_2021_merged_with_{shortname}.fits', overwrite=True)
    gntable = galnuc_merged

    pl.figure(figsize=(20,10))
    ax = pl.subplot(2,2,2, adjustable='box', aspect=0.88)

    crds = gntable['skycoord_f410m']
    sel = gntable['sep_f405n'].quantity < 0.1*u.arcsec
    sel &= gntable['mag_ab_f410m'] < 18.5
    assert sel.sum() > 0
    sel &= gntable['good_f466n']
    assert sel.sum() > 0
    sel &= gntable['good_f212n']
    assert sel.sum() > 0
    sel &= gntable['good_f410m']
    assert sel.sum() > 0
    sel &= gntable['good_f182m']
    assert sel.sum() > 0
    sel &= (ok2221[idx2][idxmatch[sepmatch < threshold]] | ok1182[idx2][idxmatch[sepmatch < threshold]])
    assert sel.sum() > 0
    xx,yy = ww410.world_to_pixel(crds[sel])

    colorby = gntable['mag_ab_f187n'] - gntable['mag_ab_f405n']

    colornorm = simple_norm(colorby[sel], stretch='linear', vmin=-0.1, vmax=4.0)
    cmap = 'YlOrRd'

    scat = ax.scatter(
                    crds.dec[sel],
                    crds.ra[sel],
                    #transform=ax.get_transform('world'),
                    #edgecolor='r', facecolor='none', marker='s')#r'$\rightarrow$')
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
    pl.setp(ax.get_xticklabels(), rotation=15, ha='right')


    ax2 = pl.subplot(2,2,3)
    #ax2.scatter(gntable['mag_ab_f410m'] - gntable['mag_ab_f466n'], gntable['mag_ab_f410m'], s=2, alpha=0.05, c='k')
    sc = ax2.scatter(
                    (gntable['Hmag'] - gntable['Ksmag'])[sel],
                    (gntable['mag_ab_f182m'] - gntable['mag_ab_f212n'])[sel],
                    s=0.5, alpha=0.75,
                    marker=',',
                    c=colorby[sel],
                    norm=colornorm, cmap=cmap
                    )
    # A_V=30 roughly corresponds to 1 mag color excess in 182-212
    # 1/(CT06_MWGC()(1.82*u.um) - CT06_MWGC()(2.12*u.um))

    plot_extvec_ccd(ax2,
                    ('Hmag', 'Ksmag'),
                    ('f182m', 'f212n'), start=(1.2, 0.35,),
                    color='k', extvec_scale=30, head_width=0.1)


    ax2.set_xlabel("(H-Ks)")
    ax2.set_ylabel("[F182M] - [F212N]")
    ax2.axis((0.0, 4, 0.0, 2.0));
    ax2.text(2.5, 0.6, "A$_V = 30$", ha='center')


    ax3 = pl.subplot(2,2,4)
    sc = ax3.scatter((gntable['Hmag'] - gntable['Ksmag'])[sel],
                (gntable['mag_ab_f410m'] - gntable['mag_ab_f466n'])[sel],
                s=0.5, alpha=0.75,
                c=colorby[sel],
                norm=colornorm, cmap=cmap
            )
    #sc = ax3.scatter((gntable_nrcb['mag_ab_f182m'] - gntable_nrcb['mag_ab_f212n'])[selb],
    #            (gntable_nrcb['mag_ab_f410m'] - gntable_nrcb['mag_ab_f466n'])[selb],
    #            s=1, alpha=0.75,
    #            c=colorbyb[selb],
    #            norm=colornorm, cmap=cmap
    #           )
    plot_extvec_ccd(ax3,
                    ('Hmag', 'Ksmag'),
                    ('f410m', 'f466n'), start=(0.5,0.75,), color='k', extvec_scale=30, head_width=0.1)


    ax3.set_ylabel("[F410M] - [F466N]")
    ax3.set_xlabel("(H-Ks)")
    ax3.axis((0,4,-0.75,1.25));
    ax3.text(1, 1., "A$_V = 30$", ha='center')

    pl.subplots_adjust(wspace=0.3, hspace=0.24)
    cb = pl.colorbar(mappable=sc, ax=pl.gcf().axes)
    cb.set_label("[F187N] - [F405N]")
    pl.suptitle(shortname)



    av182 = (gntable['mag_ab_f182m'] - gntable['mag_ab_f212n']) / e_182212
    avHK = (gntable['Hmag'] - gntable['Ksmag']) / e_HK

    ax4 = pl.subplot(2,2,1)
    ax4.scatter(avHK[sel], av182[sel], s=0.5, alpha=0.75, c=colorby[sel], norm=colornorm, cmap=cmap)
    ax4.plot(np.linspace(0,100,100), np.linspace(0,100,100), 'k--', zorder=-5, alpha=0.5)
    ax4.axvline(1.5 / e_HK, color='k', linestyle=':', zorder=-5, alpha=0.5)
    ax4.axhline(1.5 / e_HK, color='k', linestyle=':', zorder=-5, alpha=0.5)
    ax4.set_xlabel("A$_V$ (H-Ks)")
    ax4.set_ylabel("A$_V$ (F182M-F212N)")
    ax4.axis((0,100,0,100))



    pl.savefig(f"{basepath}/figures/ColorColorDiagrams_WithSourceMap_ALL_GN_2025_{shortname}.pdf", dpi=150, bbox_inches='tight')
    pl.savefig(f"{basepath}/figures/ColorColorDiagrams_WithSourceMap_ALL_GN_2025_{shortname}.png", dpi=150, bbox_inches='tight')
    print(f"Selected {sel.sum()} stars")

    
    pl.figure()
    pl.subplot(2,2,1).scatter(gntable['Ksmag'][sel], gntable['mag_ab_f212n'][sel], s=0.5, alpha=0.75)
    pl.plot(np.linspace(12.5,19.5,100), np.linspace(12.5,19.5,100), 'k--', zorder=-5, alpha=0.75)
    pl.xlabel("Ks")
    pl.ylabel("F212N")
    pl.subplot(2,2,2).scatter(gntable['Hmag'][sel], gntable['mag_ab_f182m'][sel], s=0.5, alpha=0.75)
    pl.plot(np.linspace(13,21,100), np.linspace(13,21,100), 'k--', zorder=-5, alpha=0.5)
    pl.xlabel("H")
    pl.ylabel("F182M")

    diff = gntable['Ksmag'][sel] - gntable['mag_ab_f212n'][sel]
    diff = np.array(diff[diff == diff])
    pl.subplot(2,2,3).hist(gntable['Ksmag'][sel] - gntable['mag_ab_f212n'][sel], bins=np.linspace(np.nanpercentile(diff, 0.1), np.nanpercentile(diff, 99.9), 50))
    pl.axvline(np.nanmedian(diff), color='k', linestyle=':', zorder=-5, alpha=0.5, label=f'{np.nanmedian(diff):.2f}')
    pl.legend(loc='best')
    pl.xlabel("Ks - 212N")

    pl.subplot(2,2,4).hist(gntable['Hmag'][sel] - gntable['mag_ab_f182m'][sel], bins=np.linspace(np.nanpercentile(diff, 0.1), np.nanpercentile(diff, 99.9), 50))
    diff = gntable['Hmag'][sel] - gntable['mag_ab_f182m'][sel]
    diff = diff[diff == diff]
    pl.axvline(np.nanmedian(diff), color='k', linestyle=':', zorder=-5, alpha=0.5, label=f'{np.nanmedian(diff):.2f}')
    pl.legend(loc='best')
    pl.xlabel("H - 182M")

    pl.savefig(f"{basepath}/figures/MagMagDiagrams_GN_2025_{shortname}.pdf", dpi=150, bbox_inches='tight')
    pl.savefig(f"{basepath}/figures/MagMagDiagrams_GN_2025_{shortname}.png", dpi=150, bbox_inches='tight')
    print(f"Selected {sel.sum()} stars")
