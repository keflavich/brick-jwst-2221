import pyavm
import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.wcs import WCS
import PIL
import os
from brick2221.analysis.make_icecolumn_fig9 import calc_av, ev
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from dust_extinction.averages import CT06_MWGC, G21_MWAvg, F11_MWGC
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

basepath = '/orange/adamginsburg/jwst/brick/'

def load_table():

    if os.path.exists(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20250721.fits'):
        basetable = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20250721.fits')
        print("Loaded merged1182_daophot_basic_indivexp (2025-07-21 version)")
    else:
        from brick2221.analysis.analysis_setup import fh_merged as fh, ww410_merged as ww410, ww410_merged as ww
        from brick2221.analysis.selections import load_table as load_table_
        basetable_merged1182_daophot = Table.read(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged.fits')
        result = load_table_(basetable_merged1182_daophot, ww=ww)
        ok2221 = result['ok2221']
        ok1182 = result['ok1182']
        #globals().update(result)
        basetable = basetable_merged1182_daophot[ok2221 | ok1182]
        del result
        print("Loaded merged1182_daophot_basic_indivexp")

        try:
            basetable.write(f'{basepath}/catalogs/basic_merged_indivexp_photometry_tables_merged_ok2221or1182_20250721.fits', overwrite=False)
        except:
            pass

    # this masks out all F444W sources, I don't know why.
    # from brick2221.analysis.analysis_setup import field_edge_regions, ww410_merged
    # edge_sources = field_edge_regions.contains(basetable['skycoord_f410m'], ww410_merged)
    # # don't remove masked sources
    # edge_sources &= ~basetable['skycoord_f410m'].mask
    # basetable = basetable[~edge_sources]

    return basetable


def nondetections_on_rgb(basetable,
                         rgb_imagename='images/BrickJWST_merged_longwave_narrowband_lighter.png',
                         detections=['F405N', 'F410M'],
                         nondetections=['F466N'],
                         ref_filter='f410m',
                         transform=None,
                         swapaxes_wcs=False,
                         flip_y=True,
                         flip_x=False,
                         rgb_name='RGB_merged',
                         axlims=None,
                         cmap='Reds',
                         avfilts=['F182M', 'F212N'],
                         ext=CT06_MWGC(),
                         av_threshold=17,
                         ):
    if not rgb_imagename.startswith('/'):
        rgb_imagename = f'{basepath}/{rgb_imagename}'
    avm = pyavm.AVM.from_image(rgb_imagename)
    img = PIL.Image.open(rgb_imagename)
    if transform is not None:
        img = img.transpose(transform)
    img_narrow = np.array(img)
    try:
        wwi_narrow = WCS(fits.Header.fromstring(avm.Spatial.FITSheader))
    except TypeError as ex:
        wwi_narrow = avm.to_wcs()
    if swapaxes_wcs:
        wwi_narrow = wwi_narrow.sub([2,1])
    fig = pl.figure(figsize=(12,4))
    ax = pl.subplot(projection=wwi_narrow)

    ax.imshow(img_narrow, zorder=-5)

    sel = np.ones(len(basetable), dtype=bool)
    for detection in detections:
        sel &= ~basetable[f'mag_ab_{detection.lower()}'].mask
    for nondetection in nondetections:
        sel &= basetable[f'mag_ab_{nondetection.lower()}'].mask

    av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext, return_av=True)
    sel &= (av > av_threshold)

    crds = basetable[f'skycoord_{ref_filter}']
    inds = np.argsort(av[sel])
    sc = ax.scatter_coord(crds[sel][inds], marker='o',
                          #c=basetable[f'mag_ab_{ref_filter.lower()}'][sel],
                          c=av[sel][inds],
                          norm=simple_norm(av[sel][inds], stretch='linear', vmin=av_threshold, vmax=100),
                          cmap='Greens', alpha=1, linewidths=0.5, s=5)

    pl.colorbar(mappable=sc, ax=ax)

    if axlims is not None:
        ax.axis(axlims)

    if flip_y:
        ax.set_ylim(ax.get_ylim()[::-1])
    elif flip_x:
        ax.set_xlim(ax.get_xlim()[::-1])

    if 'ra' in ax.coords:
        ra = lon = ax.coords['ra']
        dec = lat = ax.coords['dec']
        ra.set_major_formatter('hh:mm:ss.ss')
        dec.set_major_formatter('dd:mm:ss.ss')
        ra.set_axislabel('Right Ascension')
        dec.set_axislabel('Declination')
        if swapaxes_wcs:
            ra.set_ticks_position('l')
            ra.set_ticklabel_position('l')
            ra.set_axislabel_position('l')
            dec.set_ticks_position('b')
            dec.set_ticklabel_position('b')
            dec.set_axislabel_position('b')
    else:
        raise ValueError("No RA or GLON in WCS")

    fig.savefig(f"{basepath}/figures/Nondetections_on_{rgb_name}_{'+'.join(detections)}_{'-'.join(nondetections)}.png", dpi=200, bbox_inches='tight')


def blue_stars_on_rgb(basetable,
                      rgb_imagename='images/BrickJWST_merged_longwave_narrowband_lighter.png',
                      wcsaxes=None,
                      ext=CT06_MWGC(),
                      avfilts=['F182M', 'F212N'],
                      color_filter1='F405N',
                      color_filter2='F466N',
                      threshold=-0.3,
                      av_threshold=17,
                      ref_filter='f410m',
                      transform=None,
                      swapaxes_wcs=False,
                      flip_y=True,
                      flip_x=False,
                      rgb_name='RGB_merged',
                      axlims=None,
                      cmap='Reds',
                      cbar=False,
                      ):
    if not rgb_imagename.startswith('/'):
        rgb_imagename = f'{basepath}/{rgb_imagename}'
    avm = pyavm.AVM.from_image(rgb_imagename)
    img = PIL.Image.open(rgb_imagename)
    if transform is not None:
        img = img.transpose(transform)
    img_narrow = np.array(img)
    try:
        wwi_narrow = WCS(fits.Header.fromstring(avm.Spatial.FITSheader))
    except TypeError as ex:
        wwi_narrow = avm.to_wcs()

    if swapaxes_wcs:
        wwi_narrow = wwi_narrow.sub([2,1])

    if wcsaxes is None:
        wcsaxes = wwi_narrow

    fig = pl.figure(figsize=(6, 6))
    ax = pl.subplot(projection=wcsaxes)

    ax.imshow(img_narrow, zorder=-5, transform=ax.get_transform(wwi_narrow))

    av = calc_av(avfilts=avfilts, basetable=basetable, ext=ext, return_av=True)
    filter1_wav = int(color_filter1[1:-1])/100. * u.um
    filter2_wav = int(color_filter2[1:-1])/100. * u.um
    E_V_color = (ext(filter2_wav) - ext(filter1_wav))
    measured_color = basetable[f'mag_ab_{color_filter1.lower()}'] - basetable[f'mag_ab_{color_filter2.lower()}']
    unextincted_color = measured_color + E_V_color * av

    blue_co_ice = (unextincted_color < threshold) & (av > av_threshold)

    #ax.scatter(crds.ra[blue_410_405], crds.dec[blue_410_405], edgecolors='orange', facecolors='none', transform=ax.get_transform('world'))
    if 'ra' in ax.coords:
        ra = lon = ax.coords['ra']
        dec = lat = ax.coords['dec']
        ra.set_major_formatter('hh:mm:ss.ss')
        dec.set_major_formatter('dd:mm:ss.ss')
        ra.set_axislabel('Right Ascension')
        ra.set_ticklabel(rotation=25, pad=30)
        dec.set_axislabel('Declination')
        if swapaxes_wcs:
            ra.set_ticks_position('l')
            ra.set_ticklabel_position('l')
            ra.set_axislabel_position('l')
            dec.set_ticks_position('b')
            dec.set_ticklabel_position('b')
            dec.set_axislabel_position('b')
    elif 'glon' in ax.coords:
        glon = lon = ax.coords['glon']
        glat = lat = ax.coords['glat']
        glon.set_major_formatter('d.ddd')
        glat.set_major_formatter('d.ddd')
        glon.set_axislabel('Galactic Longitude')
        glat.set_axislabel('Galactic Latitude')

    #lon.set_ticks(spacing=30. * u.arcsec, color='red')
    #lat.set_ticks(spacing=30. * u.arcsec, color='blue')
    #lon.set_ticklabel(color='red')
    #lat.set_ticklabel(color='blue')
    #lon.grid(color='red')
    #lat.grid(color='blue')


    crds = basetable[f'skycoord_{ref_filter}']
    inds = np.argsort(unextincted_color[blue_co_ice])[::-1]
    sc = ax.scatter_coord(
        crds[blue_co_ice][inds],
        #transform=ax.get_transform('world'),
        marker='o', #facecolors=(0.2, 1, 0.6),
        linewidths=0.5,
        s=5,
        c=unextincted_color[blue_co_ice][inds],
        cmap=cmap,
    );

    if cbar:
        cb = pl.colorbar(mappable=sc, ax=ax, pad=0.01)
        cb.set_label(f'{color_filter1} - {color_filter2} (dereddened)')#, rotation=270, labelpad=20)

    if flip_y:
        ax.set_ylim(ax.get_ylim()[::-1])
    elif flip_x:
        ax.set_xlim(ax.get_xlim()[::-1])

    if axlims is not None:
        ax.axis(axlims)

    fig.savefig(f"{basepath}/paper_figures/BlueStars_on_{rgb_name}_{color_filter1}-{color_filter2}.png", dpi=200, bbox_inches='tight')

    fig2 = pl.figure()
    ax = pl.gca()
    ax.hist(unextincted_color, bins=np.linspace(-3, 3, 100), histtype='step', log=True, color='black')
    ax.hist(unextincted_color[blue_co_ice], bins=np.linspace(-3, 3, 100), histtype='step', log=True, color='red')
    ax.set_xlabel(f'{color_filter1} - {color_filter2} dereddened with CT06 using {avfilts[0]} and {avfilts[1]}')
    fig2.savefig(f"{basepath}/paper_figures/BlueStars_on_{rgb_name}_{color_filter1}-{color_filter2}_hist.png", dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    if 'basetable' not in globals():
        basetable = load_table()


    # cut out MUSTANG+ACES image
    mustang_aces_fn = '/orange/adamginsburg/ACES/mosaics/continuum/MUSTANG_12m_feather_noaxes.png'
    MUSTANG_ACES = PIL.Image.open(mustang_aces_fn)
    dim2sz = MUSTANG_ACES.size[1]
    MUSTANG_ACES = MUSTANG_ACES.crop((4200, 1200, 5200, 2100))
    ww = pyavm.AVM.from_image(mustang_aces_fn).to_wcs()
    #new_wcs = ww[4200:5200, dim2sz-2000:dim2sz-1300]
    new_wcs = ww[dim2sz-2100:dim2sz-1200, 4200:5200]
    new_wcs._naxis = MUSTANG_ACES.size
    new_avm = pyavm.AVM.from_wcs(new_wcs, shape=MUSTANG_ACES.size)

    crd = SkyCoord(0.253, 0.016, unit=(u.deg, u.deg), frame='galactic')
    assert new_wcs.footprint_contains(crd)

    brick_cutout = f'{basepath}/images/MUSTANG_ACES_cropped_to_brick.png'
    MUSTANG_ACES.save(brick_cutout)
    new_avm.embed(brick_cutout, brick_cutout)

    mustang_aces_fn = '/orange/adamginsburg/ACES/mosaics/continuum/MUSTANG_12m_feather_noaxes.png'
    MUSTANG_ACES = PIL.Image.open(mustang_aces_fn)
    MUSTANG_ACES = MUSTANG_ACES.crop((4400, 1400, 5000, 1950))
    ww = pyavm.AVM.from_image(mustang_aces_fn).to_wcs()
    new_wcs = ww[dim2sz-1950:dim2sz-1400, 4400:5000]
    new_wcs._naxis = MUSTANG_ACES.size
    assert new_wcs.footprint_contains(crd)

    new_avm = pyavm.AVM.from_wcs(new_wcs, shape=MUSTANG_ACES.size)
    brick_cutout = f'{basepath}/images/MUSTANG_ACES_cropped_to_brick_tight.png'
    MUSTANG_ACES.save(brick_cutout)
    new_avm.embed(brick_cutout, brick_cutout)

    wcsaxes = pyavm.AVM.from_image(f'{basepath}/images/BrickJWST_merged_longwave_narrowband_lighter.png').to_wcs()
    for rgb_name, axlims, wcsaxes in [('MUSTANG_12m_feather', None, None),
                                      ('MUSTANG_12m_feather_zoom', None, wcsaxes)]:
        print(f"Plotting {rgb_name}")
        blue_stars_on_rgb(basetable=basetable,
                        wcsaxes=wcsaxes,
                        rgb_imagename=f'{basepath}/images/MUSTANG_ACES_cropped_to_brick.png',
                        color_filter1='F444W',
                        color_filter2='F356W',
                        ref_filter='f444w',
                        avfilts=['F200W', 'F444W'],
                        swapaxes_wcs=False,
                        flip_y=False,
                        flip_x=False,
                        transform=PIL.Image.FLIP_TOP_BOTTOM,
                        threshold=-0.4,
                        rgb_name=rgb_name,
                        axlims=axlims,
                        cmap='Blues',
                        cbar=True,
                        )
        blue_stars_on_rgb(basetable=basetable,
                        wcsaxes=wcsaxes,
                        rgb_imagename=f'{basepath}/images/MUSTANG_ACES_cropped_to_brick.png',
                        color_filter1='F405N',
                        color_filter2='F466N',
                        ref_filter='f410m',
                        avfilts=['F182M', 'F212N'],
                        swapaxes_wcs=False,
                        flip_y=False,
                        flip_x=False,
                        transform=PIL.Image.FLIP_TOP_BOTTOM,
                        threshold=-0.4,
                        rgb_name=rgb_name,
                        axlims=axlims,
                        cmap='Blues',
                        cbar=True,
                      )
    pl.close('all')

    blue_stars_on_rgb(basetable=basetable, swapaxes_wcs=True, transform=None,
                      rgb_name='RGB_merged',
                      threshold=-0.4)

    nondetections_on_rgb(basetable=basetable,
                         swapaxes_wcs=True,
                         transform=None,
                         rgb_name='RGB_merged_starless',
                         detections=['F405N', 'F410M', 'F212N'],
                         avfilts=['F212N', 'F405N'],
                         nondetections=['F466N'],
                         ref_filter='f410m',
                         )

    nondetections_on_rgb(basetable=basetable, swapaxes_wcs=True, transform=None,
                         rgb_name='RGB_merged_starless',
                         detections=['F444W', 'F200W'],
                         nondetections=['F356W',],
                         ref_filter='f444w',
                         avfilts=['F200W', 'F444W'],
                         )
    nondetections_on_rgb(basetable=basetable, swapaxes_wcs=True, transform=None,
                         rgb_name='RGB_merged_starless',
                         detections=['F444W', 'F405N'],
                         nondetections=['F466N'],
                         ref_filter='f444w',
                         avfilts=['F405N', 'F444W'],
                         )
    nondetections_on_rgb(basetable=basetable, swapaxes_wcs=True, transform=None,
                         rgb_name='RGB_merged_starless',
                         detections=['F444W', 'F356W'],
                         nondetections=['F466N'],
                         ref_filter='f444w',
                         avfilts=['F356W', 'F444W'],
                         )
    pl.close('all')

    print("Plotting 444-356-200")
    blue_stars_on_rgb(basetable=basetable, rgb_imagename='pngs_444/Brick_RGB_444-356-200_log.png',
                      color_filter1='F444W',
                      color_filter2='F356W',
                      ref_filter='f444w',
                      avfilts=['F200W', 'F444W'],
                      swapaxes_wcs=False,
                      flip_y=False,
                      flip_x=False,
                      transform=PIL.Image.FLIP_LEFT_RIGHT,
                      threshold=-0.4,
                      rgb_name='RGB_merged'
                      )

    pl.close('all')