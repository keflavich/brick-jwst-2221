import numpy as np
import regions
import warnings
import glob
from astropy.io import fits
from astropy import stats
import pylab as pl
from astropy import units as u
from astropy.coordinates import SkyCoord
from grid_strategy import strategies
from astropy.table import Table
from astropy import wcs
from astropy.wcs import WCS
from astropy.visualization import simple_norm
import itertools
import dust_extinction
from astroquery.svo_fps import SvoFps
from astroquery.vizier import Vizier
from dust_extinction.averages import RRP89_MWGC, CT06_MWGC, F11_MWGC
from dust_extinction.parameter_averages import CCM89
import matplotlib as mpl

from filtering import get_fwhm

basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'
filternames = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m']

sqgrid = strategies.SquareStrategy()
rectgrid = strategies.RectangularStrategy()

mist = Table.read(f'{basepath}/isochrones/MIST_iso_633a08f2d8bb1.iso.cmd', header_start=12, data_start=13, format='ascii', delimiter=' ', comment='#')
# Hack, but good enough to first order
mist['410M405'] = mist['F410M']
mist['405M410'] = mist['F405N']
padova = Table.read(f'{basepath}/isochrones/padova_isochrone_package.dat', header_start=14, data_start=15, format='ascii', delimiter=' ', comment='#')

def crowdsource_diagnostic(basetable, exclude, filtername='f466n'):
    pl.figure(figsize=(10,6))
    ax = pl.subplot(1,2,1)
    ax.scatter(basetable[f'flux_{filtername}'][~exclude], basetable[f'fluxlbs_{filtername}'][~exclude], s=1)
    ax.plot([1e1,1e6], [1e1,1e6], 'k--', zorder=-1)
    ax.set_xlabel("Flux")
    ax.set_ylabel("Locally background-subtracted flux")
    ax.loglog();
    ax.axis([1e1,3e5,1e1,3e5]);
    ax2 = pl.subplot(1,2,2)
    ax2.scatter(basetable[f'flux_{filtername}'][~exclude], basetable[f'fluxiso_{filtername}'][~exclude], s=1)
    ax2.plot([1e1,1e6], [1e1,1e6], 'k--', zorder=-1)
    ax2.set_xlabel("Flux")
    ax2.set_ylabel("Iso flux")
    ax2.loglog();
    ax2.axis([1e1,3e5,1e1,3e5]);

def ccds(basetable, sel=True,
         colors=[('f410m', 'f466n'), ('f410m', 'f405n'), ('f405n', 'f466n'), ('410m405', 'f405n')],
         axlims=(-5,10,-5,10),
         fig=None,
         ext=CT06_MWGC(),
         extvec_scale=200,
         rasterized=True,
        ):
    if fig is None:
        fig = pl.figure()
    combos = list(itertools.combinations(colors, 2))
    gridspec = sqgrid.get_grid(len(combos))
    for ii, (color1, color2) in enumerate(combos):
        ax = fig.add_subplot(gridspec[ii])
        keys1 = [f'mag_ab_{col}' for col in color1]
        keys2 = [f'mag_ab_{col}' for col in color2]
        colorp1 = basetable[keys1[0]] - basetable[keys1[1]]
        colorp2 = basetable[keys2[0]] - basetable[keys2[1]]
        ax.scatter(colorp1, colorp2, s=5, alpha=0.5, c='k', rasterized=rasterized)
        ax.scatter(colorp1[sel], colorp2[sel], s=5, alpha=0.5, c='r', rasterized=rasterized)
        ax.set_xlabel(f"{color1[0]} - {color1[1]}")
        ax.set_ylabel(f"{color2[0]} - {color2[1]}")
        ax.axis(axlims)
        if ext is not None:
            w1 = 4.10*u.um if color1[0] == '410m405' else 4.05*u.um if color1[0] == '405m410' else int(color1[0][1:-1])/100*u.um
            w2 = 4.10*u.um if color1[1] == '410m405' else 4.05*u.um if color1[1] == '405m410' else int(color1[1][1:-1])/100*u.um
            w3 = 4.10*u.um if color2[0] == '410m405' else 4.05*u.um if color2[0] == '405m410' else int(color2[0][1:-1])/100*u.um
            w4 = 4.10*u.um if color2[1] == '410m405' else 4.05*u.um if color2[1] == '405m410' else int(color2[1][1:-1])/100*u.um

            if w1 > w2:
                w1,w2 = w2,w1
                color1 = color1[::-1]
            if w3 > w4:
                w3,w4 = w4,w3
                color2 = color2[::-1]

            e_1 = ext(w1) * extvec_scale
            e_2 = ext(w2) * extvec_scale
            e_3 = ext(w3) * extvec_scale
            e_4 = ext(w4) * extvec_scale
            ax.arrow(0, 0, e_1-e_2, e_3-e_4, color='y', head_width=0.5)
    return fig


def cmds(basetable, sel=True,
         colors=[('f410m', 'f466n'), ('f410m', 'f405n'), ('f405n', 'f466n'), ('410m405', 'f405n')],
         axlims=(-5,10,25,12),
         ext=CT06_MWGC(),
         fig=None,
         extvec_scale=30,
         markersize=5,
         rasterized=True,
        ):
    if fig is None:
        fig = pl.figure()
    gridspec = sqgrid.get_grid(len(colors))
    for ii, (f1, f2) in enumerate(colors):
        ax = fig.add_subplot(gridspec[ii])
        colorp = basetable[f'mag_ab_{f1}'] - basetable[f'mag_ab_{f2}']
        magp = basetable[f'mag_ab_{f1}']
        ax.scatter(colorp, magp, s=markersize, alpha=0.5, c='k', rasterized=rasterized)
        ax.scatter(colorp[sel], magp[sel], s=markersize, alpha=0.5, c='r', rasterized=rasterized)
        ax.set_xlabel(f"{f1} - {f2}")
        ax.set_ylabel(f"{f1}")
        ax.axis(axlims)

        if ext is not None:
            w1 = 4.10*u.um if f1 == '410m405' else 4.05*u.um if f1 == '405m410' else int(f1[1:-1])/100*u.um
            w2 = 4.10*u.um if f2 == '410m405' else 4.05*u.um if f2 == '405m410' else int(f2[1:-1])/100*u.um
            e_1 = ext(w1) * extvec_scale
            e_2 = ext(w2) * extvec_scale

            ax.arrow(0, 18, e_1-e_2, e_2, color='y', head_width=0.5)
    return fig


def color_plot(basetable,
               fh,
               sel=True,
               color=('f410m', 'f466n'),
               fig=None,
               show_extremes=False,
               reg=None,
               markersize=5,
               rasterized=True,
        ):
    if fig is None:
        fig = pl.figure()

    ww = wcs.WCS(fh['SCI'].header)
    ww2 = WCS()
    ww2.wcs.ctype = ww.wcs.ctype
    ww2.wcs.crval = ww.wcs.crval
    ww2.wcs.cdelt = ww.wcs.cdelt
    ww2.wcs.crpix = ww.wcs.crpix[::-1]
    ww2.wcs.cunit = ww.wcs.cunit

    if reg is not None:
        preg = reg.to_pixel(ww2)
        slcs, _ = preg.bounding_box.get_overlap_slices(fh['SCI'].data.shape)
        pregb = reg.to_pixel(ww)
        dslcs, _ = pregb.bounding_box.get_overlap_slices(fh['SCI'].data.shape)
    else:
        slcs = slice(None), slice(None)
        dslcs = slice(None), slice(None)


    ax = fig.add_subplot(projection=ww2[slcs])

    ax.imshow(fh['SCI'].data[dslcs], norm=simple_norm(fh['SCI'].data[dslcs], min_cut=0, max_cut=50),
              transform=ax.get_transform(ww[dslcs]), cmap='gray')
    axlims = ax.axis()

    f1, f2 = color
    crds = basetable['skycoord_f410m'][sel]
    colorp = (basetable[f'mag_ab_{f1}'] - basetable[f'mag_ab_{f2}'])[sel]
    magp = basetable[f'mag_ab_{f1}']

    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_major_formatter('dd:mm:ss.sss')
    lat.set_major_formatter('dd:mm:ss.sss')
    #lon.set_ticks(spacing= * u.arcsec)
    #lat.set_ticks(spacing= * u.arcsec)
    lon.set_axislabel("RA")
    lat.set_axislabel("Dec")
    #lon.set_format_unit(u.arcsec)
    #lat.set_format_unit(u.arcsec)

    green = (colorp > -2) & (colorp < 2)
    mappable = ax.scatter(crds.ra[green], crds.dec[green], c=colorp[green], s=markersize, alpha=0.5, cmap='jet',
                          norm=simple_norm(colorp[green], min_cut=-2, max_cut=2), transform=ax.get_transform('fk5'),
                          rasterized=rasterized
                         )
    pl.colorbar(mappable=mappable)

    if show_extremes:
        blue = colorp < -2
        mappable = ax.scatter(crds.ra[blue], crds.dec[blue],  s=markersize, alpha=0.5, #cmap='inferno',
                              #c=colorp[blue],
                              c='none',
                              edgecolor='b',
                              facecolor='none',
                              #norm=simple_norm(colorp[blue], min_cut=-5, max_cut=-2),
                              rasterized=rasterized,
                              transform=ax.get_transform('fk5'))
        #pl.colorbar(mappable=mappable)
        red = colorp > 2
        mappable = ax.scatter(crds.ra[red], crds.dec[red],
                              #c=colorp[red],
                              c='none',
                              edgecolor='r',
                              facecolor='none',
                              s=markersize, alpha=0.5, cmap='viridis',
                              #norm=simple_norm(colorp[red], min_cut=2, max_cut=6),
                              rasterized=rasterized,
                              transform=ax.get_transform('fk5'))
        #pl.colorbar(mappable=mappable)
    ax.axis(axlims)
    return fig


def cmds_withiso(basetable, sel=True,
                 colors=[('f410m', 'f466n'), ('f410m', 'f405n'), ('f405n', 'f466n'), ('410m405', 'f405n')],
                 axlims=(-5,10,25,12),
                 yval='f1',
                 fig=None,
                 ext=CT06_MWGC(),
                 iso=True,
                 exclude=False,
                 alpha_k=0.5,
                 distance_modulus=0,
                 markersize=5,
                 arrowhead_width=0.5,
                 arrow_start=(0, 18),
                 extinction_scaling_av=20,
                 rasterized=True,
        ):
    if fig is None:
        fig = pl.figure()
    gridspec = sqgrid.get_grid(len(colors))
    for ii, (f1, f2) in enumerate(colors):
        w1 = 4.10*u.um if f1 == '410m405' else 4.05*u.um if f1 == '405m410' else int(f1[1:-1])/100*u.um
        w2 = 4.10*u.um if f2 == '410m405' else 4.05*u.um if f2 == '405m410' else int(f2[1:-1])/100*u.um
        if w1 > w2:
            w1,w2 = w2,w1
            f1,f2 = f2,f1

        ax = fig.add_subplot(gridspec[ii])
        colorp = basetable[f'mag_ab_{f1}'] - basetable[f'mag_ab_{f2}']
        if yval == 'f1':
            magp = basetable[f'mag_ab_{f1}']
            ax.set_ylabel(f"{f1}")
            yval_ = f1
        elif yval == 'f2':
            magp = basetable[f'mag_ab_{f2}']
            ax.set_ylabel(f"{f2}")
            yval_ = f2
        else:
            raise ValueError("yval must be f1 or f2")
        ax.scatter(colorp[~exclude], magp[~exclude], s=markersize, alpha=alpha_k, c='k', rasterized=rasterized)
        ax.scatter(colorp[sel], magp[sel], s=markersize, alpha=0.5, c='r', rasterized=rasterized)
        ax.set_xlabel(f"{f1} - {f2}")
        ax.axis(axlims)
        if iso:
            agesel = mist['log10_isochrone_age_yr'] == 5
            ax.plot(mist[f1.upper()][agesel] - mist[f2.upper()][agesel],
                    mist[yval_.upper()][agesel] + distance_modulus,
                    color='b', linestyle='-', linewidth=0.5,
                    label='10$^5$ yr'
                    )
            agesel = mist['log10_isochrone_age_yr'] == 7
            ax.plot(mist[f1.upper()][agesel] - mist[f2.upper()][agesel],
                    mist[yval_.upper()][agesel] + distance_modulus,
                    color='g', linestyle=':', linewidth=0.5,
                    label='10$^7$ yr'
                    )
            agesel = mist['log10_isochrone_age_yr'] == 9
            ax.plot(mist[f1.upper()][agesel] - mist[f2.upper()][agesel],
                    mist[yval_.upper()][agesel] + distance_modulus,
                    color='c', linestyle='--', linewidth=0.5,
                    label='10$^9$ yr'
                    )

        if ext is not None:
            #print(w1,w2)
            e_1 = ext(w1) * extinction_scaling_av
            e_2 = ext(w2) * extinction_scaling_av

            if yval == 'f2':
                ax.arrow(arrow_start[0], arrow_start[1], e_1-e_2, e_2, color='y', head_width=arrowhead_width)
            elif yval == 'f1':
                ax.arrow(arrow_start[0], arrow_start[1], e_1-e_2, e_1, color='y', head_width=arrowhead_width)

        if ii == 1:
            ax.legend(bbox_to_anchor=(1.5, 1))

    return fig

def ccds_withiso(basetable, sel=True,
                 colors=[('f410m', 'f466n'), ('f410m', 'f405n'), ('f405n', 'f466n'), ('410m405', 'f405n')],
                 axlims=(-5,10,-5,10),
                 fig=None,
                 alpha_k=0.5,
                 ext=CT06_MWGC(),
                 exclude=False,
                 iso=True,
                 arrowhead_width=0.5,
                 rasterized=True,
        ):
    if fig is None:
        fig = pl.figure()
    combos = list(itertools.combinations(colors, 2))
    gridspec = sqgrid.get_grid(len(combos))
    for ii, (color1, color2) in enumerate(combos):
        w1 = 4.10*u.um if color1[0] == '410m405' else 4.05*u.um if color1[0] == '405m410' else int(color1[0][1:-1])/100*u.um
        w2 = 4.10*u.um if color1[1] == '410m405' else 4.05*u.um if color1[1] == '405m410' else int(color1[1][1:-1])/100*u.um
        w3 = 4.10*u.um if color2[0] == '410m405' else 4.05*u.um if color2[0] == '405m410' else int(color2[0][1:-1])/100*u.um
        w4 = 4.10*u.um if color2[1] == '410m405' else 4.05*u.um if color2[1] == '405m410' else int(color2[1][1:-1])/100*u.um

        if w1 > w2:
            w1,w2 = w2,w1
            color1 = color1[::-1]
        if w3 > w4:
            w3,w4 = w4,w3
            color2 = color2[::-1]

        e_1 = ext(w1) * 20
        e_2 = ext(w2) * 20
        e_3 = ext(w3) * 20
        e_4 = ext(w4) * 20

        ax = fig.add_subplot(gridspec[ii])
        keys1 = [f'mag_ab_{col}' for col in color1]
        keys2 = [f'mag_ab_{col}' for col in color2]
        colorp1 = basetable[keys1[0]] - basetable[keys1[1]]
        colorp2 = basetable[keys2[0]] - basetable[keys2[1]]
        ax.scatter(colorp1[~exclude], colorp2[~exclude], s=5, alpha=alpha_k, c='k', rasterized=rasterized)
        ax.scatter(colorp1[sel], colorp2[sel], s=5, alpha=0.5, c='r', rasterized=rasterized)
        ax.set_xlabel(f"{color1[0]} - {color1[1]}")
        ax.set_ylabel(f"{color2[0]} - {color2[1]}")
        ax.axis(axlims)
        if iso:
            agesel = mist['log10_isochrone_age_yr'] == 5
            ax.plot(mist[color1[0].upper()][agesel] - mist[color1[1].upper()][agesel],
                    mist[color2[0].upper()][agesel] - mist[color2[1].upper()][agesel],
                    color='b', linestyle='-', linewidth=1)
            agesel = mist['log10_isochrone_age_yr'] == 7
            ax.plot(mist[color1[0].upper()][agesel] - mist[color1[1].upper()][agesel],
                    mist[color2[0].upper()][agesel] - mist[color2[1].upper()][agesel],
                    color='g', linestyle='-', linewidth=1)
            agesel = mist['log10_isochrone_age_yr'] == 9
            ax.plot(mist[color1[0].upper()][agesel] - mist[color1[1].upper()][agesel],
                    mist[color2[0].upper()][agesel] - mist[color2[1].upper()][agesel],
                    color='c', linestyle='-', linewidth=1)
        ax.arrow(0, 0, e_1-e_2, e_3-e_4, color='y', head_width=arrowhead_width)
    return fig


def xmatch_plot(basetable, ref_filter='f410m', filternames=filternames, maxsep=0.13*u.arcsec):
    statsd = {}
    fig1 = pl.figure(1)
    fig2 = pl.figure(2)

    basecrds = basetable[f'skycoord_{ref_filter}']

    gridspec = sqgrid.get_grid(5)
    ii = 0

    for filtername in filternames:
        if filtername == ref_filter:
            continue
        ax = fig1.add_subplot(gridspec[ii])
        crds = basetable[f'skycoord_{filtername}']
        radiff = (crds.ra-basecrds.ra).to(u.arcsec)
        decdiff = (crds.dec-basecrds.dec).to(u.arcsec)
        sep = basetable[f'sep_{filtername}'].quantity.to(u.arcsec)
        ok = sep < maxsep
        ax.scatter(radiff, decdiff, marker=',', s=1, alpha=0.1)
        ax.scatter(radiff[ok], decdiff[ok], marker=',', s=1, alpha=0.1)
        ax.axis([-0.5, 0.5, -0.5, 0.5])
        ax.set_title(filtername)

        ax2 = fig2.add_subplot(gridspec[ii])
        ax2.hist(sep.to(u.arcsec).value, bins=np.linspace(0, 0.5))
        #ax2.set_xlabel("Separation (\")")
        ax2.set_title(filtername)

        statsd[filtername] = {
            'med': np.median(sep),
            'mad': stats.mad_std(sep),
            'std': np.std(sep),
            'med_thr': np.median(sep[ok]),
            'mad_thr': stats.mad_std(sep[ok]),
            'std_thr': np.std(sep[ok]),
        }
        ii+=1

    fig1.supxlabel("RA Offset (\")")
    fig1.supylabel("Dec Offset (\")")
    fig2.supxlabel("Offset (\")")
    fig1.tight_layout()
    fig2.tight_layout()
    return statsd

def starzoom(coords, cutoutsize=1*u.arcsec, fontsize=14):
    reg = regions.RectangleSkyRegion(center=coords, width=cutoutsize, height=cutoutsize)
    ii = 0
    with mpl.rc_context({"font.size": fontsize}):

        fig = pl.figure(figsize=(12,4))
        filters_plotted = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for fn in sorted(glob.glob(f'{basepath}/F*/pipeline/*nircam*nrc*_i2d.fits')):
                hdr0 = fits.getheader(fn)
                filtername = hdr0['PUPIL']+hdr0['FILTER']
                if filtername in filters_plotted:
                    continue
                hdr = fits.getheader(fn, ext=('SCI', 1))
                ww = wcs.WCS(hdr)
                if ww.footprint_contains(coords):
                    try:
                        hdr['OLCRVAL1']
                        #print(fn, fits.getheader(fn, ext=("SCI", 1))['OLCRVAL1'])
                    except Exception as ex:
                        #print(ex)
                        continue
                    data = fits.getdata(fn, ext=('SCI',1))
                    mask = reg.to_pixel(ww).to_mask()
                    slcs,_ = mask.get_overlap_slices(data.shape)


                    xc, yc = map(int, map(np.round, ww.world_to_pixel(coords)))
                    center_value = data[yc,xc]
                    good_center = np.isfinite(center_value) and center_value > 2
                    maxval = None #center_value if good_center else None
                    minval = 0 if good_center else None
                    stretch = 'log'# if np.isfinite(center_value) else 'asinh'
                    max_percent = 99.5
                    min_percent = None if good_center else 1.0
                    #print(f"center_value={center_value}, this is {'good' if good_center else 'bad'}")

                    ax = pl.subplot(1,6,ii+1)
                    ax.imshow(data[slcs], norm=simple_norm(data[slcs],
                                                           stretch=stretch,
                                                           min_percent=min_percent,
                                                           max_percent=max_percent,
                                                           min_cut=minval,
                                                           max_cut=maxval),
                              origin='lower', cmap='gray_r')
                    xx, yy = ww[slcs].world_to_pixel(coords)
                    ax.plot(xx, yy, 'rx')
                    pixscale = ww.proj_plane_pixel_area()**0.5
                    quartas = (0.25*u.arcsec/pixscale).decompose().value

                    if ii == 0:
                        xoffset = 4
                        ax.plot([xoffset, xoffset+quartas], [2, 2], color='r')
                        ax.text(xoffset+quartas/2, 3, '0.25"', color='r', horizontalalignment='center')

                    shp = data[slcs].shape

                    unit = u.Unit(hdr['BUNIT'])
                    fwhm, fwhm_pix = get_fwhm(hdr0)
                    fwhm = u.Quantity(fwhm, u.arcsec)
                    # debug print(unit, fwhm, pixscale)
                    max_flux_jy = (center_value * unit *
                                   (2*np.pi / (8*np.log(2))) *
                                   fwhm_pix**2 *
                                   pixscale**2).to(u.Jy)

                    if good_center:
                        ax.text(shp[1]-quartas*1.75, shp[0]-quartas/1.1,
                                f'{max_flux_jy.to(u.mJy):0.1f}', horizontalalignment='center',
                                #color='r',
                               )

                    ax.set_title(filtername.replace("CLEAR","").replace("F444W",""))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    filters_plotted.append(filtername)
                    ii += 1
            #if len(filters_plotted) == 0:
            #    print(f'Coordinate {coords} not in footprint')
    return fig

def starzoom_spitzer(coords, cutoutsize=10*u.arcsec, fontsize=14):
    reg = regions.RectangleSkyRegion(center=coords, width=cutoutsize, height=cutoutsize)

    flist = {
        'I1': '/orange/adamginsburg/spitzer/GLIMPSE/GLM_00000+0000_mosaic_I1.fits',
        'I2': '/orange/adamginsburg/spitzer/GLIMPSE/GLM_00000+0000_mosaic_I2.fits',
        'I3': '/orange/adamginsburg/spitzer/GLIMPSE/GLM_00000+0000_mosaic_I3.fits',
        'I4': '/orange/adamginsburg/spitzer/GLIMPSE/GLM_00000+0000_mosaic_I4.fits',
        'M1': '/orange/adamginsburg/spitzer/mips/MG0000n005_024.fits'}
    # https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/5/
    fwhms = {'I1': 1.66,
             'I2': 1.72,
             'I3': 1.88,
             'I4': 1.98,
             'M1': 5.9,
            }
    ii = 0
    with mpl.rc_context({"font.size": fontsize}):

        fig = pl.figure(figsize=(12,4))
        filters_plotted = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for filtername, fn in flist.items():
                if filtername in filters_plotted:
                    continue
                hdr0 = hdr = fits.getheader(fn)
                ww = wcs.WCS(hdr)
                if ww.footprint_contains(coords):
                    data = fits.getdata(fn)
                    mask = reg.to_pixel(ww).to_mask()
                    slcs,_ = mask.get_overlap_slices(data.shape)

                    xc, yc = map(int, map(np.round, ww.world_to_pixel(coords)))
                    center_value = data[yc,xc]
                    good_center = np.isfinite(center_value)
                    maxval = None #center_value if good_center else None
                    minval = 0 if good_center else None
                    stretch = 'log'# if np.isfinite(center_value) else 'asinh'
                    max_percent = 99.5
                    min_percent = None if good_center else 1.0
                    #print(f"center_value={center_value}, this is {'good' if good_center else 'bad'}")

                    ax = pl.subplot(1,6,ii+1)
                    ax.imshow(data[slcs], norm=simple_norm(data[slcs],
                                                           stretch=stretch,
                                                           min_percent=min_percent,
                                                           max_percent=max_percent,
                                                           min_cut=minval,
                                                           max_cut=maxval),
                              origin='lower', cmap='gray_r')
                    xx, yy = ww[slcs].world_to_pixel(coords)
                    ax.plot(xx, yy, 'rx')
                    pixscale = ww.proj_plane_pixel_area()**0.5
                    quartas = (1*u.arcsec/pixscale).decompose().value

                    if ii == 0:
                        xoffset = 1
                        ax.plot([xoffset, xoffset+quartas], [2, 2], color='r')
                        ax.text(xoffset+quartas/2, 3, '1"', color='r', horizontalalignment='center')

                    shp = data[slcs].shape
                    unit = u.Unit(hdr['BUNIT'])
                    fwhm = fwhms[filtername]
                    fwhm = u.Quantity(fwhm, u.arcsec)
                    fwhm_pix = (fwhm / pixscale).decompose().value
                    # debug print(unit, fwhm, pixscale)
                    max_flux_jy = (center_value * unit *
                                   (2*np.pi / (8*np.log(2))) *
                                   fwhm_pix**2 *
                                   pixscale**2).to(u.Jy)

                    if good_center:
                        ax.set_title(f'{filtername}: {max_flux_jy.to(u.mJy):0.1f}')
                        #ax.text(shp[1]-quartas*1.75, shp[0]-quartas/1.1,
                        #        f'{max_flux_jy.to(u.mJy):0.1f}', horizontalalignment='center',
                        #        #color='r',
                        #       )
                    else:
                        ax.set_title(filtername)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    filters_plotted.append(filtername)
                    ii += 1
            #if len(filters_plotted) == 0:
            #    print(f'Coordinate {coords} not in footprint')
    return fig

def make_sed(coord, basetable, radius=0.5*u.arcsec):
    skycrds_cat = basetable['skycoord_f410m']
    idx = coord.separation(skycrds_cat) < radius
    if len(idx) == 0:
        raise
    else:
        idx = np.argmin(coord.separation(skycrds_cat))

    spitzer = Vizier.query_region(coordinates=coord, radius=radius, catalog=['II/295/SSTGC'])[0]
    if len(spitzer) > 0:
        spitzer_crds = SkyCoord(spitzer['RAJ2000'], spitzer['DEJ2000'], frame='fk5', unit=(u.hour, u.deg))
        spitzindex = coord.separation(spitzer_crds) < radius
        #print(len(spitzindex))
        spitzermatch = spitzer[spitzindex]

    vvvdr2 = Vizier.query_region(coordinates=coord, radius=0.5*u.arcsec, catalog=['II/348/vvv2'])[0]
    if len(spitzer) > 0:
        vvvdr2_crds = SkyCoord(vvvdr2['RAJ2000'], vvvdr2['DEJ2000'], frame='fk5', unit=(u.hour, u.deg))
        vvvindex = coord.separation(vvvdr2_crds) < radius
        #print(len(vvvindex))
        vvvmatch = vvvdr2[vvvindex]

    wavelengths = []
    fluxes = []
    widths = []
    telescope = 'JWST'
    instrument = 'NIRCAM'
    filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
    filter_table.add_index('filterID')
    for filtername in filternames:
        instrument = 'NIRCam'
        filtername = filtername.upper()
        eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filtername}']['WavelengthEff'] * u.AA
        wavelengths.append(eff_wavelength)
        eff_width = filter_table.loc[f'{telescope}/{instrument}.{filtername}']['WidthEff'] * u.AA
        widths.append(eff_width)
        filtername = filtername.lower()
        fluxes.append(basetable[f'flux_jy_{filtername}'][idx] * u.Jy)


    telescope = 'Spitzer'
    instrument = 'IRAC'
    filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
    filter_table.add_index('filterID')
    for filtername,colname in [('I1', '_3.6mag'),
                               ('I2', '_4.5mag'),
                               ('I3', '_5.8mag'),
                               ('I4', '_8.0mag')]:
        eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filtername}']['WavelengthEff'] * u.AA
        wavelengths.append(eff_wavelength)
        fluxes.append(10**(-spitzer[colname]/2.5) * filter_table.loc[f'{telescope}/{instrument}.{filtername}']['ZeroPoint'] * u.Jy)
        eff_width = filter_table.loc[f'{telescope}/{instrument}.{filtername}']['WidthEff'] * u.AA
        widths.append(eff_width)

    return wavelengths, widths, fluxes
