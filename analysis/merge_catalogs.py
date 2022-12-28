import numpy as np
import os
import warnings
from astropy.io import fits
import glob
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils import CircularAperture, EPSFBuilder, find_peaks, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import stats
from astropy.table import Table
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u
import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'
pl.rcParams['figure.figsize'] = (10,8)
pl.rcParams['figure.dpi'] = 100

basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'
filternames = ['f410m', 'f212n', 'f466n', 'f405n', 'f187n', 'f182m']

def merge_catalogs(tbls, catalog_type='crowdsource', module='nrca',
                   ref_filter='f410m',
                   max_offset=0.25*u.arcsec):
    basetable = master_tbl = [tb for tb in tbls if tb.meta['filter'] == ref_filter][0].copy()
    basetable.meta['astrometric_reference_wavelength'] = ref_filter
    flag_near_saturated(basetable, filtername=ref_filter)
    # replace_saturated adds more rows
    replace_saturated(basetable, filtername=ref_filter)
    basecrds = basetable['skycoord']
    print(f"filter {basetable.meta['filter']} has {len(basetable)} rows")

    meta = {}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for colname in basetable.colnames:
            basetable.rename_column(colname, colname+"_"+basetable.meta['filter'])

        for tbl in tbls[1:]:
            wl = tbl.meta['filter']
            flag_near_saturated(tbl, filtername=wl)
            # replace_saturated adds more rows
            replace_saturated(tbl, filtername=wl)

            print(f"filter {wl} has {len(tbl)} rows")
            crds = tbl['skycoord']
            matches, sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)

            # do one iteration of bulk offset measurement
            radiff = (crds.ra[matches]-basecrds.ra).to(u.arcsec)
            decdiff = (crds.dec[matches]-basecrds.dec).to(u.arcsec)
            oksep = sep < max_offset
            medsep_ra, medsep_dec = np.median(radiff[oksep]), np.median(decdiff[oksep])
            tbl.meta[f'ra_offset_from_{ref_filter}'] = medsep_ra
            tbl.meta[f'dec_offset_from_{ref_filter}'] = medsep_dec
            newcrds = SkyCoord(crds.ra - medsep_ra, crds.dec - medsep_dec, frame=crds.frame)
            tbl['skycoord'] = newcrds


            matches, sep, _ = basecrds.match_to_catalog_sky(newcrds, nthneighbor=1)

            basetable.add_column(name=f"sep_{wl}", col=sep)
            basetable.add_column(name=f"id_{wl}", col=matches)
            matchtb = tbl[matches]
            for cn in matchtb.colnames:
                matchtb.rename_column(cn, f"{cn}_{wl}")

            basetable = table.hstack([basetable, matchtb], join_type='exact')
            meta[f'{wl[1:-1]}pxdg'.upper()] = tbl.meta['pixelscale_deg2']
            meta[f'{wl[1:-1]}pxas'.upper()] = tbl.meta['pixelscale_arcsec']
            for key in tbl.meta:
                meta[f'{wl[1:-1]}{key[:4]}'.upper()] = tbl.meta[key]
            # DEBUG if hasattr(basetable[f'{cn}_{wl}'], 'mask'):
            # DEBUG     print(f"Table has mask sum for column {cn} {basetable[cn+'_'+wl].mask.sum()}")

            bad = np.isnan(tbl['mag_ab']) & (tbl['flux'] > 0)
            if any(bad):
                raise ValueError("Bad magnitudes for good fluxes")

        # Line-subtract the F410 continuum band
        # 0.16 is from BrA_separation
        # 0.196 is the 'post-destreak' version, which might (?) be better
        # 0.11 is the theoretical version from RecombFilterDifferencing
        # 0.16 still looks like the best; 0.175ish is the median, but 0.16ish is the mode
        # but we use 0.11, the theoretica one, because we don't necessarily expect a good match!
        f405to410_scale = 0.11
        basetable.add_column(basetable['flux_jy_f410m'] - basetable['flux_jy_f405n'] * f405to410_scale, name='flux_jy_410m405')
        basetable.add_column(basetable['flux_jy_410m405'].to(u.ABmag), name='mag_ab_410m405')
        # Then subtract that remainder back from the F405 band to get the continuum-subtracted F405
        basetable.add_column(basetable['flux_jy_f405n'] - basetable['flux_jy_410m405'], name='flux_jy_405m410')
        basetable.add_column(basetable['flux_jy_405m410'].to(u.ABmag), name='mag_ab_405m410')

        # Line-subtract the F182 continuum band
        # 0.11 is the theoretical bandwidth fraction
        # PaA_separation_nrcb gives 0.175ish -> 0.183 with "latest"
        # 0.18 is closer to the histogram mode
        f187to182_scale = 0.11
        basetable.add_column(basetable['flux_jy_f182m'] - basetable['flux_jy_f187n'] * f187to182_scale, name='flux_jy_182m187')
        basetable.add_column(basetable['flux_jy_182m187'].to(u.ABmag), name='mag_ab_182m187')
        # Then subtract that remainder back from the F187 band to get the continuum-subtracted F187
        basetable.add_column(basetable['flux_jy_f187n'] - basetable['flux_jy_182m187'], name='flux_jy_187m182')
        basetable.add_column(basetable['flux_jy_187m182'].to(u.ABmag), name='mag_ab_187m182')

        # DEBUG for colname in basetable.colnames:
        # DEBUG     print(f"colname {colname} has mask: {hasattr(basetable[colname], 'mask')}")
        basetable.meta = meta
        assert '212PXDG' in meta
        assert '212PXDG' in basetable.meta

        basetable.write(f"{basepath}/catalogs/{catalog_type}_{module}_photometry_tables_merged.ecsv", overwrite=True)
        # DO NOT USE FITS in production, it drops critical metadata
        basetable.write(f"{basepath}/catalogs/{catalog_type}_{module}_photometry_tables_merged.fits", overwrite=True)

        # testtb = Table.read(f"{basepath}/catalogs/{catalog_type}_{module}_photometry_tables_merged.fits")
        # assert '212PXDG' in testtb.meta

def merge_crowdsource(module='nrca', suffix=""):
    imgfns = [x
          for filn in filternames
          for x in glob.glob(f"{basepath}/{filn.upper()}/pipeline/jw02221-o001_t001_nircam*{filn.lower()}*{module}_i2d.fits")
          if f'{module}_' in x or f'{module}1_' in x
         ]

    catfns = [x
              for filn in filternames
              for x in glob.glob(f"{basepath}/{filn.upper()}/{filn.lower()}*{module}_crowdsource{suffix}.fits")
             ]
    if len(catfns) != 6:
        raise ValueError(f"len(catfns) = {len(catfns)}.  catfns: {catfns}")
    tbls = [Table.read(catfn) for catfn in catfns]

    for catfn, tbl in zip(catfns, tbls):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = os.path.basename(catfn).split("_")[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #wcses = [wcs.WCS(fits.getheader(fn.replace("_crowdsource", "_crowdsource_skymodel"))) for fn in catfns]
        imgs = [fits.getdata(fn, ext=('SCI', 1)) for fn in imgfns]
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) for fn in imgfns]

    for tbl, ww in zip(tbls, wcses):
        # Now done in the original catalog making step tbl['y'],tbl['x'] = tbl['x'],tbl['y']
        if 'skycoord' not in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x'], tbl['y'])
            tbl.add_column(crds, name='skycoord')
        else:
            crds = tbl['skycoord']
        tbl.meta['pixelscale_deg2'] = ww.proj_plane_pixel_area()
        tbl.meta['pixelscale_arcsec'] = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        flux_jy = (tbl['flux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        eflux_jy = (tbl['dflux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
        abmag = flux_jy.to(u.ABmag)
        abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux_jy)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tbl.add_column(flux_jy, name='flux_jy')
            tbl.add_column(eflux_jy, name='eflux_jy')
            tbl.add_column(abmag, name='mag_ab')
            tbl.add_column(abmag_err, name='emag_ab')
        if hasattr(tbl['mag_ab'], 'mask'):
            print(f'ab mag tbl col has mask sum = {tbl["mag_ab"].mask.sum()} masked values')
        if hasattr(abmag, 'mask'):
            print(f'ab mag has mask sum = {abmag.mask.sum()} masked values')
        if hasattr(tbl['flux'], 'mask'):
            print(f'ab mag has mask sum = {tbl["flux"].mask.sum()} masked values')

    merge_catalogs(tbls, catalog_type=f'crowdsource{suffix}', module=module)


def merge_daophot(module='nrca', detector='', daophot_type='basic'):
    catfns = daocatfns = [
        f"{basepath}/{filtername.upper()}/{filtername.lower()}_{module}{detector}_daophot_{daophot_type}.fits"
        for filtername in filternames
    ]
    imgfns = [x
          for filn in filternames
          for x in glob.glob(f"{basepath}/{filn.upper()}/pipeline/jw02221-o001_t001_nircam*{filn.lower()}*{module}_i2d.fits")
          if f'{module}_' in x or f'{module}1_' in x
         ]

    tbls = [Table.read(catfn) for catfn in daocatfns]

    for catfn, tbl, filtername in zip(catfns, tbls, filternames):
        tbl.meta['filename'] = catfn
        tbl.meta['filter'] = filtername

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #wcses = [wcs.WCS(fits.getheader(fn.replace("_crowdsource", "_crowdsource_skymodel"))) for fn in catfns]
        imgs = [fits.getdata(fn, ext=('SCI', 1)) for fn in imgfns]
        wcses = [wcs.WCS(fits.getheader(fn, ext=('SCI', 1))) for fn in imgfns]

    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')

    for tbl, ww in zip(tbls, wcses):
        if 'x_fit' in tbl.colnames:
            crds = ww.pixel_to_world(tbl['x_fit'], tbl['y_fit'])
        else:
            crds = ww.pixel_to_world(tbl['x_0'], tbl['y_0'])
        if 'skycoord' not in tbl.colnames:
            tbl.add_column(crds, name='skycoord')
        tbl.meta['pixelscale_deg2'] = ww.proj_plane_pixel_area()
        tbl.meta['pixelscale_arcsec'] = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
        flux = tbl['flux_fit'] if 'flux_fit' in tbl.colnames else tbl['flux_0']
        filtername = tbl.meta['filter']

        row = fwhm_tbl[fwhm_tbl['Filter'] == filtername.upper()]
        fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
        fwhm_pix = float(row['PSF FWHM (pixel)'][0])
        tbl.meta['fwhm_arcsec'] = fwhm
        tbl.meta['fwhm_pix'] = fwhm_pix

        with np.errstate(all='ignore'):
            flux_jy = (flux * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * fwhm_arcsec**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
            abmag = flux_jy.to(u.ABmag)
            #eflux_jy = (tbl['dflux'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * tbl['fwhm']**2 * tbl.meta['pixelscale_deg2']).to(u.Jy)
            #abmag_err = 2.5 / np.log(10) * eflux_jy / flux_jy
        tbl.add_column(flux_jy, name='flux_jy')
        tbl.add_column(abmag, name='mag_ab')
        #tbl.add_column(eflux_jy, name='eflux_jy')
        #tbl.add_column(abmag_err, name='emag_ab')

    merge_catalogs(tbls, catalog_type=daophot_type, module=module)


def flag_near_saturated(cat, filtername, radius=None):
    satstar_cat_fn = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-merged_i2d_satstar_catalog.fits'
    satstar_cat = Table.read(satstar_cat_fn)
    satstar_coords = satstar_cat['skycoord_fit']

    cat_coords = cat['skycoord']

    if radius is None:
        radius = {'f466n': 0.55*u.arcsec,
                  'f212n': 0.55*u.arcsec,
                  'f187n': 0.55*u.arcsec,
                  'f405n': 0.55*u.arcsec,
                  'f182m': 0.55*u.arcsec,
                  'f410m': 0.55*u.arcsec,
                  }[filtername]

    idx_cat, idx_sat, sep, _ = satstar_coords.search_around_sky(cat_coords, radius)

    near_sat = np.zeros(len(cat), dtype='bool')
    near_sat[idx_cat] = True

    cat.add_column(near_sat, name=f'near_saturated_{filtername}')

def replace_saturated(cat, filtername, radius=None):
    satstar_cat_fn = f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-merged_i2d_satstar_catalog.fits'
    satstar_cat = Table.read(satstar_cat_fn)
    satstar_coords = satstar_cat['skycoord_fit']

    cat_coords = cat['skycoord']

    if radius is None:
        radius = {'f466n': 0.1*u.arcsec,
                  'f212n': 0.05*u.arcsec,
                  'f187n': 0.05*u.arcsec,
                  'f405n': 0.1*u.arcsec,
                  'f182m': 0.05*u.arcsec,
                  'f410m': 0.1*u.arcsec,
                  }[filtername]

    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    fwhm = u.Quantity(fwhm_tbl[fwhm_tbl['Filter'] == filtername.upper()]['PSF FWHM (arcsec)'], u.arcsec)

    flux_jy = (satstar_cat['flux_fit'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * fwhm**2).to(u.Jy)
    eflux_jy = (satstar_cat['flux_unc'] * u.MJy/u.sr * (2*np.pi / (8*np.log(2))) * fwhm**2).to(u.Jy)
    abmag = flux_jy.to(u.ABmag)
    abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux_jy)
    satstar_cat['mag_ab'] = abmag
    satstar_cat['emag_ab'] = abmag_err

    idx_cat, idx_sat, sep, _ = satstar_coords.search_around_sky(cat_coords, radius)

    replaced_sat = np.zeros(len(cat), dtype='bool')
    replaced_sat[idx_cat] = True

    if 'flux' in cat.colnames:
        cat['flux'][idx_cat] = satstar_cat['flux_fit'][idx_sat]
        cat['dflux'][idx_cat] = satstar_cat['flux_unc'][idx_sat]
        cat['skycoord'][idx_cat] = satstar_cat['skycoord_fit'][idx_sat]
        cat['x'][idx_cat] = satstar_cat['x_fit'][idx_sat]
        cat['y'][idx_cat] = satstar_cat['y_fit'][idx_sat]
        cat['dx'][idx_cat] = satstar_cat['x_0_unc'][idx_sat]
        cat['dy'][idx_cat] = satstar_cat['y_0_unc'][idx_sat]

        cat['mag_ab'][idx_cat] = abmag[idx_sat]
        cat['emag_ab'][idx_cat] = abmag_err[idx_sat]

        # ID the stars that are saturated-only (not INCluded in the orig cat)
        satstar_not_inc = np.ones(len(satstar_cat), dtype='bool')
        satstar_not_inc[idx_sat] = False
        satstar_toadd = satstar_cat[satstar_not_inc]

        satstar_toadd.rename_column('flux_fit', 'flux')
        satstar_toadd.rename_column('flux_unc', 'dflux')
        satstar_toadd.rename_column('skycoord_fit', 'skycoord')
        satstar_toadd.rename_column('x_fit', 'x')
        satstar_toadd.rename_column('y_fit', 'y')
        satstar_toadd.rename_column('x_0_unc', 'dx')
        satstar_toadd.rename_column('y_0_unc', 'dy')

        for colname in cat.colnames:
            if colname not in satstar_toadd.colnames:
                satstar_toadd.add_column(np.ones(len(satstar_toadd))*np.nan, name=colname)
        for colname in satstar_toadd.colnames:
            if colname not in cat.colnames:
                satstar_toadd.remove_column(colname)

        for row in satstar_toadd:
            cat.add_row(dict(row))



    elif 'flux_fit' in cat.colnames:
        # DAOPHOT
        cat['flux_fit'][idx_cat] = satstar_cat['flux_fit'][idx_sat]
        cat['flux_unc'][idx_cat] = satstar_cat['flux_unc'][idx_sat]
        cat['skycoord_fit'][idx_cat] = satstar_cat['skycoord_fit'][idx_sat]
        cat['x_fit'][idx_cat] = satstar_cat['x_fit'][idx_sat]
        cat['y_fit'][idx_cat] = satstar_cat['y_fit'][idx_sat]
        cat['x_0_unc'][idx_cat] = satstar_cat['x_0_unc'][idx_sat]
        cat['y_0_unc'][idx_cat] = satstar_cat['y_0_unc'][idx_sat]

        cat['mag_ab'][idx_cat] = abmag[idx_sat]
        cat['emag_ab'][idx_cat] = abmag_err[idx_sat]

        # ID the stars that are saturated-only (not INCluded in the orig cat)
        satstar_not_inc = np.ones(len(satstar_cat), dtype='bool')
        satstar_not_inc[idx_sat] = False
        satstar_toadd = satstar_cat[satstar_not_inc]

        for colname in cat.colnames:
            if colname not in satstar_toadd.colnames:
                satstar_toadd.add_column(np.ones(len(satstar_toadd))*np.nan, name=colname)
        for colname in satstar_toadd.colnames:
            if colname not in cat.colnames:
                satstar_toadd.remove_column(colname)

        for row in satstar_toadd:
            cat.add_row(dict(row))

    # we've added on more rows that are all 'replaced_sat'
    replaced_sat_ = np.ones(len(cat), dtype='bool')
    replaced_sat_[:len(replaced_sat)] = replaced_sat

    cat.add_column(replaced_sat_, name='replaced_saturated')

def main():
    for module in ('merged', 'nrca', 'nrcb', ):
        print(f'crowdsource {module}')
        merge_crowdsource(module=module)
        print(f'crowdsource unweighted {module}')
        merge_crowdsource(module=module, suffix='_unweighted')
        for suffix in ("_nsky0", "_nsky1", ):#"_nsky15"):
            print(f'crowdsource {suffix} {module}')
            merge_crowdsource(module=module, suffix=suffix)
        try:
            print(f'daophot basic {module}')
            merge_daophot(daophot_type='basic', module=module)
        except Exception as ex:
            print(ex)
        try:
            print(f'daophot iterative {module}')
            merge_daophot(daophot_type='iterative', module=module)
        except Exception as ex:
            print(ex)
        print()

if __name__ == "__main__":
    main()
