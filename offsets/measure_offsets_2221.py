import numpy as np
import warnings
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy import stats
import glob
import os
from astroquery.svo_fps import SvoFps

from measure_offsets import measure_offsets

project_id = '02221'
field = '001'
obsid = '001'
visit = '001'

basepath = '/blue/adamginsburg/adamginsburg/jwst/brick'

for reftbfn, reftbname in ((f'{basepath}/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-merged_vvvcat.ecsv', 'VVV'),
                           (f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.fits', 'F405ref'),
                           ):
    print()
    print(reftbname)
    reftb = Table.read(reftbfn)
    reference_coordinates = reftb['skycoord']

    # match selection to VVV by checking that K-band mags match and avoiding multiple stars
    # see AlignmentDebugViz for the development work

    if reftbname == 'VVV':
        reftb_vvv = reftb
        f212tb = Table.read(f'{basepath}/catalogs/f212n_merged_indivexp_merged_crowdsource_nsky0.fits')
        jfilts = SvoFps.get_filter_list('JWST')
        jfilts.add_index('filterID')
        zeropoint = u.Quantity(jfilts.loc[f'JWST/NIRCam.F212N']['ZeroPoint'], u.Jy)
        flux_jy = (f212tb['flux'] * u.MJy/u.sr * (f212tb.meta['pixscale_as']*u.arcsec)**2).to(u.Jy)
        mag212 = abmag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag

        # downselect to bright
        f212tb = f212tb[mag212 < 17 * u.mag]
        flux_jy = (f212tb['flux'] * u.MJy/u.sr * (f212tb.meta['pixscale_as']*u.arcsec)**2).to(u.Jy)
        mag212 = abmag = -2.5 * np.log10(flux_jy / zeropoint) * u.mag

        #mag212 = f212tb['aper_total_vegamag']
        #skycrds_f212 = f212tb['sky_centroid']
        skycrds_f212 = f212tb['skycoord']

        idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_f212, 0.5*u.arcsec)
        idx_sel = np.isin(np.arange(len(f212tb)), idx)
        dra = (skycrds_f212[idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
        ddec = (skycrds_f212[idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)
        closeneighbors_idx, closest_sep, _ = skycrds_f212.match_to_catalog_sky(skycrds_f212, 2)

        # select only sources close enough in magnitude
        magmatch = np.abs(reftb_vvv['Ksmag3'][sidx] - mag212[idx]) < 0.5

        # ignore sources where there are multiple JWST sources close to each other (confusion)
        # (closest_sep is crossmatched to the same catalog looking for the nearest nonself neighbor)
        not_closesel = (closest_sep > 0.5*u.arcsec)

        sel = (not_closesel[idx]) & magmatch

        # downselect to only the coordinates we expect to have good matches
        reference_coordinates = reference_coordinates[sidx][sel]
        print(f"There are {len(reference_coordinates)} reference coordinates out of {len(reftb_vvv)} in the reference catalog.")

        # write out downselected version so we can overplot it in CARTA
        reftb_vvv[sidx][sel].write(f'{basepath}/catalogs/jw02221-o001_t001_nircam_clear-f212n-merged_cat_VVV_downsel.fits', overwrite=True)

        # set 'flux' so we can compute ratio below.  Is arbitrary, but let's use VISTA values anyway
        # http://svo2.cab.inta-csic.es/theory/fps/index.php?id=Paranal/VISTA.Ks&&mode=browse&gname=Paranal&gname2=VISTA#filter
        reftb_vvv['flux'] = (10**(reftb_vvv['Ksmag3'] / 2.5) + 659.10)*u.Jy

        # for use below, needs to match reference_coordinates
        reftb = reftb_vvv[sidx][sel]

    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for filtername in 'F466N,F405N,F410M,F212N,F182M,F187N'.split(","):
            # old version cats = sorted(glob.glob(f"{basepath}/{filtername}/pipeline/jw{project_id}{obsid}{visit}_*nrc*destreak_cat.fits"))
            # F405N/f405n_nrcb_visit001_exp00008_crowdsource_nsky0.fits
            globstr = f"{basepath}/{filtername}/{filtername.lower()}_*visit*_exp*_crowdsource_nsky0.fits"
            cats = sorted(glob.glob(globstr))
            if len(cats) == 0:
                raise ValueError(f"No matches to {globstr}")
            cats = sorted([x for x in glob.glob(globstr) if 'fitpsf' not in x])

            print(f"{'filt':5s}, {'ab':3s}, {'expno':5s}, {'ttl_dra':15s}, {'ttl_ddec':15s}, {'med_dra':15s}, {'med_ddec':15s}, {'std_dra':15s}, {'std_dec':15s}, nmatch, nreject, niter")
            for fn in cats:
                ab = 'a' if 'nrca' in fn else 'b'
                if filtername in ('F466N', 'F405N', 'F410M'):
                    module = f'nrc{ab}long'
                else:
                    ab += fn.split('nrc')[1][1]
                    module = f'nrc{ab}'

                expno = fn.split("_")[3][-5:]
                visitnumber = os.path.basename(fn).split("_")[2][-3:]

                cat = Table.read(fn)
                fitsfn = cat.meta['FILENAME']
                ffh = fits.open(fitsfn)
                header = ffh['SCI'].header
                # try:
                #     fitsfn = fn.replace("_cat.fits", ".fits")
                #     ffh = fits.open(fitsfn)
                # except FileNotFoundError:
                #     fitsfn = fn.replace("destreak_cat.fits", "cal.fits")
                #     ffh = fits.open(fitsfn)

                if 'qf' in cat.colnames:
                    sel = cat['qf'] > 0.95
                    sel &= cat['fracflux'] > 0.8
                    cat = cat[sel]

                if 'RAOFFSET' in header:
                    raoffset = header['RAOFFSET']
                    decoffset = header['DEOFFSET']
                    header['CRVAL1'] = header['OLCRVAL1']
                    header['CRVAL2'] = header['OLCRVAL2']

                flux_colname = 'flux' if 'flux' in cat.colnames else 'flux_fit'

                skycrd_cat = cat['skycoord'] if 'skycoord' in cat.colnames else cat['sky_centroid']

                total_dra, total_ddec, med_dra, med_ddec, std_dra, std_ddec, keep, skykeep, reject, iteration = measure_offsets(reference_coordinates,
                                                                                                                       skycrd_cat,
                                                                                                                       max_offset=0.2*u.arcsec,
                                                                                                                       refflux=reftb['flux'],
                                                                                                                       skyflux=cat[flux_colname],
                                                                                                                       sel=slice(None),
                                                                                                                       verbose=True,
                                                                                                                       filtername=filtername,
                                                                                                                       ab=ab,
                                                                                                                       expno=expno,
                                                                                                                       )
                if keep.sum() < 5:
                    print(fitsfn)
                    print(fn)
                    raise

                rows.append({
                    'Filename': fn,
                    # 'Test': os.path.basename(fn),
                    # 'Filename_1': os.path.basename(fn),
                    'dra': total_dra,
                    'ddec': total_ddec,
                    'dra (arcsec)': total_dra,
                    'ddec (arcsec)': total_ddec,
                    'dra_std': std_dra,
                    'ddec_std': std_ddec,
                    'Visit': visitnumber,
                    'obsid': obsid,
                    # "Group": os.path.basename(fn).split("_")[1],
                    "Exposure": int(expno),
                    "Filter": filtername,
                    "Module": module,
                })

    tbl = Table(rows)
    tbl.write(f"{basepath}/offsets/Offsets_JWST_Brick2221_{reftbname}.csv", format='ascii.csv', overwrite=True)

    gr = tbl.group_by(['Filter', 'Module'])
    agg = gr.groups.aggregate(np.mean)
    del agg['Exposure']
    aggstd = gr.groups.aggregate(np.std)
    aggmed = gr.groups.aggregate(np.median)
    agg['dra_rms'] = aggstd['dra']
    agg['ddec_rms'] = aggstd['ddec']
    agg['dra_med'] = aggmed['dra']
    agg['ddec_med'] = aggmed['ddec']
    agg.write(f"{basepath}/offsets/Offsets_JWST_Brick2221_{reftbname}_average.csv", format='ascii.csv', overwrite=True)
