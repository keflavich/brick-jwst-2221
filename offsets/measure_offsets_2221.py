import numpy as np
import warnings
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy import stats
import glob
import os

project_id = '02221'
field = '001'
obsid = '001'
visit = '001'

#reftb = Table.read("catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv")
#reference_coordinates = reftb['skycoord']

# match selection to VVV by checking that K-band mags match and avoiding multiple stars
# see AlignmentDebugViz for the development work
reftb_vvv = Table.read('F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-merged_vvvcat.ecsv')
vvv_reference_coordinates = reference_coordinates = reftb_vvv['skycoord']

f212tb = Table.read('/orange/adamginsburg/jwst/brick/catalogs/jw02221-o001_t001_nircam_clear-f212n-merged_cat_20240301.ecsv')
mag212 = f212tb['aper_total_vegamag']
skycrds_f212 = f212tb['sky_centroid']

idx, sidx, sep, sep3d = vvv_reference_coordinates.search_around_sky(skycrds_f212, 0.5*u.arcsec)
idx_sel = np.isin(np.arange(len(f212tb)), idx)
dra = (skycrds_f212[idx].ra - vvv_reference_coordinates[sidx].ra).to(u.arcsec)
ddec = (skycrds_f212[idx].dec - vvv_reference_coordinates[sidx].dec).to(u.arcsec)
closeneighbors_idx, closest_sep, _ = skycrds_f212.match_to_catalog_sky(skycrds_f212, 2)

# select only sources close enough in magnitude
magmatch = np.abs(reftb_vvv['Ksmag3'][sidx] - mag212[idx]) < 0.5

# ignore sources where there are multiple JWST sources close to each other (confusion)
not_closesel = (closest_sep > 0.5*u.arcsec)

sel = (not_closesel[idx]) & magmatch

# downselect to only the coordinates we expect to have good matches
reference_coordinates = vvv_reference_coordinates[sidx][sel]
print(f"There are {len(reference_coordinates)} reference coordinates out of {len(reftb_vvv)} in the reference catalog.")

# write out downselected version so we can overplot it in CARTA
reftb_vvv[sidx][sel].write('/orange/adamginsburg/jwst/brick/catalogs/jw02221-o001_t001_nircam_clear-f212n-merged_cat_20240301_downsel.fits',
                       overwrite=True)


if True:

    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for filtername in 'F466N,F405N,F410M,F212N,F182M,F187N'.split(","):
            for fn in sorted(glob.glob(f"{filtername}/pipeline/jw{project_id}{obsid}{visit}_*nrc*destreak_cat.fits")):
                ab = 'a' if 'nrca' in fn else 'b'
                module = f'nrc{ab}long' if 'long' in fn else f'nrc{ab}' + fn.split('nrc')[1][1]
                expno = fn.split("_")[2]
                cat = ftb = Table.read(fn)
                try:
                    fitsfn = fn.replace("_cat.fits", ".fits")
                    ffh = fits.open(fitsfn)
                except FileNotFoundError:
                    fitsfn = fn.replace("destreak_cat.fits", "cal.fits")
                    ffh = fits.open(fitsfn)

                header = ffh['SCI'].header

                if 'RAOFFSET' in header:
                    raoffset = header['RAOFFSET']
                    decoffset = header['DEOFFSET']
                    header['CRVAL1'] = header['OLCRVAL1']
                    header['CRVAL2'] = header['OLCRVAL2']

                ww = WCS(header)

                skycrds_cat = ww.pixel_to_world(cat['x'], cat['y'])

                sel = slice(None)
                max_offset = 0.2*u.arcsec
                idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat[sel], max_offset)
                dra = (skycrds_cat[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
                ddec = (skycrds_cat[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)


                med_dra = 100*u.arcsec
                med_ddec = 100*u.arcsec
                threshold = 0.01*u.arcsec
                max_offset = 0.5*u.arcsec

                total_dra = 0*u.arcsec
                total_ddec = 0*u.arcsec

                iteration = 0
                while np.abs(med_dra) > threshold or np.abs(med_ddec) > threshold:
                    skycrds_cat = ww.pixel_to_world(cat['x'], cat['y'])

                    idx, offset, _ = reference_coordinates.match_to_catalog_sky(skycrds_cat[sel])
                    keep = offset < max_offset

                    reject = False
                    # ratio = cat['flux'][idx[keep]] / reftb['flux'][keep]
                    # reject = np.zeros(ratio.size, dtype='bool')
                    # for ii in range(4):
                    #     madstd = stats.mad_std(ratio[~reject])
                    #     med = np.median(ratio[~reject])
                    #     reject = (ratio < med - 3 * madstd) | (ratio > med + 3 * madstd) | reject
                    #     ratio = 1 / ratio
                    #     madstd = stats.mad_std(ratio[~reject])
                    #     med = np.median(ratio[~reject])
                    #     reject = (ratio < med - 3 * madstd) | (ratio > med + 3 * madstd) | reject
                    #     ratio = 1 / ratio


                    #idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat[sel], max_offset)
                    #dra = (skycrds_cat[sel][idx[keep]].ra - reference_coordinates[keep].ra).to(u.arcsec)
                    #ddec = (skycrds_cat[sel][idx[keep]].dec - reference_coordinates[keep].dec).to(u.arcsec)
                    dra = (skycrds_cat[sel][idx[keep][~reject]].ra - reference_coordinates[keep][~reject].ra).to(u.arcsec)
                    ddec = (skycrds_cat[sel][idx[keep][~reject]].dec - reference_coordinates[keep][~reject].dec).to(u.arcsec)

                    med_dra = np.median(dra)
                    med_ddec = np.median(ddec)
                    std_dra = stats.mad_std(dra)
                    std_ddec = stats.mad_std(ddec)

                    iteration += 1

                    if np.isnan(med_dra):
                        print(f'len(refcoords) = {len(reference_coordinates)}')
                        print(f'len(cat) = {len(cat)}')
                        print(f'len(idx) = {len(idx)}')
                        print(f'len(sidx) = {len(sidx)}')
                        print(cat, sel, idx, sidx, sep)
                        raise ValueError(f"median(dra) = {med_dra}.  np.nanmedian(dra) = {np.nanmedian(dra)}")

                    total_dra = total_dra + med_dra.to(u.arcsec)
                    total_ddec = total_ddec + med_ddec.to(u.arcsec)

                    ww.wcs.crval = ww.wcs.crval - [med_dra.to(u.deg).value, med_ddec.to(u.deg).value]


                print(f"{filtername:5s}, {ab}, {expno}, {total_dra:8.3f}, {total_ddec:8.3f}, {med_dra:8.3f}, {med_ddec:8.3f}, nmatch={keep.sum()} niter={iteration}")
                if keep.sum() < 5:
                    print(fitsfn)
                    print(fn)
                    raise

                rows.append({
                    'Filename': fn,
                    #'Test': os.path.basename(fn),
                    #'Filename_1': os.path.basename(fn),
                    'dra': total_dra,
                    'ddec': total_ddec,
                    'dra (arcsec)': total_dra,
                    'ddec (arcsec)': total_ddec,
                    'dra_std': std_dra,
                    'ddec_std': std_ddec,
                    'Visit': os.path.basename(fn).split("_")[0],
                    'obsid': obsid,
                    "Group": os.path.basename(fn).split("_")[1],
                    "Exposure": int(expno),
                    "Filter": filtername,
                    "Module": module
                })

    tbl = Table(rows)
    # don't necessarily want to write this: if the manual alignment has been run already, the offsets will all be zero
    tbl.write("offsets/Offsets_JWST_Brick2221_VVV.csv", format='ascii.csv', overwrite=True)

    # TODO: aggregate with weighted mean

    gr = tbl.group_by(['Filter', 'Module'])
    agg = gr.groups.aggregate(np.mean)
    del agg['Exposure']
    aggstd = gr.groups.aggregate(np.std)
    aggmed = gr.groups.aggregate(np.median)
    agg['dra_rms'] = aggstd['dra']
    agg['ddec_rms'] = aggstd['ddec']
    agg['dra_med'] = aggmed['dra']
    agg['ddec_med'] = aggmed['ddec']
    agg.write("offsets/Offsets_JWST_Brick2221_VVV_average.csv", format='ascii.csv', overwrite=True)
