import numpy as np
import warnings
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy import stats
import regions
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import glob
import os

project_id = '01182'
field = '004'
obsid = '004'
visit = '002'

basepath = '/orange/adamginsburg/jwst/brick'

handmeasured_offsets = Table.read(f'{basepath}/offsets/Offsets_JWST_Brick1182.csv')


#reftb = Table.read(f"{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog_truncated10000.ecsv")
#reference_coordinates = reftb['skycoord']

# match selection to VVV by checking that K-band mags match and avoiding multiple stars
# see AlignmentDebugViz for the development work

vvvfn = f'{basepath}/catalogs/jw01182_VVV_reference_catalog.ecsv'
if os.path.exists(vvvfn):
    reftb_vvv = Table.read(vvvfn)
else:
    Vizier.ROW_LIMIT = 1e5
    coord = SkyCoord(266.534963671, -28.710074995, unit=(u.deg, u.deg), frame='fk5')
    reftb_vvv = Vizier.query_region(coordinates=coord,
                                 width=7*u.arcmin,
                                 height=7*u.arcmin,
                                 catalog=['II/348/vvv2'])[0]
    reftb_vvv['RA'] = reftb_vvv['RAJ2000']
    reftb_vvv['DEC'] = reftb_vvv['DEJ2000']

    # FK5 because it says 'J2000' on the Vizier page (same as twomass)
    reftb_vvv_crds = SkyCoord(reftb_vvv['RAJ2000'], reftb_vvv['DEJ2000'], frame='fk5')
    reftb_vvv['skycoord'] = reftb_vvv_crds

    reftb_vvv.write(vvvfn, overwrite=True)
    reftb_vvv.write(vvvfn.replace(".ecsv", ".fits"), overwrite=True)

vvv_reference_coordinates = reference_coordinates = reftb_vvv['skycoord']

f200tb = Table.read(f'{basepath}/catalogs/jw01182-o004_t001_nircam_clear-f200w-merged_cat_20240302.ecsv')
mag200 = f200tb['aper_total_vegamag']
skycrds_f200 = f200tb['sky_centroid']

idx, sidx, sep, sep3d = vvv_reference_coordinates.search_around_sky(skycrds_f200, 0.5*u.arcsec)
idx_sel = np.isin(np.arange(len(f200tb)), idx)
dra = (skycrds_f200[idx].ra - vvv_reference_coordinates[sidx].ra).to(u.arcsec)
ddec = (skycrds_f200[idx].dec - vvv_reference_coordinates[sidx].dec).to(u.arcsec)
closeneighbors_idx, closest_sep, _ = skycrds_f200.match_to_catalog_sky(skycrds_f200, 2)

# select only sources close enough in magnitude
magmatch = np.abs(reftb_vvv['Ksmag3'][sidx] - mag200[idx]) < 0.5

# ignore sources where there are multiple JWST sources close to each other (confusion)
not_closesel = (closest_sep > 0.5*u.arcsec)

sel = (not_closesel[idx]) & magmatch
print(f"Selected {sel.sum()} reference source matching between VVV & F200W")

# downselect to only the coordinates we expect to have good matches
reference_coordinates = vvv_reference_coordinates[sidx][sel]
print(f"There are {len(reference_coordinates)} reference coordinates out of {len(reftb_vvv)} in the reference catalog.")

# write out downselected version so we can overplot it in CARTA
reftb_vvv[sidx][sel].write('/orange/adamginsburg/jwst/brick/catalogs/jw01182-o004_t001_nircam_clear-f200w-merged_cat_20240302_downsel.fits',
                       overwrite=True)

# set 'flux' so we can compute ratio below.  Is arbitrary, but let's use VISTA values anyway
# http://svo2.cab.inta-csic.es/theory/fps/index.php?id=Paranal/VISTA.Ks&&mode=browse&gname=Paranal&gname2=VISTA#filter
reftb_vvv['flux'] = (10**(reftb_vvv['Ksmag3'] / 2.5) + 659.10)*u.Jy

# for use below, needs to match reference_coordinates
reftb = reftb_vvv[sidx][sel]
print(f"reftb has length {len(reftb)}")

# undo all that work above: just accept _all_ VVV sources (and handle rejection below?)
reftb = reftb_vvv
reference_coordinates = vvv_reference_coordinates
print(f"reftb has length {len(reftb)}")


if True:

    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for filtername in 'F200W,F356W,F444W,F115W'.split(","):
            for visit in ('002', '001'):
                print(f"Working on filter {filtername}")
                print(f"{'filt':5s}, {'ab':3s}, {'expno':5s}, {'ttl_dra':15s}, {'ttl_ddec':15s}, {'med_dra':15s}, {'med_ddec':15s}, {'std_dra':15s}, {'std_dec':15s}, nmatch, nreject, niter")
                globstr = f"{basepath}/{filtername}/pipeline/jw{project_id}{obsid}{visit}_*nrc*destreak_cat.fits"
                flist = glob.glob(globstr)

                if len(flist) == 0:
                    raise ValueError(f"No matches to {globstr}")
                for fn in sorted(flist):
                    ab = 'a' if 'nrca' in fn else 'b'
                    module = f'nrc{ab}long' if 'long' in fn else f'nrc{ab}' + fn.split('nrc')[1][1]
                    expno = fn.split("_")[2]
                    visitname = os.path.basename(fn).split("_")[0]

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
                        print(f"Found RAOFFSET in header: {raoffset}, {decoffset}")
                        header['CRVAL1'] = header['OLCRVAL1']
                        header['CRVAL2'] = header['OLCRVAL2']


                    ww = WCS(header)

                    # start by shifting by measured offsets
                    match = ((handmeasured_offsets['Visit'] == visitname) &
                             (handmeasured_offsets['Exposure'] == int(expno)) &
                             ((handmeasured_offsets['Module'] == module) | (handmeasured_offsets['Module'] == module.strip('1234'))) &
                             (handmeasured_offsets['Filter'] == filtername)
                            )
                    assert match.sum() == 1
                    handsel_row = handmeasured_offsets[match][0]
                    dra_hand, ddec_hand = u.Quantity([handsel_row['dra (arcsec)'], handsel_row['ddec (arcsec)']], u.arcsec)
                    ww.wcs.crval = ww.wcs.crval + [dra_hand.to(u.deg).value, ddec_hand.to(u.deg).value]

                    #print(fitsfn, fn)
                    #print(f"Shifted original WCS by {dra_hand}, {ddec_hand}")
                    total_dra = dra_hand.to(u.arcsec)
                    total_ddec = ddec_hand.to(u.arcsec)

                    skycrds_cat = ww.pixel_to_world(cat['x'], cat['y'])

                    max_offset = 0.2*u.arcsec
                    med_dra = 100*u.arcsec
                    med_ddec = 100*u.arcsec
                    threshold = 0.01*u.arcsec

                    sel = slice(None)

                    idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat[sel], max_offset)
                    dra = -(skycrds_cat[sel][idx].ra - reference_coordinates[sidx].ra).to(u.arcsec)
                    ddec = -(skycrds_cat[sel][idx].dec - reference_coordinates[sidx].dec).to(u.arcsec)

                    iteration = 0
                    while np.abs(med_dra) > threshold or np.abs(med_ddec) > threshold:
                        skycrds_cat = ww.pixel_to_world(cat['x'], cat['y'])

                        idx, offset, _ = reference_coordinates.match_to_catalog_sky(skycrds_cat[sel])
                        keep = offset < max_offset

                        ratio = cat['flux'][idx[keep]] / reftb['flux'][keep]
                        reject = np.zeros(ratio.size, dtype='bool')
                        ii = 0
                        # rejecting based on flux may have failed?
                        if filtername == 'F200W':
                            # for the other filters, we don't expect any agreement at all
                            for ii in range(4):
                                madstd = stats.mad_std(ratio[~reject])
                                med = np.median(ratio[~reject])
                                reject = (ratio < med - 5 * madstd) | (ratio > med + 5 * madstd) | reject
                                ratio = 1 / ratio
                                madstd = stats.mad_std(ratio[~reject])
                                med = np.median(ratio[~reject])
                                reject = (ratio < med - 5 * madstd) | (ratio > med + 5 * madstd) | reject
                                ratio = 1 / ratio


                        #idx, sidx, sep, sep3d = reference_coordinates.search_around_sky(skycrds_cat[sel], max_offset)
                        #dra = (skycrds_cat[sel][idx[keep]].ra - reference_coordinates[keep].ra).to(u.arcsec)
                        #ddec = (skycrds_cat[sel][idx[keep]].dec - reference_coordinates[keep].dec).to(u.arcsec)

                        # dra and ddec should be the vector added to CRVAL to put the image in the right place
                        dra = -(skycrds_cat[sel][idx[keep][~reject]].ra - reference_coordinates[keep][~reject].ra).to(u.arcsec)
                        ddec = -(skycrds_cat[sel][idx[keep][~reject]].dec - reference_coordinates[keep][~reject].dec).to(u.arcsec)

                        med_dra = np.median(dra)
                        med_ddec = np.median(ddec)
                        std_dra = stats.mad_std(dra)
                        std_ddec = stats.mad_std(ddec)

                        if np.isnan(med_dra):
                            print(f'len(refcoords) = {len(reference_coordinates)}')
                            print(f'len(cat) = {len(cat)}')
                            print(f'len(idx) = {len(idx)}')
                            print(f'len(sidx) = {len(sidx)}')
                            print(cat, sel, idx, sidx, sep)
                            raise ValueError(f"median(dra) = {med_dra}.  np.nanmedian(dra) = {np.nanmedian(dra)}")

                        total_dra = total_dra + med_dra.to(u.arcsec)
                        total_ddec = total_ddec + med_ddec.to(u.arcsec)

                        ww.wcs.crval = ww.wcs.crval + [med_dra.to(u.deg).value, med_ddec.to(u.deg).value]

                        iteration += 1
                        if iteration > 50:
                            break # there is at least one case in which we converged to an oscillator
                            raise ValueError("Iteration is not converging")



                    print(f"{filtername:5s}, {ab:3s}, {expno:5s}, {total_dra:8.3f}, {total_ddec:8.3f}, {med_dra:8.3f}, {med_ddec:8.3f}, {std_dra:8.3f}, {std_ddec:8.3f}, {keep.sum():6d}, {reject.sum():7d}, niter={iteration:5d} [dra_hand={dra_hand}, ddec_hand={ddec_hand}]")
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
                        'Visit': visitname,
                        'visitnumber': visit,
                        'obsid': obsid,
                        "Group": os.path.basename(fn).split("_")[1],
                        "Exposure": int(expno),
                        "Filter": filtername,
                        "Module": module
                    })

    tbl = Table(rows)
    # don't necessarily want to write this: if the manual alignment has been run already, the offsets will all be zero
    tbl.write(f"{basepath}/offsets/Offsets_JWST_Brick1182_VVV.csv", format='ascii.csv', overwrite=True)

    # TODO: aggregate with weighted mean

    gr = tbl.group_by(['Filter', 'Module', 'Visit'])
    agg = gr.groups.aggregate(np.mean)
    del agg['Exposure']
    aggstd = gr.groups.aggregate(np.std)
    aggmed = gr.groups.aggregate(np.median)
    agg['dra_rms'] = aggstd['dra']
    agg['ddec_rms'] = aggstd['ddec']
    agg['dra_med'] = aggmed['dra']
    agg['ddec_med'] = aggmed['ddec']
    agg.write(f"{basepath}/offsets/Offsets_JWST_Brick1182_VVV_average.csv", format='ascii.csv', overwrite=True)

