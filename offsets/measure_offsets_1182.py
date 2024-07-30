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
from measure_offsets import measure_offsets

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


for reftbfn, reftbname in (
                           (f'{basepath}/catalogs/crowdsource_based_nircam-f405n_reference_astrometric_catalog.fits', 'F405ref'),
                           (vvvfn, 'VVV'),
                           ('/blue/adamginsburg/adamginsburg/jwst/brick//catalogs/f200w_merged_indivexp_merged_dao_basic.fits', 'F200ref'),
                           ):
    print()
    print(reftbname)
    reftb = Table.read(reftbfn)
    reference_coordinates = reftb['skycoord']

    if reftbname == 'VVV':
        vvv_reference_coordinates = reference_coordinates = reftb_vvv['skycoord']

        f200tb = Table.read(f'{basepath}/catalogs/jw01182-o004_t001_nircam_clear-f200w-merged_cat_20240302.ecsv')
        mag200 = f200tb['aper_total_vegamag']
        skycrds_f200 = f200tb['sky_centroid']

        idx, sidx, sep, sep3d = vvv_reference_coordinates.search_around_sky(skycrds_f200, 0.5*u.arcsec)
        idx_sel = np.isin(np.arange(len(f200tb)), idx)
        # dra = (skycrds_f200[idx].ra - vvv_reference_coordinates[sidx].ra).to(u.arcsec)
        # ddec = (skycrds_f200[idx].dec - vvv_reference_coordinates[sidx].dec).to(u.arcsec)
        closeneighbors_idx, closest_sep, _ = skycrds_f200.match_to_catalog_sky(skycrds_f200, 2)

        # select only sources close enough in magnitude
        magmatch = np.abs(reftb_vvv['Ksmag3'][sidx] - mag200[idx]) < 0.5

        # ignore sources where there are multiple JWST sources close to each other (confusion)
        # I dropped this from 0.5 (VVV size) to 0.2 b/c magmatch should help and I wanted more sources
        not_closesel = (closest_sep > 0.2*u.arcsec)

        sel = (not_closesel[idx]) & magmatch
        print(f"Selected {sel.sum()} reference source matching between VVV & F200W", flush=True)

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

    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for filtername in 'F444W,F200W,F356W,F115W'.split(","):
            for visit in ('002', '001'):
                print(f"Working on filter {filtername} visit {visit} with reference table {reftbname}")
                print(f"{'filt':5s}, {'ab':3s}, {'expno':5s}, {'ttl_dra':8s}, {'ttl_ddec':8s}, {'med_dra':8s}, {'med_ddec':8s}, {'std_dra':8s}, {'std_dec':8s}, nmatch, nreject, niter")
                # pipeline/tweakreg version globstr = f"{basepath}/{filtername}/pipeline/jw{project_id}{obsid}{visit}_*nrc*destreak_cat.fits"
                # F405N/f405n_nrcb_visit001_exp00008_crowdsource_nsky0.fits
                # globstr = f"{basepath}/{filtername}/{filtername.lower()}_*visit{visit}*_exp*_crowdsource_nsky0.fits"
                globstr = f"{basepath}/{filtername}/{filtername.lower()}_*visit{visit}_exp*_daophot_basic.fits"

                flist = sorted([x for x in glob.glob(globstr) if 'fitpsf' not in x])

                if len(flist) == 0:
                    raise ValueError(f"No matches to {globstr}")
                for fn in sorted(flist):

                    ab = 'a' if 'nrca' in fn else 'b'
                    if filtername in ('F444W', 'F356W'):
                        module = f'nrc{ab}long'
                    else:
                        ab += fn.split('nrc')[1][1]
                        module = f'nrc{ab}'

                    expno = fn.split("_")[3][-5:]
                    # old version visitname = os.path.basename(fn).split("_")[2][-3:]
                    visitname = f'jw01182004{visit}'

                    cat = Table.read(fn)
                    fitsfn = cat.meta['FILENAME']
                    ffh = fits.open(fitsfn)

                    if 'qf' in cat.colnames:
                        sel = cat['qf'] > 0.95
                        sel &= cat['fracflux'] > 0.8
                        cat = cat[sel]
                    elif 'qfit' in cat.colnames:
                        sel = cat['qfit'] < 0.4
                        sel &= cat['cfit'] < 0.4
                        cat = cat[sel]

                    header = ffh['SCI'].header

                    ww = WCS(header)

                    if 'RAOFFSET' in header:
                        raoffset = u.Quantity(header['RAOFFSET'], u.arcsec)
                        decoffset = u.Quantity(header['DEOFFSET'], u.arcsec)
                        # print(f"Found RAOFFSET in header: {raoffset}, {decoffset}")
                        header['CRVAL1'] = header['OLCRVAL1']
                        header['CRVAL2'] = header['OLCRVAL2']
                    elif handmeasured_offsets is not None:
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
                        raoffset = dra_hand
                        decoffset = ddec_hand
                    else:
                        raoffset = 0*u.arcsec
                        decoffset = 0*u.arcsec

                    if 'x' in cat.colnames:
                        skycrds_cat = ww.pixel_to_world(cat['x'], cat['y'])
                    elif 'xcentroid' in cat.colnames:
                        skycrds_cat = ww.pixel_to_world(cat['xcentroid'], cat['ycentroid'])
                    elif 'x_fit' in cat.colnames:
                        skycrds_cat = ww.pixel_to_world(cat['x_fit'], cat['y_fit'])

                    flux_colname = 'flux' if 'flux' in cat.colnames else 'flux_fit'

                    (total_dra, total_ddec, med_dra, med_ddec,
                     std_dra, std_ddec, keep, skykeep, reject, iteration) = measure_offsets(reference_coordinates,
                                                                                            skycrds_cat,
                                                                                            refflux=reftb['flux'],
                                                                                            skyflux=cat[flux_colname],
                                                                                            sel=slice(None),
                                                                                            verbose=True,
                                                                                            filtername=filtername,
                                                                                            ab=ab,
                                                                                            expno=expno,
                                                                                            total_dra=raoffset,
                                                                                            total_ddec=decoffset,
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
                        'Visit': visitname,
                        'visitnumber': visit,
                        'obsid': obsid,
                        # "Group": os.path.basename(fn).split("_")[1],
                        "Exposure": int(expno),
                        "Filter": filtername,
                        "Module": module,
                        'ab': ab.strip('1234long'),
                        "nmatch": keep.sum(),
                        'RAOFF_hdr': raoffset,
                        'DECOFF_hdr': decoffset,
                    })

    tbl = Table(rows)
    tbl.write(f"{basepath}/offsets/Offsets_JWST_Brick1182_{reftbname}.csv", format='ascii.csv', overwrite=True)

    gr = tbl.group_by(['Filter', 'Module', 'Visit'])
    agg = gr.groups.aggregate(np.mean)
    del agg['Exposure']
    aggstd = gr.groups.aggregate(np.std)
    aggmed = gr.groups.aggregate(np.median)
    aggsum = gr.groups.aggregate(np.sum)
    agg['dra_rms'] = aggstd['dra']
    agg['ddec_rms'] = aggstd['ddec']
    agg['dra_med'] = aggmed['dra']
    agg['ddec_med'] = aggmed['ddec']
    agg['nmatch_sum'] = aggsum['nmatch']
    agg.write(f"{basepath}/offsets/Offsets_JWST_Brick1182_{reftbname}_average.csv", format='ascii.csv', overwrite=True)

    gr = tbl.group_by(['Filter', 'Visit'])
    agg = gr.groups.aggregate(np.mean)
    del agg['Exposure']
    aggstd = gr.groups.aggregate(np.std)
    aggmed = gr.groups.aggregate(np.median)
    aggsum = gr.groups.aggregate(np.sum)
    agg['dra_rms'] = aggstd['dra']
    agg['ddec_rms'] = aggstd['ddec']
    agg['dra_med'] = aggmed['dra']
    agg['ddec_med'] = aggmed['ddec']
    agg['nmatch_sum'] = aggsum['nmatch']
    agg.write(f"{basepath}/offsets/Offsets_JWST_Brick1182_{reftbname}_average_lockmodules.cs/std_v", format='ascii.csv', overwrite=True)
