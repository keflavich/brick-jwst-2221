
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


def measure_offsets(reference_coordinates, skycrds_cat, refflux, skyflux, total_dra=0*u.arcsec,
                    total_ddec=0*u.arcsec, max_offset=0.2*u.arcsec, threshold=0.01*u.arcsec,
                    sel=slice(None),
                    verbose=False,
                    ratio_match=True,
                    nsigma_reject=5,
                    reject_niter=7,
                    filtername='', ab='', expno=''):
    med_dra = 100*u.arcsec
    med_ddec = 100*u.arcsec

    idx, sep, _ = reference_coordinates.match_to_catalog_sky(skycrds_cat[sel], nthneighbor=1)
    reverse_idx, reverse_sep, _ = skycrds_cat[sel].match_to_catalog_sky(reference_coordinates, nthneighbor=1)
    reverse_mutual_matches = (idx[reverse_idx] == np.arange(len(reverse_idx))) & (reverse_sep < max_offset)
    mutual_matches = (reverse_idx[idx] == np.arange(len(idx))) & (sep < max_offset)
    # print(f"Matched {mutual_matches.sum()} mutually of {(sep < max_offset).sum()} total")

    dra = -(skycrds_cat[sel][idx[mutual_matches]].ra - reference_coordinates[reverse_idx[reverse_mutual_matches]].ra).to(u.arcsec)
    ddec = -(skycrds_cat[sel][idx[mutual_matches]].dec - reference_coordinates[reverse_idx[reverse_mutual_matches]].dec).to(u.arcsec)

    success = False

    iteration = 0
    while np.abs(med_dra) > threshold or np.abs(med_ddec) > threshold:

        idx, offset, _ = reference_coordinates.match_to_catalog_sky(skycrds_cat[sel], nthneighbor=1)
        reverse_idx, reverse_sep, _ = skycrds_cat[sel].match_to_catalog_sky(reference_coordinates, nthneighbor=1)

        reverse_mutual_matches = (idx[reverse_idx] == np.arange(len(reverse_idx))) & (reverse_sep < max_offset)
        mutual_matches = (reverse_idx[idx] == np.arange(len(idx)))

        keep = (offset < max_offset) & mutual_matches
        skykeep = (reverse_sep < max_offset) & reverse_mutual_matches
        if keep.sum() < 5:
            print(f"Only {keep.sum()} sources matched - this is too few to be useful")
            print(f"{filtername:5s}, {ab:3s}, {expno:5s}, {keep.sum():6d}, {iteration:5d}", flush=True)
            break

        # ratio = skyflux[idx[keep]] / refflux[keep]
        # magnitude-style
        ratio = np.log(skyflux[idx[keep]]) - np.log(refflux[keep])

        reject = np.zeros(ratio.size, dtype='bool')
        if ratio_match:
            rejection_data = []
            for ii in range(reject_niter):
                madstd = stats.mad_std(ratio[~reject], ignore_nan=True)
                med = np.nanmedian(ratio[~reject])
                reject = (ratio < (med - nsigma_reject * madstd)) | (ratio > (med + nsigma_reject * madstd)) | reject
                rejection_data.append([med, madstd, reject.sum()])
            if np.all(reject):
                print("ALL SOURCES WERE REJECTED - this isn't really possible so it indicates an error")
                print(f"Iterations were: {rejection_data}")
                reject = np.zeros(ratio.size, dtype='bool')

        # dra and ddec should be the vector added to CRVAL to put the image in the right place
        dra = -(skycrds_cat[sel][idx[keep][~reject]].ra - reference_coordinates[keep][~reject].ra).to(u.arcsec)
        ddec = -(skycrds_cat[sel][idx[keep][~reject]].dec - reference_coordinates[keep][~reject].dec).to(u.arcsec)

        med_dra = np.median(dra)
        med_ddec = np.median(ddec)
        std_dra = stats.mad_std(dra)
        std_ddec = stats.mad_std(ddec)

        if np.isnan(med_dra):
            print(f'len(refcoords) = {len(reference_coordinates)}')
            print(f'len(idx) = {len(idx)}')
            print(f'keep.sum() = {keep.sum()}')
            print(f'reject.size: {reject.size}')
            print(f'reject.sum() = {reject.sum()}')
            print(f'~reject.sum() = {(~reject).sum()}')
            # print(f'len(sidx) = {len(sidx)}')
            raise ValueError(f"median(dra) = {med_dra}.  np.nanmedian(dra) = {np.nanmedian(dra)}")

        total_dra = total_dra + med_dra.to(u.arcsec)
        total_ddec = total_ddec + med_ddec.to(u.arcsec)

        skycrds_cat = SkyCoord(ra=skycrds_cat.ra + med_dra, dec=skycrds_cat.dec + med_ddec, frame=skycrds_cat.frame)

        success = True

        iteration += 1
        if iteration > 50:
            break # there is at least one case in which we converged to an oscillator
            raise ValueError("Iteration is not converging")

    if verbose and success:
        print(f"{filtername:5s}, {ab:3s}, {expno:5s}, {total_dra.value:8.3f}, {total_ddec.value:8.3f}, {med_dra.value:8.3f}, {med_ddec.value:8.3f}, {std_dra.value:8.3f}, {std_ddec.value:8.3f}, {keep.sum():6d}, {reject.sum():7d}, {iteration:5d}", flush=True)

    return total_dra, total_ddec, med_dra, med_ddec, std_dra, std_ddec, keep, skykeep, reject, iteration
