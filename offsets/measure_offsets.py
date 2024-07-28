
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

    iteration = 0
    while np.abs(med_dra) > threshold or np.abs(med_ddec) > threshold:

        idx, offset, _ = reference_coordinates.match_to_catalog_sky(skycrds_cat[sel], nthneighbor=1)
        reverse_idx, reverse_sep, _ = skycrds_cat[sel].match_to_catalog_sky(reference_coordinates, nthneighbor=1)

        reverse_mutual_matches = (idx[reverse_idx] == np.arange(len(reverse_idx))) & (reverse_sep < max_offset)
        mutual_matches = (reverse_idx[idx] == np.arange(len(idx)))

        keep = (offset < max_offset) & mutual_matches
        skykeep = (reverse_sep < max_offset) & reverse_mutual_matches

        ratio = skyflux[idx[keep]] / refflux[keep]

        reject = np.zeros(ratio.size, dtype='bool')
        ii = 0
        if ratio_match:
            for ii in range(4):
                madstd = stats.mad_std(ratio[~reject])
                med = np.median(ratio[~reject])
                reject = (ratio < med - 5 * madstd) | (ratio > med + 5 * madstd) | reject
                ratio = 1 / ratio
                madstd = stats.mad_std(ratio[~reject])
                med = np.median(ratio[~reject])
                reject = (ratio < med - 5 * madstd) | (ratio > med + 5 * madstd) | reject
                ratio = 1 / ratio

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
            # print(f'len(sidx) = {len(sidx)}')
            raise ValueError(f"median(dra) = {med_dra}.  np.nanmedian(dra) = {np.nanmedian(dra)}")

        total_dra = total_dra + med_dra.to(u.arcsec)
        total_ddec = total_ddec + med_ddec.to(u.arcsec)

        skycrds_cat = SkyCoord(ra=skycrds_cat.ra + med_dra, dec=skycrds_cat.dec + med_ddec, frame=skycrds_cat.frame)

        iteration += 1
        if iteration > 50:
            break # there is at least one case in which we converged to an oscillator
            raise ValueError("Iteration is not converging")

    if verbose:
        print(f"{filtername:5s}, {ab:3s}, {expno:5s}, {total_dra.value:8.3f}, {total_ddec.value:8.3f}, {med_dra.value:8.3f}, {med_ddec.value:8.3f}, {std_dra.value:8.3f}, {std_ddec.value:8.3f}, {keep.sum():6d}, {reject.sum():7d}, niter={iteration:5d}", flush=True)

    return total_dra, total_ddec, med_dra, med_ddec, std_dra, std_ddec, keep, skykeep, reject, iteration
