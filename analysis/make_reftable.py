import datetime
import warnings

from astropy import units as u
from astropy.table import Table

from astropy.coordinates import SkyCoord
from astroquery.svo_fps import SvoFps
from astropy import wcs
from astropy.io import fits
import numpy as np
import sys

from astropy import stats

try:
    from measure_offsets import measure_offsets
except ImportError:
    sys.path.append('/blue/adamginsburg/adamginsburg/jwst/brick/offsets')
    from measure_offsets import measure_offsets


def main(return_method=None, return_filter=None):
    """
    June 28, 2023: decided to switch to F405N-only reference
    """
    basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'

    for filtername, vvvcatfn in zip(('f405n', 'f200w', 'f444w'),
                                    ('F405N/pipeline/jw02221-o001_t001_nircam_clear-f405n-merged_vvvcat.ecsv',
                                     'catalogs/jw01182_VVV_reference_catalog.ecsv',
                                     'catalogs/jw01182_VVV_reference_catalog.ecsv')):
        print()
        print(filtername)
        for tblfilename, method in zip((f'{basepath}/catalogs/{filtername}_merged_indivexp_merged_crowdsource_nsky0.fits',
                                        f'{basepath}/catalogs/{filtername}_merged_indivexp_merged_dao_basic.fits'),
                                       ('crowdsource', 'dao_basic')):
            print(method, tblfilename)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                tbl = Table.read(tblfilename)

            if 'qf' in tbl.colnames:
                flux_colname = 'flux'
                sel = ((tbl['qf'] > 0.95) & (tbl['spread_model'] < 0.25) & (tbl['fracflux'] > 0.9) & (tbl[flux_colname] > 0))
            elif 'qfit' in tbl.colnames:
                flux_colname = 'flux_fit'
                sel = ((tbl['qfit'] < 0.4) & (tbl['cfit'] < 0.4) & (tbl[flux_colname] > 0)) & (np.isfinite(tbl['skycoord'].ra) & np.isfinite(tbl['skycoord'].dec))

            print(f"QFs are good for {sel.sum()} out of {len(tbl)} catalog entries")
            print(f"Making the reference catalog from {sel.sum()} out of {len(tbl)} catalog entries")

            reftbl = tbl['skycoord', flux_colname][sel]
            refcrds = reftbl['skycoord']
            assert not np.any(np.isnan(refcrds.ra))

            # Crossmatch to VVV and recenter
            vvvtb = Table.read(f'{basepath}/{vvvcatfn}')
            vvvtb = vvvtb[np.isfinite(vvvtb['RAJ2000']) & np.isfinite(vvvtb['DEJ2000'])]
            vvvcrds = SkyCoord(vvvtb['RAJ2000'].quantity, vvvtb['DEJ2000'].quantity, frame='fk5')

            jfilts = SvoFps.get_filter_list('JWST')
            jfilts.add_index('filterID')
            zeropoint = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername.upper()}']['ZeroPoint'], u.Jy)
            # sqpixscale = wcs.WCS(fits.getheader(reftbl.meta['FILENAME'], ext=1)).proj_plane_pixel_area()
            flux_jy = (reftbl[flux_colname] * u.MJy/u.sr * (reftbl.meta['pixscale_as']*u.arcsec)**2).to(u.Jy)
            mag_jw = -2.5 * np.log10(flux_jy / zeropoint) * u.mag

            (total_dra, total_ddec, med_dra, med_ddec, std_dra,
             std_ddec, keep, skykeep, reject, iteration) = measure_offsets(reference_coordinates=vvvcrds,
                                                                           skycrds_cat=refcrds,
                                                                           refflux=10**vvvtb['Ksmag3'],
                                                                           skyflux=10**mag_jw.value,
                                                                           sel=slice(None),
                                                                           verbose=True,
                                                                           )

            refcrds_updated = SkyCoord(refcrds.ra + total_dra, refcrds.dec + total_ddec, frame=refcrds.frame)

            reftbl['VVV_matched'] = skykeep

            print(f"Shifted {filtername} coordinates by {total_dra}, {total_ddec} in {iteration} iterations with stddev = {std_dra}, {std_ddec}"
                  f" ({(std_dra**2+std_ddec**2)**0.5})")
            print(f"Total of {keep.sum()} stars were used.")
            reftbl['skycoord'] = refcrds_updated

            # include two columns to make it a table, plus abmag for sorting
            reftbl['RA'] = reftbl['skycoord'].ra
            reftbl['DEC'] = reftbl['skycoord'].dec
            reftbl.sort(flux_colname, reverse=True) # descending

            reftbl.meta['VERSION'] = datetime.datetime.now().isoformat()
            if 'VERSION' in tbl.meta:
                reftbl.meta['PARENT_VERSION'] = tbl.meta['VERSION']
                reftbl.meta['RAOFFSET'] = total_dra
                reftbl.meta['DECOFFSET'] = total_ddec

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                reftbl.write(f'{basepath}/catalogs/{method}_based_nircam-{filtername}_reference_astrometric_catalog.ecsv', overwrite=True)
                reftbl.write(f'{basepath}/catalogs/{method}_based_nircam-{filtername}_reference_astrometric_catalog.fits', overwrite=True)

            if return_method == method:
                return reftbl
        if return_filter == filtername:
            return reftbl
    return reftbl


def main_old():
    basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'
    long_filternames = ['f410m', 'f405n', 'f466n']

    # filtername = 'F410M'
    # module = 'merged'
    # tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_nsky0.fits"

    # May 19, 2023: changed this to 'merged' b/c we can't keep going on with half a field; the workflow
    # relies on having a common catalog for both!
    # June 24, 2023: changed to merged-reproject, which worked, while merged did not.
    tblfilename = (f'{basepath}/catalogs/crowdsource_nsky0_merged-reproject_photometry_tables_merged.fits')
    tbl = Table.read(tblfilename)

    # reject sources with nondetections in F405N or F466N or bad matches
    sel = tbl['sep_f466n'].quantity < 0.1*u.arcsec
    sel &= tbl['sep_f405n'].quantity < 0.1*u.arcsec

    # reject sources with bad QFs
    goodqflong = ((tbl['qf_f410m'] > 0.90) |
                (tbl['qf_f405n'] > 0.90) |
                (tbl['qf_f466n'] > 0.90))
    goodspreadlong = ((tbl['spread_model_f410m'] < 0.25) |
                    (tbl['spread_model_f405n'] < 0.25) |
                    (tbl['spread_model_f466n'] < 0.25))
    goodfracfluxlong = ((tbl['fracflux_f410m'] > 0.8) |
                        (tbl['fracflux_f405n'] > 0.8) &
                        (tbl['fracflux_f466n'] > 0.8))

    any_saturated_ = [(tbl[f'near_saturated_{x}_{x}'] &
                    ~tbl[f'flux_{x}'].mask)
                    for x in long_filternames]
    any_saturated = any_saturated_[0]
    for col in any_saturated_[1:]:
        any_saturated = any_saturated | col

    any_replaced_saturated_ = [tbl[f'replaced_saturated_{x}'] &
                            ~tbl[f'flux_{x}'].mask for x in long_filternames]
    any_replaced_saturated = any_replaced_saturated_[0]
    for col in any_replaced_saturated_[1:]:
        any_replaced_saturated = any_replaced_saturated | col


    sel &= goodqflong & goodspreadlong & goodfracfluxlong
    print(f"QFs are good for {sel.sum()} out of {len(tbl)} catalog entries")
    print(f"Rejecting {(any_replaced_saturated & sel).sum()} replaced-saturated sources.")
    sel &= ~any_replaced_saturated
    print(f"Rejecting {(any_saturated & sel).sum()} near-saturated sources.")
    sel &= ~any_saturated
    print(f"Making the reference catalog from {sel.sum()} out of {len(tbl)} catalog entries")

    # include two columns to make it a table, plus abmag for sorting
    reftbl = tbl['skycoord_f410m', 'skycoord_f405n', 'mag_ab_f410m' ][sel]
    reftbl['RA'] = reftbl['skycoord_f410m'].ra
    reftbl['DEC'] = reftbl['skycoord_f410m'].dec
    reftbl.sort('mag_ab_f410m')

    reftbl.meta['VERSION'] = datetime.datetime.now().isoformat()
    reftbl.meta['PARENT_VERSION'] = tbl.meta['VERSION']

    reftbl.write(f'{basepath}/catalogs/{method}_based_nircam-long_reference_astrometric_catalog.ecsv', overwrite=True)
    reftbl.write(f'{basepath}/catalogs/{method}_based_nircam-long_reference_astrometric_catalog.fits', overwrite=True)

    return reftbl

if __name__ == "__main__":
    reftbl = main()
