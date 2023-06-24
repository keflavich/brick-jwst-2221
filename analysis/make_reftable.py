import datetime

from astropy import units as u
from astropy.table import Table

def main():
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

    reftbl.write(f'{basepath}/catalogs/crowdsource_based_nircam-long_reference_astrometric_catalog.ecsv', overwrite=True)
    reftbl.write(f'{basepath}/catalogs/crowdsource_based_nircam-long_reference_astrometric_catalog.fits', overwrite=True)

    return reftbl

if __name__ == "__main__":
    reftbl = main()
