import datetime

from astropy import units as u
from astropy.table import Table

def main():
    basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'

    # filtername = 'F410M'
    # module = 'merged'
    # tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_nsky0.fits"
    tblfilename = (f'{basepath}/catalogs/crowdsource_nsky0_nrca_photometry_tables_merged.fits')
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

    sel &= goodqflong & goodspreadlong & goodfracfluxlong
    print(f"Making the reference catalog from {sel.sum()} out of {len(tbl)} catalog entries")

    # include two columns to make it a table
    reftbl = tbl['skycoord_f410m', 'skycoord_f405n', ][sel]
    reftbl['RA'] = reftbl['skycoord_f410m'].ra
    reftbl['DEC'] = reftbl['skycoord_f410m'].dec

    reftbl.meta['VERSION'] = datetime.datetime.now().isoformat()
    reftbl.meta['PARENT_VERSION'] = tbl.meta['VERSION']

    reftbl.write(f'{basepath}/catalogs/crowdsource_based_nircam-long_reference_astrometric_catalog.ecsv', overwrite=True)

if __name__ == "__main__":
    main()
