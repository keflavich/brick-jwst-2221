import datetime
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

    reftbl = tbl['skycoord_f410m']
    reftbl['RA'] = reftbl['skycoord_f410m'].ra
    reftbl['DEC'] = reftbl['skycoord_f410m'].dec

    reftbl.meta['version'] = datetime.datetime.now().isoformat()
    reftbl.meta['parent_version'] = tbl.meta['version']

    reftbl.write(f'{basepath}/catalogs/crowdsource_based_nircam-long_reference_astrometric_catalog.ecsv', overwrite=True)

if __name__ == "__main__":
    main()