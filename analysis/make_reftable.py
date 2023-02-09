import datetime
from astropy.table import Table

basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'

filtername = 'F410M'
module = 'merged'
tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_nsky0.fits"
tbl = Table.read(tblfilename)
reftbl = tbl['skycoord']
reftbl['RA'] = reftbl['skycoord'].ra
reftbl['DEC'] = reftbl['skycoord'].dec

reftbl.meta['version'] = datetime.datetime.now().isoformat()

reftbl.write(f'{basepath}/catalogs/crowdsource_based_nircam-long_reference_astrometric_catalog.ecsv', overwrite=True)