
import numpy as np


specfn = '/orange/adamginsburg/jwst/cloudc/spectra/J110621_JWST_NIRSpec_FS_spectrum_McClure23.txt'

from astropy.table import Table
from astropy import units as u
spectable = Table.read(specfn, format='ascii', comment=';',)
spectable.rename_column('col1', 'wavelength')
spectable['wavelength'].unit = u.um
spectable.rename_column('col2', 'flux')
spectable['flux'].unit = u.Jy


from icemodels import fluxes_in_filters

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')

# filter_ids = ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W', 
#              'JWST/NIRCam.F444W', 'JWST/NIRCam.F405N']
filter_ids = [fid for fid in jfilts['filterID'] if fid.startswith('JWST/NIRCam')]
filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in filter_ids}
transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}

iceage_flxd = fluxes_in_filters(spectable['wavelength'], spectable['flux'], filterids=filter_ids, transdata=transdata)
iceage_mags = {key: -2.5*np.log10(iceage_flxd[key] / filter_data[key]) for key in iceage_flxd}