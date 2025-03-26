isodir = '/orange/adamginsburg/ice/iso/library/swsatlas/'

import pyspeckit
from astropy.io import fits
from astropy import units as u

from icemodels import fluxes_in_filters

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')

# filter_ids = ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W', 
#              'JWST/NIRCam.F444W', 'JWST/NIRCam.F405N']
filter_ids = [fid for fid in jfilts['filterID'] if fid.startswith('JWST/NIRCam')]
filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in filter_ids}
transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}


from tqdm.auto import tqdm
import glob
import numpy as np
from astropy.table import Table

if os.path.exists(f'{isodir}/iso_spectra_as_fluxes.fits'):
    tbl = Table.read(f'{isodir}/iso_spectra_as_fluxes.fits')
else:
    iso_data = []
    for fn in tqdm(glob.glob(f'{isodir}/*sws.fit')):
        fh = fits.open(fn)
        iso_flxd = fluxes_in_filters(fh[0].data[:,0]*u.um, fh[0].data[:,1]*u.Jy, filterids=filter_ids, transdata=transdata)
        iso_mags = {key: -2.5*np.log10(iso_flxd[key].to(u.Jy).value / filter_data[key]) for key in iso_flxd}
        iso_mags['Object'] = fh[0].header['OBJECT']
        iso_data.append(iso_mags)

    tbl = Table(iso_data)
    tbl.write(f'{isodir}/iso_spectra_as_fluxes.fits')


if __name__ == '__main__':
    import pylab as pl


    pl.figure()
    pl.scatter(tbl['JWST/NIRCam.F212N'] - tbl['JWST/NIRCam.F410M'],
            tbl['JWST/NIRCam.F410M'] - tbl['JWST/NIRCam.F466N'],
            s=1
            )
    pl.xlim(-2, 5)

    pl.xlabel(r'F212N - F410M')
    pl.ylabel(r'F410M - F466N')

    pl.figure()
    pl.scatter(tbl['JWST/NIRCam.F356W'] - tbl['JWST/NIRCam.F410M'],
            tbl['JWST/NIRCam.F410M'] - tbl['JWST/NIRCam.F466N'],
            s=1
            )
    pl.xlim(-0.2, 2)
    pl.ylim(-0.5, 1.5);
    pl.xlabel(r'F356W - F410M')
    pl.ylabel(r'F410M - F466N')


    pl.figure()
    pl.scatter(tbl['JWST/NIRCam.F356W'] - tbl['JWST/NIRCam.F410M'],
            tbl['JWST/NIRCam.F410M'] - tbl['JWST/NIRCam.F444W'],
            s=1
            )
    pl.xlim(-0.2, 2)
    pl.ylim(-0.5, 0.7);
    pl.xlabel(r'F356W - F410M')
    pl.ylabel(r'F410M - F444W')
