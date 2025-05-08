
import warnings
import os
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


nirspec_dir = '/orange/adamginsburg/jwst/spectra/mastDownload/JWST/'

if False and os.path.exists(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits'):
    tbl = Table.read(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits')
else:
    nirspec_data = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for fn in tqdm(glob.glob(f'{nirspec_dir}/jw*/*x1d.fits')):
            fh = fits.open(fn)
            spectable = Table.read(fn, hdu=1)
            nirspec_flxd = fluxes_in_filters(spectable['WAVELENGTH'].quantity,
                                             np.nan_to_num(spectable['FLUX'].quantity),
                                             filterids=filter_ids, transdata=transdata)
            nirspec_mags = {key: -2.5*np.log10(nirspec_flxd[key].to(u.Jy).value / filter_data[key])
                                 if nirspec_flxd[key] > 0 else np.nan
                            for key in nirspec_flxd}
            nirspec_mags['Object'] = fh[0].header['TARGNAME']
            if nirspec_mags['Object'] == '':
                nirspec_mags['Object'] = fh[0].header['TARGPROP']
            if nirspec_mags['Object'] is None or nirspec_mags['Object'] == '':
                nirspec_mags['Object'] = os.path.basename(fn).split('x1d')[0]
            nirspec_data.append(nirspec_mags)

    tbl = Table(nirspec_data)
    tbl.write(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits', overwrite=True)



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


    tbl.add_index('Object')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for fn in tqdm(glob.glob(f'{nirspec_dir}/jw*/*x1d.fits')):
            fh = fits.open(fn)
            obj = fh[0].header["TARGNAME"]
            if obj == '':
                obj = fh[0].header['TARGPROP']
            if obj is None or obj == '':
                obj = os.path.basename(fn).split('x1d')[0]
                continue
            spectable = Table.read(fn, hdu=1)
            sp = pyspeckit.Spectrum(xarr=spectable['WAVELENGTH'].quantity,
                                    data=spectable['FLUX'].quantity,
                                    header=fh[0].header
                                )
            sp.specname = obj
            sp.plotter()
            for key in ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F405N', 'JWST/NIRCam.F444W', 'JWST/NIRCam.F356W']:
                mag = tbl.loc[obj][key]
                if hasattr(mag, '__len__') and len(mag) > 1:
                    mag = mag[0]
                if pl.ylim()[0] < 0:
                    pl.ylim(0, pl.ylim()[1])
                mid = np.array(pl.ylim()).mean()
                pl.plot(transdata[key]['Wavelength']/1e4, transdata[key]['Transmission'] * mid, linewidth=0.5,
                        label=f"{key[-5:]}: {mag:0.2f}" if np.isfinite(mag) else f'{key[-5:]}: -'
                    )
            pl.legend(loc='best');
            pl.savefig(f'{nirspec_dir}/{obj}.png', dpi=150)