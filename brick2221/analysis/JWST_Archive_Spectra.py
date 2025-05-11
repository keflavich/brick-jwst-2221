from tqdm.auto import tqdm
import glob
import numpy as np
import warnings
import os
import pyspeckit
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy import table
from icemodels import fluxes_in_filters

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')

# filter_ids = ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W',
#              'JWST/NIRCam.F444W', 'JWST/NIRCam.F405N']
filter_ids = [fid for fid in jfilts['filterID'] if fid.startswith('JWST/NIRCam')]
filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in filter_ids}
transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}


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

            # filter out measurements that are implausibly faint
            nirspec_mags = {key: np.nan if (nirspec_mags[key] > 25) or (nirspec_mags[key] < 2) else nirspec_mags[key]
                            for key in nirspec_mags}

            nirspec_mags['Target'] = fh[0].header['TARGNAME']
            if nirspec_mags['Target'] == '':
                nirspec_mags['Target'] = fh[0].header['TARGPROP']
            if nirspec_mags['Target'] is None or nirspec_mags['Target'] == '':
                nirspec_mags['Target'] = os.path.basename(fn).split('x1d')[0]

            try:
                nirspec_mags['Program'] = fh[0].header['PROGRAM']
            except KeyError:
                nirspec_mags['Program'] = fh[1].header['PROGRAM']

            try:
                nirspec_mags['Object'] = fh[1].header['SRCNAME']
            except KeyError:
                try:
                    nirspec_mags['Object'] = fh[1].header['SRCALIAS']
                except KeyError:
                    if nirspec_mags['Target'] == 'Serpens_Targets':
                        raise ValueError(f'Serpens_Targets has no SRCNAME or SRCALIAS: {fn}')
                    nirspec_mags['Object'] = nirspec_mags['Target']
            nirspec_mags['grating'] = fh[0].header['GRATING']
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
    tbl.add_index('grating')
    tbl.add_index('Target')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for filters, setname in ((['JWST/NIRCam.F182M',
                                   'JWST/NIRCam.F212N',
                                   'JWST/NIRCam.F410M',
                                   'JWST/NIRCam.F466N',],
                                  '2221'),
                                (['JWST/NIRCam.F115W',
                                   'JWST/NIRCam.F200W',
                                   'JWST/NIRCam.F356W',
                                   'JWST/NIRCam.F444W',
                                   ], '1182'),
                                ):
            for fn in tqdm(glob.glob(f'{nirspec_dir}/jw*/*x1d.fits')):
                fh = fits.open(fn)
                targ = fh[0].header["TARGNAME"]
                if targ == '':
                    targ = fh[0].header['TARGPROP']
                if targ is None or targ == '':
                    targ = os.path.basename(fn).split('x1d')[0]
                    continue
                grating = fh[0].header['GRATING']
                srcname = fh[1].header.get('SRCNAME', targ)
                spectable = Table.read(fn, hdu=1)
                sp = pyspeckit.Spectrum(xarr=spectable['WAVELENGTH'].quantity,
                                        data=spectable['FLUX'].quantity,
                                        header=fh[0].header
                                    )
                sp.specname = f'{targ} {grating}'
                sp.plotter()
                row = tbl.loc[('Target', targ)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('grating', grating)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Object', srcname)]

                mags = {}
                for key in filters:
                    mag = row[key]
                    mags[key.split(".")[-1]] = mag
                    if isinstance(row, Table) and len(row) > 1: # and hasattr(mag, '__len__') and len(mag) > 1:
                        # print(row['Object', 'Target', 'grating'])
                        # print(f'{targ} {srcname} {grating} {key} has multiple values: {mag}')
                        # raise ValueError('multiple values')
                        mag = row[key].max()
                    if pl.ylim()[0] < 0:
                        pl.ylim(0, pl.ylim()[1])
                    mid = np.array(pl.ylim()).mean()
                    pl.plot(transdata[key]['Wavelength']/1e4, transdata[key]['Transmission'] * mid, linewidth=0.5,
                            label=f"{key[-5:]}: {mag:0.2f}" if np.isfinite(mag) else f'{key[-5:]}: -'
                        )
                if setname == '2221':
                    pl.plot([], [], label=f'[F182M] - [F212N] = {mags["F182M"] - mags["F212N"]:0.2f}')
                    pl.plot([], [], label=f'[F212N] - [F466N] = {mags["F212N"] - mags["F466N"]:0.2f}')
                    pl.plot([], [], label=f'[F410M] - [F466N] = {mags["F410M"] - mags["F466N"]:0.2f}')
                elif setname == '1182':
                    pl.plot([], [], label=f'[F115W] - [F200W] = {mags["F115W"] - mags["F200W"]:0.2f}')
                    pl.plot([], [], label=f'[F200W] - [F444W] = {mags["F200W"] - mags["F444W"]:0.2f}')
                    pl.plot([], [], label=f'[F356W] - [F444W] = {mags["F356W"] - mags["F444W"]:0.2f}')
                pl.legend(loc='best');
                pl.savefig(f'{nirspec_dir}/{targ}_{srcname}_{grating}_{setname}.png', dpi=150)