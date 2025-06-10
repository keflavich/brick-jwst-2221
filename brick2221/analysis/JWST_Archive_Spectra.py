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
from astropy.convolution import Gaussian1DKernel, convolve
from icemodels import fluxes_in_filters

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')

# filter_ids = ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W',
#              'JWST/NIRCam.F444W', 'JWST/NIRCam.F405N']
filter_ids = [fid for fid in jfilts['filterID'] if fid.startswith('JWST/NIRCam')]
filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in filter_ids}
transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}

max_flux = 1*u.Jy


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

            namestart = "_".join(os.path.basename(fn).split("_")[:-3])
            other_gratings = glob.glob(f'{nirspec_dir}/{namestart}*/*x1d.fits')
            if len(other_gratings) > 0:
                grating_ids = [os.path.basename(fn).split("_")[-2] for fn in other_gratings]
                if len(other_gratings) > 2 and len(other_gratings) != len(set(grating_ids)):
                    raise ValueError(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                #print(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                fn2 = other_gratings[0]
                spectable2 = Table.read(fn2, hdu=1)
                spectable = table.vstack([spectable, spectable2])
                spectable.sort('WAVELENGTH')

            # remove super high values
            spectable['FLUX'] = np.clip(spectable['FLUX'], 0, max_flux.value)

            def fixed_flux(fluxes, threshold=0.2):
                n_nans = np.sum(np.isnan(fluxes))
                if n_nans / len(fluxes) < threshold:
                    fluxes = fluxes.copy()
                    fluxes_conv = convolve(fluxes, Gaussian1DKernel(stddev=10))
                    fluxes[np.isnan(fluxes)] = fluxes_conv[np.isnan(fluxes)]
                return fluxes

            nirspec_flxd = fluxes_in_filters(spectable['WAVELENGTH'].quantity,
                                             fixed_flux(spectable['FLUX'].quantity),
                                             filterids=filter_ids, transdata=transdata)
            nirspec_mags = {key: -2.5*np.log10(nirspec_flxd[key].to(u.Jy).value / filter_data[key])
                                 if nirspec_flxd[key] > 0 else np.nan
                            for key in nirspec_flxd}

            nirspec_flxd_nonan = fluxes_in_filters(spectable['WAVELENGTH'].quantity,
                                             np.nan_to_num(spectable['FLUX'].quantity),
                                             filterids=filter_ids, transdata=transdata)
            nirspec_mags_nonan = {key: -2.5*np.log10(nirspec_flxd_nonan[key].to(u.Jy).value / filter_data[key])
                                 if nirspec_flxd_nonan[key] > 0 else np.nan
                            for key in nirspec_flxd_nonan}
            for key in nirspec_mags_nonan:
                nirspec_mags[key+"_nonan"] = nirspec_mags_nonan[key]

            # filter out measurements that are implausibly faint
            nirspec_mags = {key: np.nan if (nirspec_mags[key] > 26) or (nirspec_mags[key] < 2) else nirspec_mags[key]
                            for key in nirspec_mags}
            nirspec_mags_nonan = {key: np.nan if (nirspec_mags_nonan[key] > 26) or (nirspec_mags_nonan[key] < 2) else nirspec_mags_nonan[key]
                            for key in nirspec_mags_nonan}

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

            slitid = fn.split("_")[-4]
            if slitid.startswith('s'):
                nirspec_mags['SlitID'] = slitid
            else:
                nirspec_mags['SlitID'] = ''

            if nirspec_mags['Object'] == '':
                nirspec_mags['Object'] = slitid

            nirspec_mags['Observation'] = fh[0].header['OBSERVTN']
            nirspec_mags['Visit'] = fh[0].header['VISIT']
            nirspec_mags['VisitGroup'] = fh[0].header['VISITGRP']

            nirspec_mags['Filename'] = fn

            nirspec_mags['Grating'] = fh[0].header['GRATING']
            if 'SLIT_RA' in fh[0].header:
                nirspec_mags['RA'] = fh[0].header['SLIT_RA']
                nirspec_mags['Dec'] = fh[0].header['SLIT_DEC']
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
    tbl.add_index('Grating')
    tbl.add_index('Target')
    tbl.add_index('SlitID')
    tbl.add_index('Observation')
    tbl.add_index('Visit')
    tbl.add_index('VisitGroup')
    tbl.add_index('Filename')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for filters, setname in ((['JWST/NIRCam.F182M',
                                   'JWST/NIRCam.F212N',
                                   'JWST/NIRCam.F405N',
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
                grating = fh[0].header['GRATING']
                srcname = fh[1].header.get('SRCNAME', targ)
                spectable = Table.read(fn, hdu=1)

                namestart = "_".join(os.path.basename(fn).split("_")[:-3])
                other_gratings = glob.glob(f'{nirspec_dir}/{namestart}*/*x1d.fits')
                if len(other_gratings) > 0:
                    grating_ids = [os.path.basename(fn).split("_")[-2] for fn in other_gratings]
                    if len(other_gratings) > 2 and len(other_gratings) != len(set(grating_ids)):
                        raise ValueError(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                    #print(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                    fn2 = other_gratings[0]
                    spectable2 = Table.read(fn2, hdu=1)
                    spectable = table.vstack([spectable, spectable2])
                    spectable.sort('WAVELENGTH')
                    fh2 = fits.open(fn2)
                    grating2 = fh2[0].header['GRATING']
                else:
                    grating2 = ''

                spectable['FLUX'][spectable['FLUX_ERROR'] > max_flux.value] = np.nan

                sp = pyspeckit.Spectrum(xarr=spectable['WAVELENGTH'].quantity,
                                        data=spectable['FLUX'].quantity,
                                        header=fh[0].header
                                    )
                row = tbl.loc[('Target', targ)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Grating', grating)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Object', srcname)]


                obsid = fh[0].header['OBSERVTN']
                visitid = fh[0].header['VISIT']
                visitgroupid = fh[0].header['VISITGRP']
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Observation', obsid)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Visit', visitid)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('VisitGroup', visitgroupid)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Filename', fn)]

                slitid = fn.split("_")[-4]
                if isinstance(row, Table) and len(row) > 0:
                    if slitid.startswith('s'):
                        row = row.loc[('SlitID', slitid)]
                    else:
                        print(fn)
                        print(row)
                        raise ValueError(f"There are multiple rows matching target={targ}, grating={grating}, object={srcname}.  There was no slitid {slitid} to distinguish them.")

                if 'MSA_Cat' in targ:
                    sp.specname = f'{slitid} {grating} {grating2}'
                else:
                    sp.specname = f'{targ} {grating} {grating2}'
                sp.plotter()
                sp.plotter.axis.set_xlabel("Wavelength [$\\mu m$]")

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
                    if pl.ylim()[1] > max_flux.value:
                        pl.ylim(0, max_flux.value)
                    mid = np.array(pl.ylim()).mean()
                    pl.plot(transdata[key]['Wavelength']/1e4, transdata[key]['Transmission'] * mid, linewidth=0.5,
                            label=f"{key[-5:]}: {mag:0.2f}" if np.isfinite(mag) else f'{key[-5:]}: -'
                        )
                if setname == '2221':
                    pl.plot([], [], label=f'[F182M] - [F212N] = {mags["F182M"] - mags["F212N"]:0.2f}')
                    pl.plot([], [], label=f'[F212N] - [F466N] = {mags["F212N"] - mags["F466N"]:0.2f}')
                    pl.plot([], [], label=f'[F410M] - [F466N] = {mags["F410M"] - mags["F466N"]:0.2f}')
                    pl.plot([], [], label=f'[F405N] - [F410M] = {mags["F405N"] - mags["F410M"]:0.2f}')
                elif setname == '1182':
                    pl.plot([], [], label=f'[F115W] - [F200W] = {mags["F115W"] - mags["F200W"]:0.2f}')
                    pl.plot([], [], label=f'[F200W] - [F444W] = {mags["F200W"] - mags["F444W"]:0.2f}')
                    pl.plot([], [], label=f'[F356W] - [F444W] = {mags["F356W"] - mags["F444W"]:0.2f}')
                pl.legend(loc='best');
                pl.savefig(f'{nirspec_dir}/{targ}_{srcname}_{grating}{grating2}_{setname}_{slitid}_o{obsid}_v{visitid}_vg{visitgroupid}.png', dpi=150)