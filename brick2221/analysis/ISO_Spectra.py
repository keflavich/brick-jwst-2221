import os

import pyspeckit
from astropy.io import fits
from astropy import units as u

from icemodels import fluxes_in_filters

from brick2221.analysis.JWST_Archive_Spectra import adjust_yaxis_for_legend_overlap

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

isodir = '/orange/adamginsburg/ice/iso/library/swsatlas/'


def make_iso_spectra_as_fluxes_table():
    iso_data = []
    for fn in tqdm(glob.glob(f'{isodir}/*sws.fit')):
        fh = fits.open(fn)
        iso_flxd = fluxes_in_filters(fh[0].data[:,0]*u.um, fh[0].data[:,1]*u.Jy, filterids=filter_ids, transdata=transdata)
        iso_mags = {key: -2.5*np.log10(iso_flxd[key].to(u.Jy).value / filter_data[key]) for key in iso_flxd}
        iso_mags['Object'] = fh[0].header['OBJECT']
        iso_data.append(iso_mags)

    tbl = Table(iso_data)
    tbl.write(f'{isodir}/iso_spectra_as_fluxes.fits', overwrite=True)

    return tbl


if __name__ == '__main__':
    import pylab as pl

    if os.path.exists(f'{isodir}/iso_spectra_as_fluxes.fits'):
        tbl = Table.read(f'{isodir}/iso_spectra_as_fluxes.fits')
    else:
        tbl = make_iso_spectra_as_fluxes_table()


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


    for fn in tqdm(glob.glob(f'{isodir}/*sws.fit')):
        fh = fits.open(fn)
        iso_flxd = fluxes_in_filters(fh[0].data[:,0]*u.um, fh[0].data[:,1]*u.Jy, filterids=filter_ids, transdata=transdata)
        iso_mags = {key: -2.5*np.log10(iso_flxd[key].to(u.Jy).value / filter_data[key]) for key in iso_flxd}
        iso_mags['Object'] = fh[0].header['OBJECT']

        mags = {}
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
            sp = pyspeckit.Spectrum(data=fh[0].data[:,1]*u.Jy, xarr=fh[0].data[:,0]*u.um)
            sp.specname = fh[0].header['OBJECT']

            if setname == '2221':
                try:
                    sp.plotter(xmin=3.5, xmax=4.8)
                except Exception as ex:
                    print(fn, ex)
                    sp.plotter(xmin=2.3, xmax=5.1)
            else:
                sp.plotter(xmin=2.3, xmax=5.1)
            ax = sp.plotter.axis
            ax.set_xlabel("Wavelength [$\\mu m$]")
            ax.set_title(f'{sp.specname}')

            for key in filters:
                mag = iso_mags[key]
                mags[key.split(".")[-1]] = mag
                if ax.get_ylim()[0] < 0:
                    ax.set_ylim(0, ax.get_ylim()[1])
                mid = np.array(ax.get_ylim()).mean()
                ax.plot(transdata[key]['Wavelength']/1e4,
                        transdata[key]['Transmission'] / transdata[key]['Transmission'].max() * mid,
                        linewidth=0.5,
                        # don't show anything for non-overlapping, but we need the plot to happen so the color cycles stay consistent w/JWST spectra
                        label=f"{key[-5:]}: {mag:0.2f}" if np.isfinite(mag) else None,
                    )
            if setname == '2221':
                #ax.plot([], [], label=f'[F182M] - [F212N] = {mags["F182M"] - mags["F212N"]:0.2f}', linestyle='none', color='k')
                #ax.plot([], [], label=f'[F212N] - [F466N] = {mags["F212N"] - mags["F466N"]:0.2f}', linestyle='none', color='k')
                ax.plot([], [], label=f'[F405N] - [F466N] = {mags["F405N"] - mags["F466N"]:0.2f}', linestyle='none', color='k')
                ax.plot([], [], label=f'[F405N] - [F410M] = {mags["F405N"] - mags["F410M"]:0.2f}', linestyle='none', color='k')

            elif setname == '1182':
                #ax.plot([], [], label=f'[F115W] - [F200W] = {mags["F115W"] - mags["F200W"]:0.2f}', linestyle='none', color='k')
                #ax.plot([], [], label=f'[F200W] - [F444W] = {mags["F200W"] - mags["F444W"]:0.2f}', linestyle='none', color='k')
                #ax.plot([], [], label=f'[F200W] - [F356W] = {mags["F200W"] - mags["F356W"]:0.2f}', linestyle='none', color='k')
                ax.plot([], [], label=f'[F356W] - [F444W] = {mags["F356W"] - mags["F444W"]:0.2f}', linestyle='none', color='k')

            ax.set_ylim(0, ax.get_ylim()[1])

            ax.legend(loc='upper left', fontsize=10);
            adjust_yaxis_for_legend_overlap(ax)


            save_specname = sp.specname.replace(" ", "_").replace("/", "_")
            pl.savefig(f'{isodir}/pngs/{os.path.splitext(os.path.basename(fn))[0]}_{save_specname}_{setname}.png', dpi=150)
            pl.close('all')