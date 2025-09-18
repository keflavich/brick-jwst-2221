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
import astropy.io.registry.base
from astropy.coordinates import SkyCoord

from astroquery.svo_fps import SvoFps
jfilts = SvoFps.get_filter_list('JWST')
jfilts.add_index('filterID')


# jwst_archive_spectra_checklist.csv is a hand-edited checklist
# key for checklist: y = yes, all filters covered.  n=no, some filters not covered.  g=there is overlap between a filter and a gap w=weird l=low s/n in long-wavelength data, e=emission

import matplotlib.pyplot as plt
import numpy as np

def adjust_yaxis_for_legend_overlap(ax, margin_factor=1.1):
    """
    Adjust y-axis to avoid legend overlap with plotted data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Axes object containing the plot.
    margin_factor : float
        Factor to increase y-axis range by if overlap is detected.
    """
    fig = ax.figure
    fig.canvas.draw()  # Make sure the legend is rendered

    legend = ax.get_legend()
    if legend is None:
        return

    # Get legend bounding box in display coordinates (pixels)
    legend_bbox = legend.get_window_extent()

    # Convert to data coordinates
    inv = ax.transData.inverted()
    legend_bbox_data = inv.transform(legend_bbox)

    # Extract bounding box ranges
    x0, y0 = legend_bbox_data[0]
    x1, y1 = legend_bbox_data[1]

    # Get data from lines in the plot
    for line in ax.lines:
        xdata, ydata = line.get_data()
        mask = (
            (xdata >= x0) & (xdata <= x1) &
            (ydata >= y0) & (ydata <= y1)
        )
        if np.any(mask):
            # Overlap detected, expand y-axis
            ymin, ymax = ax.get_ylim()
            new_ymax = ymax * margin_factor if ymax > 0 else ymax / margin_factor
            ax.set_ylim(ymin, new_ymax)
            fig.canvas.draw()  # Re-render with new limits
            # Recursive call to ensure multiple adjustments if needed
            adjust_yaxis_for_legend_overlap(ax, margin_factor)
            return  # Stop after the first adjustment



# filter_ids = ['JWST/NIRCam.F410M', 'JWST/NIRCam.F466N', 'JWST/NIRCam.F356W',
#              'JWST/NIRCam.F444W', 'JWST/NIRCam.F405N']
filter_ids = [fid for fid in jfilts['filterID'] if fid.startswith('JWST/NIRCam')]
filter_data = {fid: float(jfilts.loc[fid]['ZeroPoint']) for fid in filter_ids}
transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}

max_flux = 1*u.Jy


nirspec_dir = '/orange/adamginsburg/jwst/spectra/mastDownload/JWST/'
checklist = Table.read(f'{nirspec_dir}/jwst_archive_spectra_checklist.csv')

def make_nirspec_spectra_as_fluxes_table():
    nirspec_data = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for fn in tqdm(glob.glob(f'{nirspec_dir}/jw*/*x1d.fits')):
            try:
                fh = fits.open(fn)
            except OSError as ex:
                print(f'{fn} failed to open: {ex} (it should be deleted or re-downloaded)')
                continue
            spectable = Table.read(fn, hdu=1)

            namestart = "_".join(os.path.basename(fn).split("_")[:-3])
            other_gratings = glob.glob(f'{nirspec_dir}/{namestart}*/*x1d.fits')
            if fh[0].header['GRATING'] != 'PRISM' and len(other_gratings) > 0:
                grating_ids = [os.path.basename(fn).split("_")[-2] for fn in other_gratings]
                if len(other_gratings) > 2 and len(other_gratings) != len(set(grating_ids)):
                    raise ValueError(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                #print(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                fn2 = other_gratings[0]
                try:
                    spectable2 = Table.read(fn2, hdu=1)
                except astropy.io.registry.base.IORegistryError as ex:
                    print(f'{fn2} failed to open: {ex} (it should be deleted or re-downloaded)')
                    continue
                spectable = table.vstack([spectable, spectable2])
                spectable.sort('WAVELENGTH')

            # remove super high values
            spectable['FLUX'][spectable['FLUX'] > max_flux.value] = np.nan
            # flag out negatives too
            spectable['FLUX'][spectable['FLUX'] < 0] = np.nan

            # Remove only positive outliers using sigma clipping
            from astropy.stats import sigma_clip
            mean_flux = np.nanmean(np.array(spectable['FLUX']))
            std_flux = np.nanstd(np.array(spectable['FLUX']))
            bad_flux = spectable['FLUX'] > mean_flux + 10*std_flux
            print(f"Removing {bad_flux.sum()} pixels with flux > {mean_flux + 10*std_flux}.  These will be infilled by interpolation.")
            spectable['FLUX'][bad_flux] = np.nan

            nirspec_flxd = fluxes_in_filters(spectable['WAVELENGTH'].quantity,
                                             (spectable['FLUX'].quantity),
                                             filterids=filter_ids, transdata=transdata)
            nirspec_mags = {key: -2.5*np.log10(nirspec_flxd[key].to(u.Jy).value / filter_data[key])
                                 if nirspec_flxd[key] > 0 else np.nan
                            for key in nirspec_flxd}

            # filter out measurements that are implausibly faint
            nirspec_mags = {key: np.nan if (nirspec_mags[key] > 26) or (nirspec_mags[key] < 2) else nirspec_mags[key]
                            for key in nirspec_mags}
            # do this after filtering to avoid quantity comparison failures
            for key in nirspec_flxd:
                try:
                    nirspec_mags[key+"_flx"] = nirspec_flxd[key].value
                except AttributeError:
                    nirspec_mags[key+"_flx"] = nirspec_flxd[key]

            nirspec_mags['wl_min'] = spectable['WAVELENGTH'].quantity.min()
            nirspec_mags['wl_max'] = spectable['WAVELENGTH'].quantity.max()
            nirspec_mags['bad_pixels'] = bad_flux.sum()
            nirspec_mags['neg_pixels'] = (spectable['FLUX'] < 0).sum()

            emission_line = ((spectable['FLUX'][np.argmin(np.abs(spectable['WAVELENGTH'] - 1.875*u.um))] >
                              2*spectable['FLUX'][np.argmin(np.abs(spectable['WAVELENGTH'] - 1.82*u.um))]) and
                             (spectable['FLUX'][np.argmin(np.abs(spectable['WAVELENGTH'] - 4.05*u.um))] >
                              2*spectable['FLUX'][np.argmin(np.abs(spectable['WAVELENGTH'] - 4.00*u.um))]))
            nirspec_mags['emission_line'] = emission_line


            nirspec_mags['Target'] = fh[0].header['TARGNAME']
            if nirspec_mags['Target'] == '':
                nirspec_mags['Target'] = fh[0].header['TARGPROP']
            if nirspec_mags['Target'] is None or nirspec_mags['Target'] == '':
                nirspec_mags['Target'] = os.path.basename(fn).split('x1d')[0]

            try:
                nirspec_mags['Program'] = fh[0].header['PROGRAM']
            except KeyError:
                nirspec_mags['Program'] = fh[1].header['PROGRAM']

            nirspec_mags['Grating'] = fh[0].header['GRATING']
            if nirspec_mags['Grating'] != 'PRISM' and len(other_gratings) > 0:
                fh2 = fits.open(fn2)
                nirspec_mags['Grating2'] = fh2[0].header['GRATING']
            else:
                nirspec_mags['Grating2'] = ''

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

            if 'SLIT_RA' in fh[0].header:
                nirspec_mags['RA'] = fh[0].header['SLIT_RA']
                nirspec_mags['Dec'] = fh[0].header['SLIT_DEC']


            # flag out bad spectra
            # if slitid, reason_bad in {'none': 'foo'}:
            #     nirspec_mags['bad'] = True
            #     nirspec_mags['bad_reason'] = reason_bad
            # else:
            #     nirspec_mags['bad'] = False

            nirspec_data.append(nirspec_mags)

    tbl = Table(nirspec_data)

    for key in tbl.colnames:
        if key.endswith('_flx'):
            tbl[key] = u.Quantity(tbl[key], unit=u.Jy)

    # 'Grating', 'Grating2',
    common_keys = ['Target', 'Program', 'Object', 'SlitID',
                   'Observation', 'Visit', 'VisitGroup', 'Filename']
    checklist['SlitID'][checklist['SlitID'].mask] = ''
    tbl['Program'] = tbl['Program'].astype(int)
    tbl['Observation'] = tbl['Observation'].astype(int)
    tbl['Visit'] = tbl['Visit'].astype(int)
    tbl['VisitGroup'] = tbl['VisitGroup'].astype('int')
    tbl = table.join(tbl, checklist, keys=common_keys)
    if 'Grating_1' in tbl.colnames:
        tbl['Grating'] = tbl['Grating_1']
        tbl['Grating2'] = tbl['Grating2_1']
        del tbl['Grating_1']
        del tbl['Grating2_1']
        del tbl['Grating_2']
        del tbl['Grating2_2']

    tbl.write(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.ecsv', overwrite=True)
    tbl.write(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits', overwrite=True)

    return tbl


if __name__ == '__main__':
    import pylab as pl

    # "if false" to force re-run; mostly we want this b/c we're downloading new data
    if False and os.path.exists(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits'):
        tbl = Table.read(f'{nirspec_dir}/jwst_archive_spectra_as_fluxes.fits')
    else:
        tbl = make_nirspec_spectra_as_fluxes_table()

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
                try:
                    fh = fits.open(fn)
                except OSError as ex:
                    print(f'{fn} failed to open: {ex} (it should be deleted or re-downloaded)')
                    continue

                targ = fh[0].header["TARGNAME"]
                if targ == '':
                    targ = fh[0].header['TARGPROP']
                if targ is None or targ == '':
                    targ = os.path.basename(fn).split('x1d')[0]
                grating = fh[0].header['GRATING']
                srcname = fh[1].header.get('SRCNAME', targ)
                srcalias = fh[1].header.get('SRCALIAS', srcname)
                spectable = Table.read(fn, hdu=1)

                namestart = "_".join(os.path.basename(fn).split("_")[:-3])
                other_gratings = glob.glob(f'{nirspec_dir}/{namestart}*/*x1d.fits')
                if grating != 'PRISM' and len(other_gratings) > 0:
                    grating_ids = [os.path.basename(fn).split("_")[-2] for fn in other_gratings]
                    if len(other_gratings) > 2 and len(other_gratings) != len(set(grating_ids)):
                        raise ValueError(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                    fn2 = other_gratings[0]
                    fh2 = fits.open(fn2)
                    grating2 = fh2[0].header['GRATING']
                    if grating2 != grating:
                        #print(f'{fn} -> {namestart} has multiple gratings: {other_gratings}')
                        try:
                            spectable2 = Table.read(fn2, hdu=1)
                        except astropy.io.registry.base.IORegistryError as ex:
                            print(f'{fn2} failed to open: {ex} (it should be deleted or re-downloaded)')
                            continue
                        spectable = table.vstack([spectable, spectable2])
                        spectable.sort('WAVELENGTH')
                    else:
                        print(f"File {fn2} has the same grating {grating2} as {fn}, skipping")
                        grating2 = ''
                else:
                    grating2 = ''


                spectable['FLUX'][spectable['FLUX_ERROR'] > max_flux.value] = np.nan

                sp = pyspeckit.Spectrum(xarr=spectable['WAVELENGTH'].quantity,
                                        data=spectable['FLUX'].quantity,
                                        header=fh[0].header
                                    )
                if targ not in tbl['Target']:
                    print(f'{fn} has target {targ} not in {tbl["Target"]}')
                    continue
                row = tbl.loc[('Target', targ)]
                if grating not in row['Grating']:
                    print(f'{fn} has grating {grating} not in {row["Grating"]} for target {targ}')
                    continue
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Grating', grating)]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Object', srcname)]


                program = fh[0].header['PROGRAM']
                obsid = fh[0].header['OBSERVTN']
                visitid = fh[0].header['VISIT']
                visitgroupid = fh[0].header['VISITGRP']
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Observation', int(obsid))]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('Visit', int(visitid))]
                if isinstance(row, Table) and len(row) > 0:
                    row = row.loc[('VisitGroup', int(visitgroupid))]
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
                    sp.specname = f'{program} {srcalias} {slitid} {grating} {grating2}'
                elif targ == 'Serpens_Targets':
                    sp.specname = f'Serpens {program} {srcalias} {slitid} {grating} {grating2}'
                elif targ == 'Catalogue_4':
                    sp.specname = f'Orion {program} {srcalias} {grating} {grating2}'
                else:
                    sp.specname = f'{targ} {program} {srcalias} {grating} {grating2}'
                if not np.any(np.isfinite(sp.data)):
                    print(f'{fn} has no finite data')
                    continue
                sp.plotter()
                ax = sp.plotter.axis
                ax.set_xlabel("Wavelength [$\\mu m$]")

                try:
                    slit_coord = SkyCoord(fh[0].header['SLIT_RA'], fh[0].header['SLIT_DEC'], unit=(u.deg, u.deg))
                except KeyError:
                    slit_coord = SkyCoord(fh[1].header['SLIT_RA'], fh[1].header['SLIT_DEC'], unit=(u.deg, u.deg))
                ax.set_title(f'{sp.specname}\n{slit_coord.to_string(sep=":", style="hmsdms", decimal=False)}')

                pl.savefig(f'{nirspec_dir}/pngs/{targ}_{srcname}_{grating}{grating2}_{slitid}_o{obsid}_v{visitid}_vg{visitgroupid}_p{program}.png', dpi=150)

                mags = {}
                for key in filters:
                    mag = row[key]
                    mags[key.split(".")[-1]] = mag
                    if isinstance(row, Table) and len(row) > 1: # and hasattr(mag, '__len__') and len(mag) > 1:
                        # print(row['Object', 'Target', 'grating'])
                        # print(f'{targ} {srcname} {grating} {key} has multiple values: {mag}')
                        # raise ValueError('multiple values')
                        mag = row[key].max()
                    if ax.get_ylim()[0] < 0:
                        ax.set_ylim(0, ax.get_ylim()[1])
                    if ax.get_ylim()[1] > max_flux.value:
                        ax.set_ylim(0, max_flux.value)
                    mid = np.array(ax.get_ylim()).mean()
                    ax.plot(transdata[key]['Wavelength']/1e4, transdata[key]['Transmission'] * mid, linewidth=0.5,
                            label=f"{key[-5:]}: {mag:0.2f}" if np.isfinite(mag) else f'{key[-5:]}: -'
                        )
                if setname == '2221':
                    ax.plot([], [], label=f'[F182M] - [F212N] = {mags["F182M"] - mags["F212N"]:0.2f}', linestyle='none', color='k')
                    ax.plot([], [], label=f'[F212N] - [F466N] = {mags["F212N"] - mags["F466N"]:0.2f}', linestyle='none', color='k')
                    ax.plot([], [], label=f'[F405N] - [F466N] = {mags["F405N"] - mags["F466N"]:0.2f}', linestyle='none', color='k')
                    ax.plot([], [], label=f'[F405N] - [F410M] = {mags["F405N"] - mags["F410M"]:0.2f}', linestyle='none', color='k')
                elif setname == '1182':
                    ax.plot([], [], label=f'[F115W] - [F200W] = {mags["F115W"] - mags["F200W"]:0.2f}', linestyle='none', color='k')
                    ax.plot([], [], label=f'[F200W] - [F444W] = {mags["F200W"] - mags["F444W"]:0.2f}', linestyle='none', color='k')
                    ax.plot([], [], label=f'[F200W] - [F356W] = {mags["F200W"] - mags["F356W"]:0.2f}', linestyle='none', color='k')
                    ax.plot([], [], label=f'[F356W] - [F444W] = {mags["F356W"] - mags["F444W"]:0.2f}', linestyle='none', color='k')

                ax.legend(loc='upper left', fontsize=10);
                adjust_yaxis_for_legend_overlap(ax)

                # if row['bad']:
                #     title = pl.title()
                #     pl.title(f'BAD: {title}')
                #     # Add red X across the figure
                #     xlim = pl.xlim()
                #     ylim = pl.ylim()
                #     pl.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'r-', linewidth=2, alpha=0.7)
                #     pl.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], 'r-', linewidth=2, alpha=0.7)

                pl.savefig(f'{nirspec_dir}/pngs/{targ}_{srcname}_{grating}{grating2}_{setname}_{slitid}_o{obsid}_v{visitid}_vg{visitgroupid}_p{program}.png', dpi=150)
                pl.close('all')