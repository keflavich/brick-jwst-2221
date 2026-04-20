print("Starting crowdsource_catalogs_long", flush=True)
import sys
import glob
import time
import json
import re
import inspect
import numpy
import regions
import numpy as np
from pathlib import Path
from functools import cache
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, interpolate_replace_nans
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import wcs
from astropy import table
from astropy import stats
from astropy import units as u
from astropy.nddata import NDData
from astropy.io import fits
from scipy import ndimage
from scipy.spatial import cKDTree
import requests
import requests.exceptions
import urllib3
import urllib3.exceptions
from jwst.datamodels import dqflags
from jwst.datamodels import ImageModel
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.resample import ResampleStep
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import extract_stars, EPSFStars, EPSFBuilder
# EPSFModel was deprecated in photutils 2.0 in favour of ImagePSF
try:
    from photutils.psf import ImagePSF as EPSFModel
except ImportError:
    from photutils.psf import EPSFModel
# PSFPhotometry, IterativePSFPhotometry, SourceGrouper present since photutils 1.9
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper
# LocalBackground present since photutils 1.9
from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D, LocalBackground

import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)

import crowdsource
from crowdsource import crowdsource_base
from crowdsource.crowdsource_base import fit_im, psfmod

from brick2221.reduction.saturated_star_finding import remove_saturated_stars

from astroquery.svo_fps import SvoFps

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'

import os
print("Importing webbpsf", flush=True)
import stpsf as webbpsf
import stpsf
print(f"Webbpsf version: {webbpsf.__version__}")
from stpsf.utils import to_griddedpsfmodel
import datetime
print("Done with imports", flush=True)

FWHM_TABLE = Path(__file__).resolve().parents[1] / 'reduction' / 'fwhm_table.ecsv'


def normalize_vgroup_id(vgroup_id):
    if vgroup_id is None or vgroup_id == '':
        return '', None

    vgroup_token = str(vgroup_id)
    if vgroup_token.startswith('_vgroup'):
        vgroup_token = vgroup_token.removeprefix('_vgroup')

    if vgroup_token.isdigit():
        return f'_vgroup{vgroup_token}', int(vgroup_token)

    digit_match = re.search(r'\d+', vgroup_token)
    if digit_match is not None:
        return f'_vgroup{vgroup_token}', int(digit_match.group(0))

    return f'_vgroup{vgroup_token}', None


def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    return printfunc(f"{now}:", *args, **kwargs)


class WrappedPSFModel(crowdsource.psf.SimplePSF):
    """
    wrapper for photutils GriddedPSFModel
    """
    def __init__(self, psfgridmodel, stampsz=19):
        self.psfgridmodel = psfgridmodel
        self.default_stampsz = stampsz

    def __call__(self, col, row, stampsz=None, deriv=False):

        if stampsz is None:
            stampsz = self.default_stampsz

        parshape = numpy.broadcast(col, row).shape
        tparshape = parshape if len(parshape) > 0 else (1,)

        # numpy uses row, column notation
        rows, cols = np.indices((stampsz, stampsz)) - (np.array([stampsz, stampsz])-1)[:, None, None] / 2.

        # explicitly broadcast
        col = np.atleast_1d(col)
        row = np.atleast_1d(row)
        #rows = rows[:, :, None] + row[None, None, :]
        #cols = cols[:, :, None] + col[None, None, :]

        # photutils seems to use column, row notation
        # only works with photutils <= 1.6.0 - but is wrong there
        #stamps = self.psfgridmodel.evaluate(cols, rows, 1, col, row)
        # it returns something in (nstamps, row, col) shape
        # pretty sure that ought to be (col, row, nstamps) for crowdsource

        # andrew saydjari's version here:
        # it returns something in (nstamps, row, col) shape
        stamps = []
        for i in range(len(col)):
            # the +0.5 is required to actually center the PSF (empirically)
            #stamps.append(self.psfgridmodel.evaluate(cols+col[i]+0.5, rows+row[i]+0.5, 1, col[i], row[i]))
            # the above may have been true when we were using (incorrectly) offset PSFs
            stamps.append(self.psfgridmodel.evaluate(cols+col[i], rows+row[i], 1, col[i], row[i]))

        stamps = np.array(stamps)

        # for oversampled stamps, they may not be normalized
        stamps /= stamps.sum(axis=(1,2))[:,None,None]
        # this is evidently an incorrect transpose
        #stamps = np.transpose(stamps, axes=(0,2,1))

        if deriv:
            dpsfdrow, dpsfdcol = np.gradient(stamps, axis=(1, 2))

        ret = stamps
        if parshape != tparshape:
            ret = ret.reshape(stampsz, stampsz)
            if deriv:
                dpsfdrow = dpsfdrow.reshape(stampsz, stampsz)
                dpsfdcol = dpsfdcol.reshape(stampsz, stampsz)
        if deriv:
            ret = (ret, dpsfdcol, dpsfdrow)

        return ret

    def render_model(self, col, row, stampsz=None):
        """
        this function likely does nothing?
        """
        if stampsz is not None:
            self.stampsz = stampsz

        rows, cols = np.indices(self.stampsz, dtype=float) - (np.array(self.stampsz)-1)[:, None, None] / 2.

        return self.psfgridmodel.evaluate(cols, rows, 1, col, row).T.squeeze()


def save_epsf(epsf, filename, overwrite=True):
    header = {}
    header['OVERSAMP'] = list(epsf.oversampling)
    hdu = fits.PrimaryHDU(data=epsf.data, header=header)
    hdu.writeto(filename, overwrite=overwrite)


def read_epsf(filename):
    fh = fits.open(filename)
    hdu = fh[0]
    return EPSFModel(data=hdu.data, oversampling=hdu.header['OVERSAMP'])


def catalog_zoom_diagnostic(data, modsky, zoomcut, stars):

    # make sure stars is a table
    try:
        'qf' in stars.colnames
    except AttributeError:
        stars = Table(stars)

    pl.figure(figsize=(12,12))
    im = pl.subplot(2,2,1).imshow(data[zoomcut],
                                  norm=simple_norm(data[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   vmin=0), cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title("Data")
    pl.colorbar(mappable=im)
    im = pl.subplot(2,2,2).imshow(modsky[zoomcut],
                                  norm=simple_norm(modsky[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   vmin=0), cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
    pl.colorbar(mappable=im)

    resid = (data[zoomcut] - modsky[zoomcut])
    rms = stats.mad_std(resid, ignore_nan=True)
    if np.isnan(rms):
        raise ValueError("RMS is nan, this shouldn't happen")

    norm = (simple_norm(resid, stretch='asinh', max_percent=99.95, min_percent=0.5)
            if np.nanmin(resid) > 0 else
            simple_norm(resid, stretch='log', vmax=np.nanpercentile(resid, 99.95), vmin=-2*rms))

    im = pl.subplot(2,2,3).imshow(resid,
                                  norm=norm,
                                  cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title(f"data-modsky (rms={rms:10.3g})")
    pl.colorbar(mappable=im)
    im = pl.subplot(2,2,4).imshow(data[zoomcut],
                                  norm=simple_norm(data[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   vmin=0), cmap='gray')

    if 'qf' in stars.colnames:
        # used in analysis
        qgood = ((stars['qf'] > 0.9) &
                 (stars['spread_model'] < 0.25) &
                 (stars['fracflux'] > 0.75)
                )
        neg = stars['flux'] < 0
    elif 'qfit' in stars.colnames:
        # guesses, no tests don
        qgood = ((stars['qfit'] < 0.4) &
                 (stars['cfit'] < 0.1) &
                 (stars['flags'] == 0))
        neg = stars['flux_fit'] < 0
    else:
        qgood = np.ones(len(stars), dtype='bool')
        neg = np.zeros(len(stars), dtype='bool')

    axlims = pl.axis()
    if zoomcut[0].start:
        # pl.axis([0,zoomcut[0].stop-zoomcut[0].start, 0, zoomcut[1].stop-zoomcut[1].start])
        ok = ((stars['x'] >= zoomcut[1].start) &
              (stars['x'] <= zoomcut[1].stop) &
              (stars['y'] >= zoomcut[0].start) &
              (stars['y'] <= zoomcut[0].stop))
        pl.subplot(2,2,4).scatter(stars['x'][ok & ~qgood]-zoomcut[1].start,
                                  stars['y'][ok & ~qgood]-zoomcut[0].start,
                                  marker='+', color='y', s=8, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][ok & qgood]-zoomcut[1].start,
                                  stars['y'][ok & qgood]-zoomcut[0].start,
                                  marker='x', color='r', s=8, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][neg]-zoomcut[1].start,
                                  stars['y'][neg]-zoomcut[0].start,
                                  marker='1', color='b', s=8, linewidth=0.5)
    else:
        pl.subplot(2,2,4).scatter(stars['x'][~qgood],
                                  stars['y'][~qgood], marker='+', color='lime', s=5, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][qgood],
                                  stars['y'][qgood], marker='x', color='r', s=5, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][neg],
                                  stars['y'][neg], marker='1', color='b', s=5, linewidth=0.5)
    pl.axis(axlims)
    pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
    pl.colorbar(mappable=im)
    pl.tight_layout()


def _get_source_xy(tbl):
    """Return source x/y columns using the first available coordinate convention."""
    if 'x_fit' in tbl.colnames and 'y_fit' in tbl.colnames:
        return np.asarray(tbl['x_fit']), np.asarray(tbl['y_fit'])
    if 'xcentroid' in tbl.colnames and 'ycentroid' in tbl.colnames:
        return np.asarray(tbl['xcentroid']), np.asarray(tbl['ycentroid'])
    if 'x_init' in tbl.colnames and 'y_init' in tbl.colnames:
        return np.asarray(tbl['x_init']), np.asarray(tbl['y_init'])
    if 'x' in tbl.colnames and 'y' in tbl.colnames:
        return np.asarray(tbl['x']), np.asarray(tbl['y'])
    raise KeyError(f"No recognized x/y coordinate columns in {tbl.colnames}")


def _column_to_float_array(tbl, colname):
    col = tbl[colname]
    if hasattr(col, 'filled'):
        return np.asarray(col.filled(np.nan), dtype=float)
    return np.asarray(col, dtype=float)


def _best_available_xy(tbl):
    candidates = [
        ('xcentroid', 'ycentroid'),
        ('x_fit', 'y_fit'),
        ('x_init', 'y_init'),
        ('x', 'y'),
    ]
    best_pair = None
    best_score = -1
    best_x = None
    best_y = None
    for xname, yname in candidates:
        if xname in tbl.colnames and yname in tbl.colnames:
            xvals = _column_to_float_array(tbl, xname)
            yvals = _column_to_float_array(tbl, yname)
            score = np.isfinite(xvals).sum() + np.isfinite(yvals).sum()
            if score > best_score:
                best_score = score
                best_pair = (xname, yname)
                best_x = xvals
                best_y = yvals
    if best_pair is None:
        raise KeyError(f"No recognized x/y coordinate columns in {tbl.colnames}")
    return best_x, best_y


def _has_any_xy_columns(tbl):
    return any(
        xname in tbl.colnames and yname in tbl.colnames
        for xname, yname in (('xcentroid', 'ycentroid'), ('x_fit', 'y_fit'),
                             ('x_init', 'y_init'), ('x', 'y'))
    )


def _skycoord_entries(tbl, colname):
    if colname not in tbl.colnames:
        return [None] * len(tbl)

    col = tbl[colname]
    mask = np.zeros(len(tbl), dtype=bool)
    if hasattr(col, 'mask'):
        mask = np.asarray(col.mask, dtype=bool)

    entries = [None] * len(tbl)
    for ii in range(len(tbl)):
        if mask[ii]:
            continue
        val = col[ii]
        if isinstance(val, SkyCoord):
            entries[ii] = val
    return entries


def _skycoord_radec_arrays(tbl, colname):
    ra = np.full(len(tbl), np.nan, dtype=float)
    dec = np.full(len(tbl), np.nan, dtype=float)
    entries = _skycoord_entries(tbl, colname)
    for ii, coord in enumerate(entries):
        if coord is None:
            continue
        ra[ii] = float(coord.ra.deg)
        dec[ii] = float(coord.dec.deg)
    return ra, dec


def _resolve_seed_skycoords(seed_table, ww=None, preferred_skycoord_col=None):
    seed_table = _as_table(seed_table)
    nsrc = len(seed_table)
    if nsrc == 0:
        return seed_table

    sky_entries = [None] * nsrc
    sky_columns = []
    if preferred_skycoord_col is not None:
        sky_columns.append(preferred_skycoord_col)
    sky_columns.extend(['skycoord', 'skycoord_fit', 'skycoord_centroid', 'skycoord_ref'])
    for colname in sky_columns:
        if colname not in seed_table.colnames:
            continue
        entries = _skycoord_entries(seed_table, colname)
        for ii, entry in enumerate(entries):
            if sky_entries[ii] is None and entry is not None:
                sky_entries[ii] = entry

    if ww is not None and _has_any_xy_columns(seed_table):
        xvals, yvals = _best_available_xy(seed_table)
        missing = np.array([entry is None for entry in sky_entries], dtype=bool)
        finite = np.isfinite(xvals) & np.isfinite(yvals)
        convert = missing & finite
        if np.any(convert):
            converted = ww.pixel_to_world(xvals[convert], yvals[convert])
            for idx, coord in zip(np.where(convert)[0], converted):
                sky_entries[idx] = coord

    if all(entry is None for entry in sky_entries):
        raise ValueError('Could not determine sky coordinates for any seed sources')

    if 'skycoord' not in seed_table.colnames:
        seed_table['skycoord'] = np.empty(nsrc, dtype=object)
    for ii, coord in enumerate(sky_entries):
        seed_table['skycoord'][ii] = coord

    return seed_table


def _sample_background_map(background_map, xvals, yvals):
    """Sample a 2D background image at source coordinates using nearest-neighbor lookup."""
    sampled = np.full(len(xvals), np.nan, dtype='float32')
    if background_map is None:
        return sampled

    xi = np.rint(np.asarray(xvals)).astype(int)
    yi = np.rint(np.asarray(yvals)).astype(int)
    inbounds = ((xi >= 0) & (yi >= 0) &
                (yi < background_map.shape[0]) &
                (xi < background_map.shape[1]))
    sampled[inbounds] = background_map[yi[inbounds], xi[inbounds]]
    return sampled


def _iteration_token(iteration_label):
    if iteration_label in (None, ''):
        return ''

    token = str(iteration_label)
    if token.startswith('_'):
        return token
    return f'_{token}'


def _as_table(data):
    if isinstance(data, Table):
        return Table(data, copy=True)
    if isinstance(data, str):
        return Table.read(data)
    return Table(data)


def _combine_seed_and_satstars(seed_catalog, satstar_table):
    seed_table = _as_table(seed_catalog)
    if 'is_saturated' not in seed_table.colnames:
        seed_table['is_saturated'] = np.zeros(len(seed_table), dtype=bool)

    if satstar_table is None:
        return seed_table

    satstar_table = _as_table(satstar_table)
    if len(satstar_table) == 0:
        return seed_table

    if 'is_saturated' not in satstar_table.colnames:
        satstar_table['is_saturated'] = np.ones(len(satstar_table), dtype=bool)

    return vstack([seed_table, satstar_table], metadata_conflicts='silent')


def _augment_seed_catalog_with_detections(seed_catalog, detection_catalog, match_radius_pix=1.0):
    raise RuntimeError('Use _augment_seed_catalog_with_detections_sky for seeded augmentation')


def _augment_seed_catalog_with_detections_sky(seed_catalog, detection_catalog, ww,
                                              match_radius_pix=1.0,
                                              preferred_seed_skycoord_col=None,
                                              return_stats=False):
    seed_table = _resolve_seed_skycoords(_as_table(seed_catalog), ww=ww,
                                         preferred_skycoord_col=preferred_seed_skycoord_col)
    detection_table = _as_table(detection_catalog)
    stats = {
        'seed_input': len(seed_table),
        'detection_input': len(detection_table),
        'detection_finite_xy': 0,
        'detection_added': 0,
        'detection_rejected_match': 0,
    }

    if len(seed_table) == 0:
        stats['detection_added'] = len(detection_table)
        if return_stats:
            return detection_table, stats
        return detection_table
    if len(detection_table) == 0:
        if return_stats:
            return seed_table, stats
        return seed_table

    det_x, det_y = _best_available_xy(detection_table)
    det_finite = np.isfinite(det_x) & np.isfinite(det_y)
    if not np.any(det_finite):
        if return_stats:
            return seed_table, stats
        return seed_table

    stats['detection_finite_xy'] = int(np.sum(det_finite))

    det_sky = ww.pixel_to_world(det_x[det_finite], det_y[det_finite])
    det_ra = np.asarray(det_sky.ra.deg, dtype=float)
    det_dec = np.asarray(det_sky.dec.deg, dtype=float)
    detection_table = detection_table[det_finite]
    if 'skycoord' not in detection_table.colnames:
        detection_table['skycoord'] = np.empty(len(detection_table), dtype=object)
    for ii, coord in enumerate(det_sky):
        detection_table['skycoord'][ii] = coord
    if 'is_saturated' not in detection_table.colnames:
        detection_table['is_saturated'] = np.zeros(len(detection_table), dtype=bool)

    seed_ra, seed_dec = _skycoord_radec_arrays(seed_table, 'skycoord')
    valid_seed_idx = np.isfinite(seed_ra) & np.isfinite(seed_dec)
    if not np.any(valid_seed_idx):
        combined = vstack([seed_table, detection_table], metadata_conflicts='silent')
        if 'is_saturated' not in combined.colnames:
            combined['is_saturated'] = np.zeros(len(combined), dtype=bool)
        stats['detection_added'] = len(detection_table)
        stats['detection_rejected_match'] = 0
        if return_stats:
            return combined, stats
        return combined

    seed_sky = SkyCoord(ra=seed_ra[valid_seed_idx] * u.deg,
                        dec=seed_dec[valid_seed_idx] * u.deg,
                        frame='icrs')
    det_sky_all = SkyCoord(ra=det_ra * u.deg,
                           dec=det_dec * u.deg,
                           frame='icrs')
    _, sep2d, _ = det_sky_all.match_to_catalog_sky(seed_sky)
    pixscale = ww.proj_plane_pixel_area()**0.5
    match_radius = (match_radius_pix * pixscale).to(u.arcsec)
    keep = sep2d > match_radius

    stats['detection_added'] = int(np.sum(keep))
    stats['detection_rejected_match'] = int(len(keep) - np.sum(keep))

    combined = vstack([seed_table, detection_table[keep]], metadata_conflicts='silent')
    if 'is_saturated' not in combined.colnames:
        combined['is_saturated'] = np.zeros(len(combined), dtype=bool)
    if return_stats:
        return combined, stats
    return combined


class SeededFinder:
    def __init__(self, seed_table, ww=None, preferred_skycoord_col=None):
        self.seed_table = _as_table(seed_table)
        self.ww = ww
        self.preferred_skycoord_col = preferred_skycoord_col

    def __call__(self, data, mask=None):
        seeds = _resolve_seed_skycoords(
            Table(self.seed_table, copy=True),
            ww=self.ww,
            preferred_skycoord_col=self.preferred_skycoord_col,
        )
        if self.ww is None:
            xvals, yvals = _best_available_xy(seeds)
        else:
            sky_ra, sky_dec = _skycoord_radec_arrays(seeds, 'skycoord')
            valid_idx = np.isfinite(sky_ra) & np.isfinite(sky_dec)
            xvals = np.full(len(seeds), np.nan, dtype=float)
            yvals = np.full(len(seeds), np.nan, dtype=float)
            if np.any(valid_idx):
                skycoords = SkyCoord(ra=sky_ra[valid_idx] * u.deg,
                                     dec=sky_dec[valid_idx] * u.deg,
                                     frame='icrs')
                xx, yy = self.ww.world_to_pixel(skycoords)
                xvals[valid_idx] = np.asarray(xx, dtype=float)
                yvals[valid_idx] = np.asarray(yy, dtype=float)

        finite = np.isfinite(xvals) & np.isfinite(yvals)
        seeds = seeds[finite]
        xvals = xvals[finite]
        yvals = yvals[finite]

        if 'flux' not in seeds.colnames:
            if 'flux_fit' in seeds.colnames:
                seeds['flux'] = np.asarray(seeds['flux_fit'], dtype=float)
            else:
                seeds['flux'] = np.ones(len(seeds), dtype=float)
        seeds['xcentroid'] = np.asarray(xvals, dtype=float)
        seeds['ycentroid'] = np.asarray(yvals, dtype=float)
        seeds['x_init'] = np.asarray(xvals, dtype=float)
        seeds['y_init'] = np.asarray(yvals, dtype=float)
        seeds['flux_init'] = np.asarray(seeds['flux'], dtype=float)
        return seeds


def build_hybrid_saturated_artifact_mask(shape, satstar_table, core_radius_pix=12, halo_radius_pix=28,
                                         flux_scale_pix=1.0):
    mask = np.zeros(shape, dtype=bool)
    if satstar_table is None:
        return mask

    satstar_table = _as_table(satstar_table)
    if len(satstar_table) == 0:
        return mask

    xvals, yvals = _get_source_xy(satstar_table)
    if 'flux_fit' in satstar_table.colnames:
        fluxvals = np.asarray(satstar_table['flux_fit'], dtype=float)
    else:
        fluxvals = np.ones(len(satstar_table), dtype=float)

    yy, xx = np.indices(shape)
    for xval, yval, fluxval in zip(xvals, yvals, fluxvals):
        if not (np.isfinite(xval) and np.isfinite(yval)):
            continue
        flux_term = flux_scale_pix * np.log10(max(float(fluxval), 1.0))
        core_radius = max(float(core_radius_pix), core_radius_pix + flux_term)
        halo_radius = max(core_radius + 2.0, halo_radius_pix + flux_term)
        distance2 = (xx - xval) ** 2 + (yy - yval) ** 2
        mask |= distance2 <= halo_radius ** 2
        mask |= distance2 <= core_radius ** 2

    return mask


def postprocess_residual_image(data, fwhm_pix, negative_threshold=0.0, satstar_table=None,
                               core_radius_pix=12, halo_radius_pix=28, flux_scale_pix=1.0):
    processed = np.array(data, dtype=float, copy=True)
    kernel = Gaussian2DKernel(x_stddev=fwhm_pix / 2.355)

    if negative_threshold is not None:
        negative_mask = processed < negative_threshold
        if np.any(negative_mask):
            processed[negative_mask] = np.nan

    if satstar_table is not None:
        saturated_mask = build_hybrid_saturated_artifact_mask(
            processed.shape,
            satstar_table,
            core_radius_pix=core_radius_pix,
            halo_radius_pix=halo_radius_pix,
            flux_scale_pix=flux_scale_pix,
        )
        if np.any(saturated_mask):
            processed[saturated_mask] = np.nan

    if np.any(np.isnan(processed)):
        processed = interpolate_replace_nans(processed, kernel, convolve=convolve_fft)

    return processed


def compute_local_noise_map(data, smooth_sigma_pix=3.0):
    """
    Build a local noise map from an image using the sequence:
    1) Gaussian smooth
    2) high-pass residual = original - smooth
    3) local variance from smoothed residual**2
    4) local noise = sqrt(local variance)
    """
    image = np.asarray(np.nan_to_num(data), dtype=float)
    smoothed = ndimage.gaussian_filter(image, sigma=float(smooth_sigma_pix))
    residual = image - smoothed
    local_var = ndimage.gaussian_filter(residual ** 2, sigma=float(smooth_sigma_pix))
    local_var = np.where(local_var < 0, 0, local_var)
    noise_map = np.sqrt(local_var)
    return noise_map


def _sample_map_at_positions(image_map, xvals, yvals):
    xpix = np.rint(np.asarray(xvals, dtype=float)).astype(int)
    ypix = np.rint(np.asarray(yvals, dtype=float)).astype(int)

    sampled = np.full(len(xpix), np.nan, dtype=float)
    valid = ((xpix >= 0) & (ypix >= 0) &
             (ypix < image_map.shape[0]) & (xpix < image_map.shape[1]))
    sampled[valid] = image_map[ypix[valid], xpix[valid]]
    return sampled


def annotate_and_filter_by_local_snr(detection_table, noise_map, snr_threshold=5.0):
    tbl = _as_table(detection_table)
    if len(tbl) == 0:
        if 'local_noise' not in tbl.colnames:
            tbl['local_noise'] = np.array([], dtype=float)
        if 'local_snr' not in tbl.colnames:
            tbl['local_snr'] = np.array([], dtype=float)
        return tbl, {'input_count': 0, 'kept_count': 0, 'dropped_count': 0}

    xvals, yvals = _best_available_xy(tbl)
    local_noise = _sample_map_at_positions(noise_map, xvals, yvals)

    if 'peak' in tbl.colnames:
        signal = np.asarray(tbl['peak'], dtype=float)
    elif 'flux' in tbl.colnames:
        signal = np.asarray(tbl['flux'], dtype=float)
    elif 'flux_fit' in tbl.colnames:
        signal = np.asarray(tbl['flux_fit'], dtype=float)
    elif 'flux_init' in tbl.colnames:
        signal = np.asarray(tbl['flux_init'], dtype=float)
    else:
        signal = np.full(len(tbl), np.nan, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        local_snr = np.abs(signal) / local_noise

    tbl['local_noise'] = np.asarray(local_noise, dtype=float)
    tbl['local_snr'] = np.asarray(local_snr, dtype=float)

    keep = (np.isfinite(local_snr) & np.isfinite(local_noise) &
            (local_noise > 0) & (local_snr >= float(snr_threshold)))
    filtered = tbl[keep]
    stats = {
        'input_count': int(len(tbl)),
        'kept_count': int(np.sum(keep)),
        'dropped_count': int(len(tbl) - np.sum(keep)),
    }
    return filtered, stats


def load_or_make_satstar_catalog(filename, path_prefix, use_merged_psf_for_merged=False, overwrite=False,
                                 outside_star_pixels=None, outside_star_fit_box=512):
    satstar_filename = filename.replace('.fits', '_satstar_catalog.fits')
    if os.path.exists(satstar_filename) and not overwrite:
        return Table.read(satstar_filename)

    remove_saturated_stars(filename, overwrite=overwrite, path_prefix=path_prefix,
                           use_merged_psf_for_merged=use_merged_psf_for_merged,
                           outside_star_pixels=outside_star_pixels,
                           outside_star_fit_box=outside_star_fit_box)
    if os.path.exists(satstar_filename):
        return Table.read(satstar_filename)
    return None


def load_outside_fov_satstar_pixels(basepath, ww):
    regfn = f'{basepath}/regions_/saturated_stars_outside_fov.reg'
    if not os.path.exists(regfn):
        return []

    reglist = regions.Regions.read(regfn)
    outside_pixels = []
    for reg in reglist:
        preg = reg
        if hasattr(reg, 'to_pixel'):
            preg = reg.to_pixel(ww)

        center = getattr(preg, 'center', None)
        if center is None:
            continue

        xval = float(center.x)
        yval = float(center.y)
        if np.isfinite(xval) and np.isfinite(yval):
            outside_pixels.append((xval, yval))

    print(f"Loaded {len(outside_pixels)} outside-FOV saturated-star seeds from {regfn}", flush=True)
    return outside_pixels


def save_photutils_results(result, ww, filename,
                           im1, detector,
                           basepath, filtername, module, desat, bgsub, exposure_, visitid_, vgroupid_,
                           psf=None,
                           blur=False,
                           basic_or_iterative='basic',
                           options=None,
                           epsf_="",
                           group="",
                           fpsf="",
                           background_map=None,
                           iteration_label=None):
    print("Saving photutils results.")
    blur_ = "_blur" if blur else ""

    pixscale = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
    if 'x_fit' in result.colnames:
        if hasattr(result['x_fit'], 'mask'):
            bad = result['x_fit'].mask
        else:
            bad = ~np.isfinite(result['x_fit'])
        print(f'Found and removed {np.sum(bad)} bad fits out of {len(result)} total [fit resulted in masked x_fit, y_fit]', flush=True)
        result = result[~bad]
        coords = ww.pixel_to_world(result['x_fit'], result['y_fit'])
        result['skycoord_centroid'] = coords
    elif 'xcentroid' in result.colnames:
        coords = ww.pixel_to_world(result['xcentroid'], result['ycentroid'])
        result['skycoord_centroid'] = coords
    elif 'x_init' in result.colnames:
        coords = ww.pixel_to_world(result['x_init'], result['y_init'])
        result['skycoord_init'] = coords
    else:
        raise KeyError(f"No x value found in {result.colnames}")
    print(f'len(result) = {len(result)}, len(coords) = {len(coords)}, type(result)={type(result)}', flush=True)
    if options.each_exposure:
        result.meta['exposure'] = exposure_
    if visitid_ is not None and visitid_ != '':
        result.meta['visit'] = int(visitid_[-3:])
    if vgroupid_ is not None and vgroupid_ != '':
        result.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')

    result.meta['filename'] = filename
    result.meta['filter'] = filtername
    result.meta['module'] = module
    result.meta['detector'] = detector
    result.meta['pixscale'] = pixscale.to(u.deg).value
    result.meta['pixscale_as'] = pixscale.to(u.arcsec).value
    result.meta['proposal_id'] = options.proposal_id

    if 'RAOFFSET' in im1[0].header:
        result.meta['RAOFFSET'] = im1[0].header['RAOFFSET']
        result.meta['DEOFFSET'] = im1[0].header['DEOFFSET']
    elif 'RAOFFSET' in im1[1].header:
        result.meta['RAOFFSET'] = im1[1].header['RAOFFSET']
        result.meta['DEOFFSET'] = im1[1].header['DEOFFSET']

    if 'x_err' in result.colnames:
        result['dra'] = result['x_err'] * pixscale
        result['ddec'] = result['y_err'] * pixscale

    if iteration_label not in (None, ''):
        result.meta['iteration'] = str(iteration_label)

    if 'local_bkg' in result.colnames:
        result.meta['BKGCOL'] = 'local_bkg'
        result.meta['BKGMETH'] = 'photutils_local'
    else:
        xpos, ypos = _get_source_xy(result)
        result['local_bkg'] = _sample_background_map(background_map, xpos, ypos)
        result.meta['BKGCOL'] = 'local_bkg'
        result.meta['BKGMETH'] = 'bkg2d_sampled' if background_map is not None else 'none'

    iter_ = _iteration_token(iteration_label)
    tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_{basic_or_iterative}.fits"

    long_keys = [k for k in result.meta if len(k) > 8]
    for k in long_keys:
        result.meta[k[:8]] = result.meta[k]
        del result.meta[k]

    print(f"tblfilename={tblfilename}, filename={filename}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    result.write(tblfilename, overwrite=True)
    print(f"Completed {basic_or_iterative} photometry, and wrote out file {tblfilename}")

    return result


def save_crowdsource_results(results, ww, filename, suffix,
                             im1, detector,
                             basepath, filtername, module, desat, bgsub, exposure_, visitid_, vgroupid_,
                             psf=None,
                             blur=False,
                             options=None,
                             fpsf="",
                             iteration_label=None):
    print("Saving crowdsource results.")
    blur_ = "_blur" if blur else ""

    stars, modsky, skymsky, psf_ = results
    stars = Table(stars)
    coords = ww.pixel_to_world(stars['y'], stars['x'])
    stars['skycoord'] = coords
    stars['x'], stars['y'] = stars['y'], stars['x']
    stars['dx'], stars['dy'] = stars['dy'], stars['dx']

    pixscale = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
    stars['dra'] = stars['dx'] * pixscale
    stars['ddec'] = stars['dy'] * pixscale
    if visitid_ is not None and visitid_ != '':
        stars.meta['visit'] = int(visitid_[-3:])
    if vgroupid_ is not None and vgroupid_ != '':
        stars.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')
    stars.meta['filename'] = filename
    stars.meta['filter'] = filtername
    stars.meta['module'] = module
    stars.meta['detector'] = detector
    stars.meta['pixscale'] = pixscale.to(u.deg).value
    stars.meta['pixscale_as'] = pixscale.to(u.arcsec).value
    stars.meta['proposal_id'] = options.proposal_id
    if exposure_:
        stars.meta['exposure'] = exposure_
    if iteration_label not in (None, ''):
        stars.meta['iteration'] = str(iteration_label)
    if visitid_:
        stars.meta['visit'] = int(visitid_[-3:])
    if vgroupid_:
        stars.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')

    if 'RAOFFSET' in im1[0].header:
        stars.meta['RAOFFSET'] = im1[0].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[0].header['DEOFFSET']
    elif 'RAOFFSET' in im1[1].header:
        stars.meta['RAOFFSET'] = im1[1].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[1].header['DEOFFSET']

    iter_ = _iteration_token(iteration_label)
    tblfilename = (f"{basepath}/{filtername}/"
                   f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}"
                   f"_crowdsource_{suffix}.fits")

    print(f"tblfilename={tblfilename}, filename={filename}, suffix={suffix}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    stars.write(tblfilename, overwrite=True)
    with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
        fh[0].header.update(im1[1].header)
    skymskyhdu = fits.PrimaryHDU(data=skymsky, header=im1[1].header)
    modskyhdu = fits.ImageHDU(data=modsky, header=im1[1].header)
    hdul = fits.HDUList([skymskyhdu, modskyhdu])
    hdul.writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}_crowdsource_skymodel_{suffix}.fits", overwrite=True)

    if psf is not None:
        if hasattr(psf, 'stamp'):
            psfhdu = fits.PrimaryHDU(data=psf.stamp)
            psf_fn = (f"{basepath}/{filtername}/"
                      f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}"
                      f"_crowdsource_{suffix}_psf.fits")
            psfhdu.writeto(psf_fn, overwrite=True)
        else:
            raise ValueError(f"PSF did not have a stamp attribute.  It was: {psf}, type={type(psf)}")

    return stars


def load_data(filename):
    fh = fits.open(filename)
    im1 = fh
    data = im1['SCI'].data
    try:
        wht = im1['WHT'].data
    except KeyError:
        wht = None
    err = im1['ERR'].data
    instrument = im1[0].header['INSTRUME']
    telescope = im1[0].header['TELESCOP']
    obsdate = im1[0].header['DATE-OBS']
    return fh, im1, data, wht, err, instrument, telescope, obsdate


def get_psf_model(filtername, proposal_id, field,
                  module,
                  use_webbpsf=False,
                  obsdate=None,
                  use_grid=False,
                  blur=False,
                  target='brick',
                  stampsz=19,
                  oversample=1,
                  basepath='/blue/adamginsburg/adamginsburg/jwst/'):
    """
    Return two types of PSF model, the first for DAOPhot and the second for Crowdsource
    """

    basepath = f'{basepath}/{target}'

    blur_ = "_blur" if blur else ""

    # psf_fn = f'{basepath}/{instrument.lower()}_{filtername}_samp{oversample}_nspsf{npsf}_npix{fov_pixels}.fits'
    # if os.path.exists(str(psf_fn)):
    #     # As a file
    #     print(f"Loading grid from psf_fn={psf_fn}", flush=True)
    #     grid = to_griddedpsfmodel(psf_fn)  # file created 2 cells above
    #     if isinstance(big_grid, list):
    #         print(f"PSF IS A LIST OF GRIDS!!! this is incompatible with the return from nrc.psf_grid")
    #         grid = grid[0]

    # TODO: factor this out into its own downloading function and make it work with NIRCAM and MIRI both
    if use_webbpsf:
        with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
            api_token = fh.read().strip()
        from astroquery.mast import Mast

        for ii in range(10):
            try:
                Mast.login(api_token.strip())
                break
            except (requests.exceptions.ReadTimeout, urllib3.exceptions.ReadTimeoutError, TimeoutError) as ex:
                print(f"Attempt {ii} to log in to MAST: {ex}")
                time.sleep(5)
        os.environ['MAST_API_TOKEN'] = api_token.strip()

        has_downloaded = False
        ntries = 0
        while not has_downloaded:
            ntries += 1
            try:
                print("Attempting to download WebbPSF data", flush=True)
                nrc = webbpsf.NIRCam()
                nrc.load_wss_opd_by_date(f'{obsdate}T00:00:00')
                nrc.filter = filtername
                if module in ('nrca', 'nrcb'):
                    if 'F4' in filtername.upper() or 'F3' in filtername.upper():
                        nrc.detector = f'{module.upper()}5' # I think NRCA5 must be the "long" detector?
                    else:
                        nrc.detector = f'{module.upper()}1' #TODO: figure out a way to use all 4?
                    # default oversampling is 4
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True)
                elif 'nrc' in module:
                    # Allow nrca1, nrca2, ...
                    nrc.detector = module.upper()
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True)
                else:
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=True, verbose=True, save=True)
                has_downloaded = True
            except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
                print(f"Failed to build PSF: {ex}", flush=True)
            except Exception as ex:
                print(ex, flush=True)
                if ntries > 10:
                    # avoid infinite loops
                    raise ValueError("Failed to download PSF, probably because of an error listed above")
                else:
                    continue

        if use_grid:
            return grid, WrappedPSFModel(grid, stampsz=stampsz)
        else:
            # there's no way to use a grid across all detectors.
            # the right way would be to use this as a grid of grids, but that apparently isn't supported.
            if isinstance(grid, list):
                grid = grid[0]

            #yy, xx = np.indices([31,31], dtype=float)
            #grid.x_0 = grid.y_0 = 15.5
            #psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx,yy))

            # bigger PSF probably needed
            yy, xx = np.indices([61, 61], dtype=float)
            grid.x_0 = grid.y_0 = 30
            psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx, yy))

            return grid, psf_model
    else:

        grid = psfgrid = to_griddedpsfmodel(f'{basepath}/psfs/{filtername.upper()}_{proposal_id}_{field}_merged_PSFgrid_oversample{oversample}{blur_}.fits')

        # if isinstance(grid, list):
        #     print(f"Grid is a list: {grid}")
        #     psf_model = WrappedPSFModel(grid[0])
        #     dao_psf_model = grid[0]
        # else:

        psf_model = WrappedPSFModel(grid, stampsz=stampsz)
        dao_psf_model = grid

        return grid, psf_model


def get_uncertainty(err, data, dq=None, wht=None):

    if dq is None:
        dq = np.zeros(data.shape, dtype='int')

    # crowdsource uses inverse-sigma, not inverse-variance
    weight = err**-1
    #maxweight = np.percentile(weight[np.isfinite(weight)], 95)
    #minweight = np.percentile(weight[np.isfinite(weight)], 5)
    #badweight =  np.percentile(weight[np.isfinite(weight)], 1)
    #weight[err < 1e-5] = 0
    #weight[(err == 0) | (wht == 0)] = np.nanmedian(weight)
    #weight[np.isnan(weight)] = 0
    bad = np.isnan(weight) | (data == 0) | np.isnan(data) | (weight == 0) | (err == 0)
    #if dq is not None:
    #    # only 0 is OK
    #    bad |= (dq != 0)
    if wht is not None:
        bad |= (wht == 0)

    #weight[weight > maxweight] = maxweight
    #weight[weight < minweight] = minweight
    # it seems that crowdsource doesn't like zero weights
    # may have caused broked f466n? weight[bad] = badweight
    #weight[bad] = minweight
    # crowdsource explicitly handles weight=0, so this _should_ work.
    weight[bad] = 0

    # Expand bad pixel zones for dq
    #bad_for_dq = ndimage.binary_dilation(bad, iterations=2)
    #dq[bad_for_dq] = 2 | 2**30 | 2**31
    #print(f"Total bad pixels = {bad.sum()}, total bad for dq={bad_for_dq.sum()}")

    return dq, weight, bad


def mosaic_each_exposure_residuals(basepath, filtername, proposal_id, field, module,
                                   residual_kind='iterative', desat=False, bgsub=False,
                                   epsf=False, blur=False, group=False, pupil='clear',
                                   iteration_label=None):
    """
    Resample per-exposure residual images into one JWST-style *_residual_i2d.fits product.
    """
    if residual_kind not in ('basic', 'iterative'):
        raise ValueError(f"residual_kind must be one of ('basic', 'iterative'), got {residual_kind}")

    pipeline_dir = f'{basepath}/{filtername}/pipeline'
    desat_ = '_unsatstar' if desat else ''
    bgsub_ = '_bgsub' if bgsub else ''
    epsf_ = '_epsf' if epsf else ''
    blur_ = '_blur' if blur else ''
    group_ = '_group' if group else ''
    iter_ = _iteration_token(iteration_label)

    if proposal_id == '3958' and field == '007' and filtername in ('F187N', 'F210M') and module == 'nrcb':
        module_patterns = [f'nrcb{number}' for number in range(1, 5)]
    else:
        module_patterns = [module]

    residual_files = []
    iter_regex = re.compile(r'_iter[^_]*_daophot_')
    iter_marker = f'{iter_}_daophot_' if iter_ else None
    flag_tokens = {
        '_unsatstar': desat,
        '_bgsub': bgsub,
        '_epsf': epsf,
        '_blur': blur,
        '_group': group,
    }

    def _matches_expected_tokens(residual_path):
        name = os.path.basename(residual_path)

        # Enforce exact flag matching: no accidental mixing of bgsub/non-bgsub,
        # desaturated/non-desaturated, etc.
        for token, enabled in flag_tokens.items():
            has_token = token in name
            if enabled and not has_token:
                return False
            if (not enabled) and has_token:
                return False

        # Enforce exact iteration matching: unlabeled mosaics must exclude iter* files.
        if iter_marker is None:
            if iter_regex.search(name):
                return False
        else:
            if iter_marker not in name:
                return False

        return True

    for module_pattern in module_patterns:
        residual_glob = (
            f'{pipeline_dir}/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-'
            f'{module_pattern}_visit*_vgroup*_exp*{desat_}{bgsub_}{epsf_}{blur_}{group_}'
            f'{iter_}_daophot_{residual_kind}_residual.fits'
        )
        residual_files.extend(glob.glob(residual_glob))
    residual_files = sorted(set(fn for fn in residual_files if _matches_expected_tokens(fn)))
    if len(residual_files) == 0:
        raise ValueError(
            f'No per-exposure residuals found for module={module} '
            f'patterns={module_patterns} filter={filtername} residual_kind={residual_kind}'
        )

    product_name = (
        f'jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}'
        f'{desat_}{bgsub_}{epsf_}{blur_}{group_}{iter_}_daophot_{residual_kind}_residual'
    )
    asn = asn_from_list.asn_from_list(
        residual_files,
        rule=DMS_Level3_Base,
        product_name=product_name,
    )

    asn_filename = f'{pipeline_dir}/{product_name}_asn.json'
    with open(asn_filename, 'w') as asn_fh:
        _, serialized = asn.dump()
        asn_fh.write(serialized)

    output_filename = f'{pipeline_dir}/{product_name}_i2d.fits'
    print(f'Resampling {len(residual_files)} residual exposures into {product_name}_i2d.fits')
    resampled = ResampleStep.call(asn_filename, output_dir=pipeline_dir, save_results=False)
    resampled.save(output_filename, overwrite=True)
    if hasattr(resampled, 'close'):
        resampled.close()

    if not os.path.exists(output_filename):
        raise FileNotFoundError(f'Expected output was not created: {output_filename}')
    print(f'Wrote residual mosaic {output_filename}')

    fwhm_tbl = Table.read(FWHM_TABLE)
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])
    with ImageModel(output_filename) as model:
        infilled_data = postprocess_residual_image(
            model.data,
            fwhm_pix,
            negative_threshold=0.0,
            satstar_table=None,
        )
        model.data = infilled_data
        infilled_filename = output_filename.replace('_residual_i2d.fits', '_residual_infilled_i2d.fits')
        model.save(infilled_filename, overwrite=True)
    print(f'Wrote residual infilled mosaic {infilled_filename}')
    return infilled_filename


def save_residual_datamodel(input_filename, output_filename, data):
    with ImageModel(input_filename) as model:
        model.data = data
        model.save(output_filename, overwrite=True)


def main(smoothing_scales={'f182m': 0.25, 'f187n':0.25, 'f212n':0.55,
                           'f410m': 0.55, 'f405n':0.55, 'f466n':0.55,
                           'f335m': 0.55, 'f470n': 0.55, 'f480m': 0.55},
        bg_boxsizes={'f182m': 19, 'f187n':11, 'f212n':11,
                     'f210m': 11,
                     'f410m': 11, 'f405n':11, 'f466n':11,
                     'f444w': 11, 'f356w':11, 'f335m': 11, 'f470n': 11, 'f480m': 11,
                     'f200w':19, 'f115w':19,
                    },
        crowdsource_default_kwargs={'maxstars': 500000, },
        ):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                    default='F466N,F405N,F410M',
                    help="filter name list", metavar="filternames")
    parser.add_option("-m", "--modules", dest="modules",
                    default='nrca,nrcb,merged',
                    help="module list", metavar="modules")
    parser.add_option("-d", "--desaturated", dest="desaturated",
                    default=False,
                    action='store_true',
                    help="use image with saturated stars removed?", metavar="desaturated")
    parser.add_option("--daophot", dest="daophot",
                    default=False,
                    action='store_true',
                    help="run daophot?", metavar="daophot")
    parser.add_option("--skip-crowdsource", dest="nocrowdsource",
                    default=False,
                    action='store_true',
                    help="skip crowdsource?", metavar="nocrowdsource")
    parser.add_option("--bgsub", dest="bgsub",
                    default=False,
                    action='store_true',
                    help="perform global background-subtraction first?", metavar="bgsub")
    parser.add_option("--epsf", dest="epsf",
                    default=False,
                    action='store_true',
                    help="try to make & use an ePSF?", metavar="epsf")
    parser.add_option("--blur", dest="blur",
                    default=False,
                    action='store_true',
                    help="blur the PSF?", metavar="blur")
    parser.add_option("--proposal_id", dest="proposal_id",
                    default='2221',
                    help="proposal_id", metavar="proposal_id")
    parser.add_option("--target", dest="target",
                    default='brick',
                    help="target", metavar="target")
    parser.add_option("--group", dest="group",
                      default=False,
                      action='store_true')
    parser.add_option('--each-exposure', dest='each_exposure',
                      default=False, action='store_true',
                      help='Photometer _each_ exposure?', metavar='each_exposure')
    parser.add_option('--each-suffix', dest='each_suffix',
                      default='destreak_o001_crf',
                      help='Suffix for the level-2 products', metavar='each_suffix')
    parser.add_option('--seed-catalog', dest='seed_catalog',
                      default='',
                      help='Optional seed catalog for a seeded photometry rerun', metavar='seed_catalog')
    parser.add_option('--iteration-label', dest='iteration_label',
                      default='',
                      help='Optional iteration label to embed in output filenames', metavar='iteration_label')
    parser.add_option('--postprocess-residuals', dest='postprocess_residuals',
                      default=False,
                      action='store_true',
                      help='Apply negative-pixel masking and saturated-star infill before detection')
    parser.add_option('--basic-only', dest='basic_only',
                      default=False,
                      action='store_true',
                      help='Run only BASIC daophot photometry and residual generation')
    parser.add_option('--residual-negative-threshold', dest='residual_negative_threshold',
                      default=0.0,
                      type='float',
                      help='Pixels below this threshold are replaced with Gaussian infill before detection')
    parser.add_option('--local-snr-threshold', dest='local_snr_threshold',
                      default=5.0,
                      type='float',
                      help='Per-source local S/N threshold for retaining DAO detections')
    parser.add_option('--daofind-roundlo', dest='daofind_roundlo',
                      default=-1.0,
                      type='float',
                      help='DAOStarFinder roundness lower bound')
    parser.add_option('--daofind-roundhi', dest='daofind_roundhi',
                      default=1.0,
                      type='float',
                      help='DAOStarFinder roundness upper bound')
    parser.add_option('--skip-mosaic-each-exposure-residuals',
                      dest='skip_mosaic_each_exposure_residuals',
                      default=False,
                      action='store_true',
                      help='After --each-exposure, resample all per-exposure residuals into a residual_i2d product by default; this parameter skips that step. Residual kinds are auto-determined based on enabled photometry types.')
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    proposal_id = options.proposal_id
    target = options.target

    nvisits = {'2221': {'brick': 1, 'cloudc': 2},
               '1182': {'brick': 2},
               '3958': {'sickle': 1},
               '2092': {'cloudef': 1},
               '4147': {'sgrc': 1},
               '5365': {'sgrb2': 1},
               '2045': {'arches': 1, 'quintuplet': 1},
               '1939': {'sgra': 1},
               }
    field_to_reg_mapping = {'2221': {'001': 'brick', '002': 'cloudc'},
                            '1182': {'004': 'brick'},
                            '3958': {'007': 'sickle'},
                            '2092': {'005': 'cloudef'},
                            '4147': {'012': 'sgrc'},
                            '5365': {'001': 'sgrb2'},
                            '2045': {'001': 'arches', '003': 'quintuplet'},
                            '1939': {'001': 'sgra'}}[proposal_id]
    reg_to_field_mapping = {v:k for k,v in field_to_reg_mapping.items()}
    field = reg_to_field_mapping[target]

    # Module restrictions per proposal/field/filter for single-module datasets
    # Sickle is NRCB-only (SUB640 subarray) but detectors differ by wavelength:
    # - Short-wavelength (F187N, F210M): nrcb1, nrcb2, nrcb3, nrcb4
    # - Long-wavelength (F335M, F470N, F480M): nrcb only
    modules_by_proposal_field_filter = {
        '3958': {
            '007': {
                'F187N': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
                'F210M': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
                'F335M': ('nrcb',),
                'F470N': ('nrcb',),
                'F480M': ('nrcb',),
            }
        }
    }
    # Check if there's a filter-specific policy
    allowed_modules = None
    if proposal_id in modules_by_proposal_field_filter:
        if field in modules_by_proposal_field_filter[proposal_id]:
            field_policy = modules_by_proposal_field_filter[proposal_id][field]
            # Check if any of the requested filters have a policy
            for filt in filternames:
                if filt in field_policy:
                    allowed_modules = field_policy[filt]
                    break
    
    if allowed_modules is not None:
        expanded_modules = []
        for module in modules:
            if proposal_id == '3958' and field == '007' and module in ('nrca', 'nrcb'):
                if any(filt in ('F187N', 'F210M') for filt in filternames):
                    expanded_modules.extend([f'{module}{number}' for number in range(1, 5)])
                    continue
            expanded_modules.append(module)

        filtered_modules = [module for module in expanded_modules if module in allowed_modules]
        if len(filtered_modules) == 0:
            raise ValueError(
                f"No requested modules are allowed for proposal_id={proposal_id} field={field} "
                f"Requested modules={modules}, expanded_modules={expanded_modules}, allowed modules={allowed_modules}"
            )
        if tuple(filtered_modules) != tuple(modules):
            print(
                f"Restricting modules for proposal_id={proposal_id} field={field} filters={filternames} "
                f"to {filtered_modules} because this dataset is explicitly single-module."
            )
        modules = filtered_modules

    if field_to_reg_mapping[field] in ('sickle', 'cloudef', 'sgrc', 'sgrb2', 'arches', 'quintuplet', 'sgra'):
        basepath = f'/orange/adamginsburg/jwst/{field_to_reg_mapping[field]}/'
    else:
        basepath = f'/blue/adamginsburg/adamginsburg/jwst/{field_to_reg_mapping[field]}/'

    pl.close('all')

    print(f"options: {options}")

    # need to have incrementing _before_ test
    index = -1

    for module in modules:
        detector = module # no sub-detectors for long-NIRCAM
        for filtername in filternames:
            if options.each_exposure:
                for visitid in range(1, nvisits[proposal_id][target] + 1):
                    visitid = f'{visitid:03d}'
                    filenames = get_filenames(basepath, filtername, proposal_id,
                                              field, visitid=visitid,
                                              each_suffix=options.each_suffix,
                                              module=module, pupil='clear')
                    if len(filenames) > 0:
                        print(f"Looping over filenames {filenames} for filter={filtername} proposal={proposal_id} field={field} visitid={visitid}")
                        # jw02221001001_07101_00024_nrcblong_destreak_o001_crf.fits
                        for filename in filenames:

                            index += 1
                            # enable array jobs
                            if os.getenv('SLURM_ARRAY_TASK_ID') is not None and int(os.getenv('SLURM_ARRAY_TASK_ID')) != index:
                                print(f'Task={os.getenv("SLURM_ARRAY_TASK_ID")} does not match index {index}')
                                continue

                            exposure_id = filename.split("_")[2]
                            visit_id = filename.split("_")[0][-3:]
                            vgroup_id = filename.split("_")[1]
                            do_photometry_step(options, filtername, module, detector,
                                               field, basepath, filename, proposal_id,
                                               crowdsource_default_kwargs, exposurenumber=int(exposure_id),
                                               visit_id=visit_id, vgroup_id=vgroup_id,
                                               use_webbpsf=True,
                                               bg_boxsizes=bg_boxsizes,
                                               seed_catalog=options.seed_catalog or None,
                                               iteration_label=options.iteration_label or None,
                                               postprocess_residuals=options.postprocess_residuals or bool(options.seed_catalog),
                                               residual_negative_threshold=options.residual_negative_threshold,
                                               local_snr_threshold=options.local_snr_threshold,
                                               daofind_roundlo=options.daofind_roundlo,
                                               daofind_roundhi=options.daofind_roundhi)

                if not options.skip_mosaic_each_exposure_residuals:
                    if os.getenv('SLURM_ARRAY_TASK_ID') is None:
                        # Determine which residual kinds to mosaic based on enabled photometry types
                        mosaic_residual_kinds = []
                        if options.daophot:
                            mosaic_residual_kinds = ['basic'] if options.basic_only else ['basic', 'iterative']
                        
                        for residual_kind in mosaic_residual_kinds:
                            mosaic_each_exposure_residuals(basepath=basepath,
                                                          filtername=filtername,
                                                          proposal_id=proposal_id,
                                                          field=field,
                                                          module=module,
                                                          residual_kind=residual_kind,
                                                          desat=options.desaturated,
                                                          bgsub=options.bgsub,
                                                          epsf=options.epsf,
                                                          blur=options.blur,
                                                          group=options.group,
                                                          pupil='clear',
                                                          iteration_label=options.iteration_label or None)
                    else:
                        print('Skipping residual mosaicking in SLURM array-task mode.')
            else:
                filename = get_filename(basepath, filtername, proposal_id, field, module, options=options, pupil='clear')
                do_photometry_step(options, filtername, module, detector, field,
                                   basepath, filename, proposal_id, crowdsource_default_kwargs,
                                   bg_boxsizes=bg_boxsizes,
                                   seed_catalog=options.seed_catalog or None,
                                   iteration_label=options.iteration_label or None,
                                   postprocess_residuals=options.postprocess_residuals or bool(options.seed_catalog),
                                   residual_negative_threshold=options.residual_negative_threshold,
                                   local_snr_threshold=options.local_snr_threshold,
                                   daofind_roundlo=options.daofind_roundlo,
                                   daofind_roundhi=options.daofind_roundhi
                                   )


def get_filenames(basepath, filtername, proposal_id, field, each_suffix, module, pupil='clear', visitid='001'):

    # jw01182004002_02101_00012_nrcalong_destreak_o004_crf.fits
    # jw02221001001_07101_00012_nrcalong_destreak_o001_crf.fits
    # jw02221001001_05101_00022_nrcb3_destreak_o001_crf.fits
    glstr = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}{field}{visitid}*{module}*{each_suffix}.fits'
    fglob = glob.glob(glstr)
    if len(fglob) == 0:
        raise ValueError(f"No matches found to {glstr}")
    else:
        return fglob


def get_filename(basepath, filtername, proposal_id, field, module, options, pupil='clear'):
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = '_bgsub' if options.bgsub else ''
    #epsf_ = "_epsf" if options.epsf else ""
    #blur_ = "_blur" if options.blur else ""

    filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d.fits'
    if os.path.exists(filename):
        return filename

    candidate_patterns = [
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_realigned-to-refcat.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d{desat}.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_F444W-{filtername.lower()}-{module}_nodestreak_realigned-to-refcat.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t*_nircam_*{filtername.lower()}*{module}*i2d*.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t*_nircam_*{filtername.lower()}*i2d*.fits',
        f'{basepath}/mastDownload/JWST/**/jw0{proposal_id}-o{field}_t*_nircam_*{filtername.lower()}*{module}*i2d*.fits',
        f'{basepath}/mastDownload/JWST/**/jw0{proposal_id}-o{field}_t*_nircam_*{filtername.lower()}*i2d*.fits',
    ]

    for glstr in candidate_patterns:
        fglob = glob.glob(glstr, recursive=True)
        if len(fglob) == 1:
            return fglob[0]
        if len(fglob) > 1:
            return sorted(fglob)[-1]

    raise ValueError(f"No input file found for filter={filtername} proposal={proposal_id} field={field} module={module} in {basepath}")


def do_photometry_step(options, filtername, module, detector, field, basepath,
                       filename, proposal_id, crowdsource_default_kwargs, exposurenumber=None,
                       visit_id=None, vgroup_id=None,
                       bg_boxsizes=None,
                       use_webbpsf=False,
                       nsigma=5,
                       local_snr_threshold=5.0,
                       daofind_roundlo=-1.0,
                       daofind_roundhi=1.0,
                       pupil='clear',
                       seed_catalog=None,
                       iteration_label=None,
                       postprocess_residuals=False,
                       residual_negative_threshold=0.0):
    """
    nsigma is the threshold to multiply the error estimate by to get the detection threshold
    """
    print(f"Starting {field} filter {filtername} module {module} detector {detector} {exposurenumber}", flush=True)
    fwhm_tbl = Table.read(FWHM_TABLE)
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    # redundant, saves me renaming variables....
    filt = filtername

    # file naming suffixes
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = '_bgsub' if options.bgsub else ''
    epsf_ = "_epsf" if options.epsf else ""
    exposure_ = f'_exp{exposurenumber:05d}' if exposurenumber is not None else ''
    visitid_ = f'_visit{int(visit_id):03d}' if visit_id is not None else ''
    vgroupid_, vgroup_numeric = normalize_vgroup_id(vgroup_id)
    blur_ = "_blur" if options.blur else ""
    group = "_group" if options.group else ""
    iter_ = _iteration_token(iteration_label)

    print(f"Starting cataloging on {filename}", flush=True)
    fh, im1, data, wht, err, instrument, telescope, obsdate = load_data(filename)
    background_map = None

    # set up coordinate system
    ww = wcs.WCS(im1[1].header)
    pixscale = ww.proj_plane_pixel_area()**0.5
    cen = ww.pixel_to_world(im1[1].shape[1]/2, im1[1].shape[0]/2)

    if options.bgsub:
        # background subtraction
        # see BackgroundEstimationExperiments.ipynb
        bkg = Background2D(data, box_size=bg_boxsizes[filt.lower()], bkg_estimator=MedianBackground())
        background_map = bkg.background
        fits.PrimaryHDU(data=bkg.background,
                        header=im1['SCI'].header).writeto(filename.replace(".fits",
                                                                           "_background.fits"),
                                                          overwrite=True)

        # subtract background, but then re-zero the edges
        zeros = data == 0
        data = data - bkg.background
        data[zeros] = 0

        fits.PrimaryHDU(data=data, header=im1['SCI'].header).writeto(filename.replace(".fits", "_bgsub.fits"), overwrite=True)

    # try to limit memory use before we start photometry
    data = data.astype('float32')

    # Load PSF model
    grid, psf_model = get_psf_model(filtername, proposal_id, field,
                                    module=module,
                                    use_webbpsf=use_webbpsf,
                                    # if we're doing each exposure, we want the full grid
                                    use_grid=options.each_exposure,
                                    blur=options.blur,
                                    target=options.target,
                                    obsdate=obsdate,
                                    basepath='/blue/adamginsburg/adamginsburg/jwst/')
    dao_psf_model = grid

    # bound the flux to be >= 0 (no negative peak fitting)
    dao_psf_model.flux.min = 0

    dq, weight, bad = get_uncertainty(err, data, wht=wht, dq=im1['DQ'].data if 'DQ' in im1 else None)

    filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
    filter_table.add_index('filterID')
    instrument = 'NIRCam'
    eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filt}']['WavelengthEff'] * u.AA

    # DAO Photometry setup
    grouper = SourceGrouper(2 * fwhm_pix)
    mmm_bkg = MMMBackground()

    # empirically determined in debugging session with Taehwa on 2025-12-09:
    # with just nan_to_num, setting pixels to zero, some stars got "erased"
    kernel = Gaussian2DKernel(x_stddev=fwhm_pix/2.355)
    mask = np.isnan(data)
    if 'DQ' in im1:
        dqarr = im1['DQ'].data
        is_saturated = (dqarr & dqflags.pixel['SATURATED']) != 0
        # we want original data_ to be untouched for imshowing diagnostics etc.
        data_ = data.copy()
        data_[is_saturated] = np.nan
        mask |= is_saturated
    else:
        data_ = data

    nan_replaced_data = interpolate_replace_nans(data_, kernel, convolve=convolve_fft)

    if seed_catalog is None and iteration_label not in (None, ''):
        inferred_seed_catalog = (
            f'{basepath}/{filtername}/'
            f'{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_basic.fits'
        )
        if os.path.exists(inferred_seed_catalog):
            seed_catalog = inferred_seed_catalog

    is_second_iteration = seed_catalog is not None
    if is_second_iteration:
        # Second-iteration tuned finder settings: lower local-SNR cut plus tighter
        # roundness/sharpness to suppress diffuse-background false detections.
        iter2_local_snr_threshold = 3.0
        iter2_roundlo = -0.3
        iter2_roundhi = 0.3
        iter2_sharplo = 0.50
        iter2_sharphi = 1.00

        # Local-noise-map DAO thresholding for second-iteration residual search.
        local_noise_map = compute_local_noise_map(nan_replaced_data, smooth_sigma_pix=3.0)
        finite_noise = np.isfinite(local_noise_map) & (local_noise_map > 0)
        if not np.any(finite_noise):
            raise ValueError('Local noise map has no positive finite values')
        daofind_threshold = float(np.nanmin(local_noise_map[finite_noise]))
        daofind_tuned = DAOStarFinder(threshold=daofind_threshold,
                                      fwhm=fwhm_pix, roundhi=iter2_roundhi, roundlo=iter2_roundlo,
                                      sharplo=iter2_sharplo, sharphi=iter2_sharphi)
        print(
            f'DAO iter2 local-noise threshold={daofind_threshold}; '
            f'local_snr_threshold={iter2_local_snr_threshold}; '
            f'roundlo={iter2_roundlo}; roundhi={iter2_roundhi}; '
            f'sharplo={iter2_sharplo}; sharphi={iter2_sharphi}',
            flush=True,
        )
    else:
        # Keep original first-pass starfinding behavior unchanged.
        filtered_errest = np.nanmedian(err)
        print(f'Error estimate for DAO from median(err): {filtered_errest}', flush=True)
        # sigma_clipped stats get _much_ lower uncertainty for frames dominated by extended emission (maybe?).  At least, Sickle F470N had 3x too high error
        mean, med, std = stats.sigma_clipped_stats(data, stdfunc='mad_std')
        print(f'Error estimate for DAO from stats.: std={std}', flush=True)
        filtered_errest = min([filtered_errest, std])

        daofind_threshold = nsigma * filtered_errest
        daofind_tuned = DAOStarFinder(threshold=daofind_threshold,
                                      fwhm=fwhm_pix, roundhi=daofind_roundhi, roundlo=daofind_roundlo,
                                      sharplo=0.30, sharphi=1.40)
        print(
            f'DAO first-pass threshold={daofind_threshold}; '
            f'roundlo={daofind_roundlo}; roundhi={daofind_roundhi}',
            flush=True,
        )

    print("Finding stars with daofind_tuned", flush=True)

    satstar_table = None
    if options.each_exposure and seed_catalog is not None:
        outside_star_pixels = load_outside_fov_satstar_pixels(basepath, ww)
        satstar_table = load_or_make_satstar_catalog(
            filename,
            path_prefix=f'{basepath}/psfs',
            use_merged_psf_for_merged=(module == 'merged'),
            overwrite=bool(outside_star_pixels),
            outside_star_pixels=outside_star_pixels,
            outside_star_fit_box=512,
        )

    seeded_init_params = None
    if seed_catalog is not None:
        preferred_seed_skycoord_col = f'skycoord_{filtername.lower()}'
        merged_seed_table = _as_table(seed_catalog)
        seed_catalog = _combine_seed_and_satstars(seed_catalog, satstar_table)
        seed_after_sat_table = _as_table(seed_catalog)
        sat_seed_count = int(np.sum(np.asarray(seed_after_sat_table['is_saturated'], dtype=bool)))
        nonsat_seed_count = int(len(seed_after_sat_table) - sat_seed_count)
        detection_image = nan_replaced_data
        if postprocess_residuals:
            detection_image = postprocess_residual_image(
                nan_replaced_data,
                fwhm_pix,
                negative_threshold=residual_negative_threshold,
                satstar_table=satstar_table,
            )
        if postprocess_residuals:
            extra_noise_map = compute_local_noise_map(detection_image, smooth_sigma_pix=3.0)
            finite_extra_noise = np.isfinite(extra_noise_map) & (extra_noise_map > 0)
            if not np.any(finite_extra_noise):
                raise ValueError('Postprocessed local noise map has no positive finite values')
            extra_noise_floor = float(np.nanmin(extra_noise_map[finite_extra_noise]))
            extra_finder = DAOStarFinder(threshold=extra_noise_floor,
                                         fwhm=fwhm_pix, roundhi=iter2_roundhi, roundlo=iter2_roundlo,
                                         sharplo=iter2_sharplo, sharphi=iter2_sharphi)
            extra_detections = extra_finder(detection_image, mask=mask)
            extra_noise_for_snr = extra_noise_map
            print(f'Postprocessed DAO local-noise threshold: {extra_noise_floor}', flush=True)
        else:
            extra_detections = daofind_tuned(detection_image, mask=mask)
            extra_noise_for_snr = local_noise_map

        if extra_detections is None:
            extra_detections = Table()
        extra_detections, extra_snr_stats = annotate_and_filter_by_local_snr(
            extra_detections,
            extra_noise_for_snr,
            snr_threshold=iter2_local_snr_threshold,
        )
        print(
            'Extra DAO detections local-SNR filter: '
            f'in={extra_snr_stats["input_count"]} '
            f'kept={extra_snr_stats["kept_count"]} '
            f'dropped={extra_snr_stats["dropped_count"]}',
            flush=True,
        )
        seed_catalog, seed_aug_stats = _augment_seed_catalog_with_detections_sky(
            seed_catalog,
            extra_detections,
            ww=ww,
            match_radius_pix=max(1.0, 0.5 * fwhm_pix),
            preferred_seed_skycoord_col=preferred_seed_skycoord_col,
            return_stats=True,
        )
        print(
            'Seed composition: '
            f'merged_seed_rows={len(merged_seed_table)} '
            f'sat_seed_rows={sat_seed_count} '
            f'nonsat_seed_rows={nonsat_seed_count} '
            f'dao_detect_total={seed_aug_stats["detection_input"]} '
            f'dao_detect_finite_xy={seed_aug_stats["detection_finite_xy"]} '
            f'dao_added={seed_aug_stats["detection_added"]} '
            f'dao_rejected_duplicates={seed_aug_stats["detection_rejected_match"]} '
            f'seed_rows_final={len(_as_table(seed_catalog))}'
        )
        finstars = SeededFinder(seed_catalog, ww=ww,
                                preferred_skycoord_col=preferred_seed_skycoord_col)(nan_replaced_data, mask=mask)
        seeded_init_params = Table()
        seeded_init_params['x_init'] = np.asarray(finstars['x_init'], dtype=float)
        seeded_init_params['y_init'] = np.asarray(finstars['y_init'], dtype=float)
        seeded_init_params['flux_init'] = np.asarray(finstars['flux_init'], dtype=float)
        finding_label = 'seeded'
    else:
        finstars = daofind_tuned(nan_replaced_data,
                                 mask=mask)
        if finstars is None:
            finstars = Table()
        finding_label = 'daofind'

    print(f"Found {len(finstars)} with daofind_tuned", flush=True)
    # for diagnostic plotting convenience
    finstars['x'] = finstars['xcentroid']
    finstars['y'] = finstars['ycentroid']
    finstars['skycoord'] = ww.pixel_to_world(finstars['x'], finstars['y'])

    result = save_photutils_results(finstars, ww, filename,
                                    im1=im1, detector=detector,
                                    basepath=basepath,
                                    filtername=filtername, module=module,
                                    desat=desat, bgsub=bgsub,
                                    blur="",
                                    exposure_=exposure_,
                                    visitid_=visitid_, vgroupid_=vgroupid_,
                                    basic_or_iterative=finding_label,
                                    options=options,
                                    epsf_="",
                                    fpsf="",
                                    group=group,
                                    psf=None,
                                    background_map=background_map,
                                    iteration_label=iteration_label)

    stars = finstars # because I'm copy-pasting code...

    # Set up visualization
    reg = regions.RectangleSkyRegion(center=cen, width=1.5*u.arcmin, height=1.5*u.arcmin)
    preg = reg.to_pixel(ww)
    #mask = preg.to_mask()
    #cutout = mask.cutout(im1[1].data)
    #err = mask.cutout(im1[2].data)
    region_list = [y for x in glob.glob('regions_/*zoom*.reg') for y in
                   regions.Regions.read(x)]
    zoomcut_list = {reg.meta['text']: reg.to_pixel(ww).to_mask().get_overlap_slices(data.shape)[0]
                    for reg in region_list}
    zoomcut_list = {nm:slc for nm,slc in zoomcut_list.items()
                    if slc is not None and
                    slc[0].start > 0 and slc[1].start > 0
                    and slc[0].stop < data.shape[0] and slc[1].stop < data.shape[1]}

    zoomcut = slice(128, 256), slice(128, 256)
    modsky = data*0 # no model for daofind
    nullslice = (slice(None), slice(None))

    try:
        catalog_zoom_diagnostic(data, modsky, nullslice, stars)
        pl.suptitle(f"daofind Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_daofind.png',
                bbox_inches='tight')

        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
        pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom_daofind.png',
                bbox_inches='tight')

        for name, zoomcut in zoomcut_list.items():
            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub} zoom {name}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom{name.replace(" ","_")}_daofind.png',
                    bbox_inches='tight')
    except Exception as ex:
        print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for basic daofinder: {ex}')

    if not options.nocrowdsource:

        t0 = time.time()

        if False: # why do the unweighted version?
            print()
            print("starting crowdsource unweighted", flush=True)
            results_unweighted = fit_im(nan_replaced_data, psf_model,
                                        weight=np.ones_like(data)*np.nanmedian(weight)*(~mask),
                                        # psfderiv=np.gradient(-psf_initial[0].data),
                                        dq=dq,
                                        nskyx=0, nskyy=0, refit_psf=False, verbose=True,
                                        **crowdsource_default_kwargs,
                                        )
            print(f"Done with unweighted crowdsource. dt={time.time() - t0}")
            stars, modsky, skymsky, psf = results_unweighted
            stars = save_crowdsource_results(results_unweighted, ww, filename,
                                             im1=im1, detector=detector,
                                             basepath=basepath,
                                             filtername=filtername, module=module,
                                             desat=desat, bgsub=bgsub,
                                             blur=options.blur,
                                             exposure_=exposure_,
                                             visitid_=visitid_,
                                             vgroupid_=vgroupid_,
                                             options=options,
                                             suffix="unweighted", psf=None,
                                             iteration_label=iteration_label)

            zoomcut = slice(128, 256), slice(128, 256)

            try:
                catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_unweighted.png',
                        bbox_inches='tight')

                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom_unweighted.png',
                        bbox_inches='tight')
                for name, zoomcut in zoomcut_list.items():
                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Crowdsource nsky=0 Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_} zoom {name}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom{name.replace(" ","_")}_unweighted.png',
                            bbox_inches='tight')
            except Exception as ex:
                print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for unweighted crowdsource: {ex}')
                exc_tb = sys.exc_info()[2]
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

            fig = pl.figure(0, figsize=(10,10))
            fig.clf()
            ax = fig.gca()
            im = ax.imshow(weight, norm=simple_norm(weight, stretch='log'))
            pl.colorbar(mappable=im)
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_weights.png',
                    bbox_inches='tight')

        #for refit_psf, fpsf in zip((False, True), ('', '_fitpsf',)):
        for refit_psf, fpsf in zip((False, ), ('', )):
            for nsky in (0, ): #1, ):
                t0 = time.time()
                print()
                print(f"Running crowdsource fit_im with weights & nskyx=nskyy={nsky} & fpsf={fpsf} & blur={blur_}")
                print(f"data.shape={data.shape} weight_shape={weight.shape}", flush=True)
                results = fit_im(nan_replaced_data, psf_model, weight=weight * (~mask),
                                 nskyx=nsky, nskyy=nsky, refit_psf=refit_psf, verbose=True,
                                 dq=dq,
                                 **crowdsource_default_kwargs
                                 )
                print(f"Done with weighted, refit={fpsf}, nsky={nsky} crowdsource. dt={time.time() - t0}")
                stars, modsky, skymsky, psf = results
                stars = save_crowdsource_results(results, ww, filename,
                                                 im1=im1, detector=detector,
                                                 basepath=basepath,
                                                 filtername=filtername,
                                                 module=module, desat=desat,
                                                 bgsub=bgsub, fpsf=fpsf,
                                                 blur=options.blur,
                                                 exposure_=exposure_,
                                                 visitid_=visitid_,
                                                 vgroupid_=vgroupid_,
                                                 psf=psf if refit_psf else None,
                                                 options=options,
                                                 suffix=f"nsky{nsky}",
                                                 iteration_label=iteration_label)

                zoomcut = slice(128, 256), slice(128, 256)

                try:
                    catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                    pl.suptitle(f"Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics.png',
                            bbox_inches='tight')

                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom.png',
                            bbox_inches='tight')

                    for name, zoomcut in zoomcut_list.items():
                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"Crowdsource nsky={nsky} weighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} zoom {name}")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom{name.replace(" ","_")}.png',
                                bbox_inches='tight')
                except Exception as ex:
                    print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for crowdsource nsky={nsky} refitpsf={refit_psf} blur={options.blur}: {ex}')
                    exc_tb = sys.exc_info()[2]
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

    if options.daophot:
        t0 = time.time()
        print("Starting basic PSF photometry", flush=True)

        basic_finder = None if seeded_init_params is not None else daofind_tuned
        phot_basic = PSFPhotometry(finder=basic_finder,
                                   # 6,10 avoids the first sidelobe/airy ring
                                   # it's not optimal b/c the background variation is significant over a bigger scale...
                                   localbkg_estimator=LocalBackground(6, 10),
                                   grouper=grouper if options.group else None,
                                   psf_model=dao_psf_model,
                                   fitter=LevMarLSQFitter(),
                                   fit_shape=(5, 5),
                                   aperture_radius=2*fwhm_pix,
                                   progress_bar=True,
                                  )

        print("About to do BASIC photometry....")
        if seeded_init_params is not None:
            result = phot_basic(nan_replaced_data, mask=mask, init_params=seeded_init_params)
        else:
            result = phot_basic(nan_replaced_data, mask=mask)
        print(f"Done with BASIC photometry. len(result)={len(result)}  dt={time.time() - t0}")

        result = save_photutils_results(result, ww, filename,
                                        im1=im1, detector=detector,
                                        basepath=basepath,
                                        filtername=filtername, module=module,
                                        desat=desat, bgsub=bgsub,
                                        blur=options.blur,
                                        exposure_=exposure_,
                                        visitid_=visitid_,
                                        vgroupid_=vgroupid_,
                                        basic_or_iterative='basic',
                                        options=options,
                                        epsf_=epsf_,
                                        group=group,
                                        psf=None,
                                        background_map=background_map,
                                        iteration_label=iteration_label)

        stars = result
        stars['x'] = stars['x_fit']
        stars['y'] = stars['y_fit']
        print("Creating BASIC residual image, using 21x21 patches")
        modsky = phot_basic.make_model_image(data.shape, psf_shape=(21, 21), include_localbkg=False)
        residual = data - modsky
        print("Done creating BASIC residual image, using 21x21 patches")
        save_residual_datamodel(
            filename,
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_basic_residual.fits',
            residual,
        )
        save_residual_datamodel(
            filename,
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_basic_model.fits',
            modsky,
        )
        print("Saved BASIC residual image, now making diagnostics.")
        catalog_zoom_diagnostic(data, modsky, nullslice, stars)
        pl.suptitle(f"daophot basic Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_basic.png',
                bbox_inches='tight')

        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
        pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_basic.png',
                bbox_inches='tight')

        for name, zoomcut in zoomcut_list.items():
            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_basic{name.replace(" ","_")}.png',
                    bbox_inches='tight')

        print(f"Done with diagnostics for BASIC photometry.  dt={time.time() - t0}")
        pl.close('all')

        if not options.basic_only:
            t0 = time.time()
            print("Iterative PSF photometry")
            if options.epsf:
                print("Building EPSF")
                epsf_builder = EPSFBuilder(oversampling=3, maxiters=10,
                                           smoothing_kernel='quadratic',
                                           progress_bar=True)

                epsfsel = ((finstars['peak'] > 200) &
                           (finstars['roundness1'] > -0.25) &
                           (finstars['roundness1'] < 0.25) &
                           (finstars['roundness2'] > -0.25) &
                           (finstars['roundness2'] < 0.25) &
                           (finstars['sharpness'] > 0.4) &
                           (finstars['sharpness'] < 0.8))

                print(f"Extracting {epsfsel.sum()} stars")
                stars = extract_stars(NDData(data=nan_replaced_data), finstars[epsfsel], size=35)

                for star in stars:
                    background = np.nanpercentile(star.data, 5)
                    star.data[:] -= background

                epsf, fitted_stars = epsf_builder(stars)
                epsf._data = epsf.data[2:-2, 2:-2]

                norm = simple_norm(epsf.data, 'log', percent=99.0)
                pl.figure(1).clf()
                pl.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
                pl.colorbar()
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_epsf.png',
                           bbox_inches='tight')
                dao_psf_model = epsf

            phot_iter = IterativePSFPhotometry(finder=daofind_tuned,
                                               localbkg_estimator=LocalBackground(6, 10),
                                               grouper=grouper if options.group else None,
                                               psf_model=dao_psf_model,
                                               fitter=LevMarLSQFitter(),
                                               maxiters=5,
                                               fit_shape=(5, 5),
                                               sub_shape=(15, 15),
                                               aperture_radius=2*fwhm_pix,
                                               progress_bar=True,
                                              )

            print("About to do ITERATIVE photometry....")
            if seeded_init_params is not None:
                result2 = phot_iter(nan_replaced_data, mask=mask, init_params=seeded_init_params)
            else:
                result2 = phot_iter(nan_replaced_data, mask=mask)
            print(f"Done with ITERATIVE photometry. len(result2)={len(result2)}  dt={time.time() - t0}")

            result2 = save_photutils_results(result2, ww, filename,
                                             im1=im1, detector=detector,
                                             basepath=basepath,
                                             filtername=filtername, module=module,
                                             desat=desat, bgsub=bgsub,
                                             blur=options.blur,
                                             exposure_=exposure_,
                                             visitid_=visitid_,
                                             vgroupid_=vgroupid_,
                                             basic_or_iterative='iterative',
                                             options=options,
                                             epsf_=epsf_,
                                             group=group,
                                             psf=None,
                                             background_map=background_map,
                                             iteration_label=iteration_label)

            stars = result2
            stars['x'] = stars['x_fit']
            stars['y'] = stars['y_fit']

            print("Creating iterative residual")
            modsky = phot_iter.make_model_image(data.shape, psf_shape=(21, 21), include_localbkg=False)
            residual = data - modsky
            print("finished iterative residual")
            save_residual_datamodel(
                filename,
                f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_iterative_residual.fits',
                residual,
            )
            save_residual_datamodel(
                filename,
                f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_iterative_model.fits',
                modsky,
            )
            print("Saved iterative residual")
            catalog_zoom_diagnostic(data, modsky, nullslice, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_iterative.png',
                    bbox_inches='tight')

            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_iterative.png',
                    bbox_inches='tight')

            for name, zoomcut in zoomcut_list.items():
                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_iterative{name.replace(" ","_")}.png',
                        bbox_inches='tight')

            print(f"Done with diagnostics for ITERATIVE photometry.  dt={time.time() - t0}")
            pl.close('all')
        else:
            print("Skipping ITERATIVE photometry because --basic-only was requested")


if __name__ == "__main__":
    main()
