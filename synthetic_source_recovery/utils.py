"""
Shared utilities for synthetic source recovery tests.

Physical calibration anchored to Brick program 2221 (JWST/NIRCam):
  PHOTMJSR  = 1.964 MJy sr⁻¹ per (e⁻/s) for F200W combined mosaic
  PIXAR_SR  = 2.274e-14 sr pixel⁻¹  →  pixel scale ~0.0631 arcsec
  Background noise std (combined mosaic): ~101 MJy/sr

Saturation calibration (from catalog, sources flagged is_saturated_f200w):
  F200W: saturates at ~14.5 Vega mag  (wide band, most photons)
  F182M: saturates at ~14.2 Vega mag  (medium band)
  F212N: saturates at ~10.5 Vega mag  (narrow band, ~10× fewer photons)

Working units in synthetic images: MJy/sr per pixel, matching the mosaic.
"""

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from stpsf.utils import to_griddedpsfmodel

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PSF_DIR = "/orange/adamginsburg/jwst/brick/psfs"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ──────────────────────────────────────────────────────────────────────────────
# Photometric calibration (Brick F200W mosaic)
# ──────────────────────────────────────────────────────────────────────────────
PHOTMJSR  = 1.964        # MJy/sr per (e-/s)
PIXAR_SR  = 2.274e-14    # sr / pixel
BKG_NOISE_MJYSR = 101.0  # 1-sigma background noise in MJy/sr (combined mosaic)

# Vega flux densities (Jy) from filter effective wavelengths
VEGA_FLUX_JY = {
    "F200W": 1720.0,   # broad, ~2.0 µm
    "F182M": 1880.0,   # medium, ~1.82 µm
    "F212N": 1620.0,   # narrow, ~2.12 µm
}

# Saturation magnitude (Vega) — peak pixel reaches full well in one exposure
# Calibrated from catalog: is_saturated_f<filter> flag onset
SAT_MAG_VEGA = {
    "F200W": 14.5,
    "F182M": 14.2,
    "F212N": 10.5,   # narrow band ≈ 4 mag harder to saturate
}

# Noise scale: individual exposure noise relative to combined mosaic
# The combined mosaic has ~16 combined exposures; individual noise is ~4× higher
N_EXPOSURES = 16
SINGLE_EXP_NOISE_FACTOR = np.sqrt(N_EXPOSURES)


# ──────────────────────────────────────────────────────────────────────────────
# PSF loading
# ──────────────────────────────────────────────────────────────────────────────

def load_psf(filtername, detector="nrca1", psf_dir=PSF_DIR):
    """
    Load a pre-computed STPSF GriddedPSFModel.

    F200W has no pre-computed file; falls back to F182M (1.82 µm vs 2.0 µm —
    diffraction-limited PSF differs by <10% in FWHM, adequate for saturation tests).
    """
    fn = os.path.join(psf_dir, f"nircam_{detector}_{filtername.lower()}_fovp512_samp2_npsf16.fits")
    if not os.path.exists(fn):
        if filtername.upper() == "F200W":
            fallback = fn.replace("f200w", "f182m")
            print(f"No F200W PSF found; using F182M as proxy: {fallback}")
            fn = fallback
        else:
            raise FileNotFoundError(f"PSF file not found: {fn}")
    return to_griddedpsfmodel(fn)


def get_psf_peak_fraction(psf_model, imsize=512, center=None):
    """
    Evaluate the normalized PSF at image center and return the peak pixel
    fraction (fraction of total PSF flux in the brightest pixel).
    """
    if center is None:
        center = (imsize // 2, imsize // 2)
    y, x = np.mgrid[0:imsize, 0:imsize]
    psf_img = psf_model(x - center[1], y - center[0])
    psf_img = psf_img / psf_img.sum()   # re-normalise to sum=1
    return float(psf_img.max()), psf_img


# ──────────────────────────────────────────────────────────────────────────────
# Flux conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def vega_to_total_flux_mjysr(mag_vega, filtername):
    """
    Convert a Vega magnitude to total integrated PSF flux in MJy/sr × pixel
    (i.e., the sum over all pixels of the PSF image in MJy/sr units).

    Uses:  flux_jy = F_vega × 10^(-mag/2.5)
           surface brightness (MJy/sr) = flux_jy / (1e6 × PIXAR_SR)
    The 'total PSF flux' is that surface-brightness integral, equal to
    flux_jy / (1e6 × PIXAR_SR) per pixel when the PSF sums to 1 pixel.
    """
    f_vega_jy = VEGA_FLUX_JY[filtername.upper()]
    flux_jy = f_vega_jy * 10 ** (-0.4 * mag_vega)
    # surface brightness contribution per pixel (when PSF integrates to 1)
    flux_mjysr_total = flux_jy / (1e6 * PIXAR_SR)
    return flux_mjysr_total


def saturation_threshold_mjysr(filtername):
    """
    Peak-pixel saturation threshold in MJy/sr.

    Derived from SAT_MAG_VEGA: at the saturation magnitude, the peak pixel
    equals this threshold.  For stars brighter than SAT_MAG_VEGA, the core
    region exceeds this threshold and is flagged SATURATED in the DQ array.
    """
    sat_mag = SAT_MAG_VEGA[filtername.upper()]
    total_flux = vega_to_total_flux_mjysr(sat_mag, filtername)
    # need PSF peak fraction — use a representative value
    # For NIRCam SW ~2 µm, FWHM ~2 pix: peak fraction ≈ 0.20
    peak_frac_approx = 0.20
    return total_flux * peak_frac_approx


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic image builder
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_image(mag_vega, filtername, psf_model,
                          imsize=300, noise_sigma=None, rng=None):
    """
    Build a synthetic NIRCam image of a single point source.

    Parameters
    ----------
    mag_vega : float
        Vega magnitude of the star.
    filtername : str
        Filter name ('F200W', 'F182M', or 'F212N').
    psf_model : GriddedPSFModel
        Normalised PSF model (sum ≈ 1 over a large aperture).
    imsize : int
        Square image side (pixels).
    noise_sigma : float or None
        1-sigma background noise in MJy/sr.  Defaults to the combined-mosaic
        value scaled to a single exposure (≈ 4× larger).
    rng : numpy.random.Generator or None

    Returns
    -------
    sci : 2-D float array, MJy/sr
    dq  : 2-D int array  (SATURATED bit = 2, matching JWST DQ convention)
    true_flux : float, total PSF flux in MJy/sr (sum over all pixels)
    sat_radius_pix : float, approximate radius of saturated core
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if noise_sigma is None:
        noise_sigma = BKG_NOISE_MJYSR * SINGLE_EXP_NOISE_FACTOR

    center = imsize // 2
    y, x = np.mgrid[0:imsize, 0:imsize]

    # Evaluate PSF and normalise
    psf_img = psf_model(x - center, y - center)
    psf_img = np.clip(psf_img, 0, None)
    psf_img /= psf_img.sum()

    # Scale to physical flux
    total_flux = vega_to_total_flux_mjysr(mag_vega, filtername)
    star_image = psf_img * total_flux

    # Background (constant sky) + Poisson noise + read noise
    sky_level = 1.7   # MJy/sr, typical Brick background
    noise = rng.normal(0.0, noise_sigma, size=(imsize, imsize))
    sci = star_image + sky_level + noise

    # ── Saturation ──────────────────────────────────────────────────────────
    sat_thresh = saturation_threshold_mjysr(filtername)
    # recompute with the actual PSF peak to be precise
    peak_frac = psf_img.max()
    sat_thresh_actual = total_flux * peak_frac  # what the peak pixel would be
    # but the threshold is fixed by the detector, not by this star's flux:
    # sat_level is the threshold at which saturation occurs
    sat_level = vega_to_total_flux_mjysr(SAT_MAG_VEGA[filtername], filtername) * peak_frac

    sat_mask = star_image > sat_level
    # In real data, saturated pixels are clipped at full-well value
    sci[sat_mask] = sat_level + sky_level   # clipped at saturation
    # Also apply a small amount of blooming: saturated pixels bleed into
    # neighbours (charge spilling upward). We approximate this simply.
    from scipy.ndimage import binary_dilation
    bleed_mask = binary_dilation(sat_mask, iterations=2)
    sci[bleed_mask & ~sat_mask] *= 0.85     # neighbours slightly depressed

    # DQ: SATURATED bit = 2 (JWST convention: dqflags.pixel['SATURATED'])
    SATURATED_BIT = 2
    dq = np.zeros((imsize, imsize), dtype=np.uint32)
    dq[sat_mask] |= SATURATED_BIT

    sat_radius = np.sqrt(sat_mask.sum() / np.pi) if sat_mask.any() else 0.0

    return sci.astype(np.float32), dq, total_flux, sat_radius


def make_synthetic_fits(mag_vega, filtername, psf_model,
                         imsize=300, noise_sigma=None, rng=None,
                         outdir=None):
    """
    Build and write a minimal JWST-like FITS file for a synthetic star.

    The file has SCI, ERR, DQ, and VAR_POISSON extensions so it can be
    read by the existing ``get_saturated_stars`` plumbing.

    Returns path to the written file and the true total flux (MJy/sr).
    """
    sci, dq, true_flux, sat_rad = make_synthetic_image(
        mag_vega, filtername, psf_model,
        imsize=imsize, noise_sigma=noise_sigma, rng=rng
    )

    if outdir is None:
        outdir = os.path.join(RESULTS_DIR, "images")
    os.makedirs(outdir, exist_ok=True)

    # Minimal WCS: point toward Brick field center
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [imsize / 2, imsize / 2]
    wcs.wcs.cdelt = [-0.0000175, 0.0000175]   # ~0.063 arcsec/pix in degrees
    wcs.wcs.crval = [266.535, -28.707]         # Brick field center (RA, Dec)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    primary_hdr = fits.Header()
    primary_hdr["INSTRUME"] = "NIRCAM"
    primary_hdr["FILTER"]   = filtername.upper()
    primary_hdr["DETECTOR"] = "NRCA1"
    primary_hdr["MODULE"]   = "A"
    primary_hdr["DATE-OBS"] = "2022-09-04"
    primary_hdr["PHOTMJSR"] = PHOTMJSR
    primary_hdr["PIXAR_SR"] = PIXAR_SR
    primary_hdr["TRUE_MAG"] = mag_vega
    primary_hdr["TRUE_FLUX"] = true_flux
    primary_hdr["SAT_RAD"]  = sat_rad
    primary_hdr["SAT_MAG"]  = SAT_MAG_VEGA[filtername.upper()]

    sci_hdr = wcs.to_header()
    sci_hdr["BUNIT"] = "MJy/sr"

    var_poisson = (np.abs(sci) / PHOTMJSR).astype(np.float32)  # rough Poisson variance

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=primary_hdr),
        fits.ImageHDU(sci, header=sci_hdr, name="SCI"),
        fits.ImageHDU(np.sqrt(np.abs(sci)).astype(np.float32), name="ERR"),
        fits.ImageHDU(dq.astype(np.uint32), name="DQ"),
        fits.ImageHDU(var_poisson, name="VAR_POISSON"),
    ])

    fname = os.path.join(
        outdir, f"synthetic_{filtername.lower()}_mag{mag_vega:.1f}.fits"
    )
    hdul.writeto(fname, overwrite=True)
    return fname, true_flux, sat_rad
