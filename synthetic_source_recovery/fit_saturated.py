"""
Lightweight PSF fitter for saturated sources — mirrors the core logic of
``brick2221.reduction.saturated_star_finding.get_saturated_stars`` but
accepts a pre-loaded PSF model and configurable parameters so we can sweep
over them in recovery tests without touching real JWST files.
"""

import os
import numpy as np
from scipy.ndimage import binary_dilation, label, center_of_mass, sum_labels, find_objects
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable

try:
    from photutils.psf import PSFPhotometry
    from photutils.background import LocalBackground
except ImportError as e:
    raise ImportError(f"photutils >= 1.9 required: {e}")


SATURATED_BIT = 2   # JWST DQ convention


def adaptive_mask_buffer(n_sat_pixels, mask_buffer_min: int = 2, cap: int = 6) -> int:
    """
    Brightness-dependent dilation radius for the saturation mask.

    Deeper saturation → larger saturated core → wider non-linear transition
    zone → need more aggressive masking.  We scale with sqrt(n_sat) so the
    buffer grows sub-linearly (the non-linear fringe is roughly one PSF FWHM
    wide regardless of how big the core is, but scales with the perimeter
    which goes as sqrt(area)).

    Typical values for NIRCam SW at ~2 µm:
      n_sat=5   → buffer=2  (marginally saturated)
      n_sat=20  → buffer=2  (Δmag~2.5 above sat limit)
      n_sat=100 → buffer=3  (Δmag~4)
      n_sat=400 → buffer=5  (very bright)
    """
    sat_radius = np.sqrt(n_sat_pixels / np.pi)
    return int(min(cap, max(mask_buffer_min, int(np.ceil(sat_radius * 0.4)))))


def adaptive_bkg_annulus(n_sat_pixels, bkg_inner_min: int = 15, bkg_inner_max: int = 50):
    """
    Brightness-dependent background annulus radii.

    For deeply saturated sources the PSF wings contaminate a close-in annulus;
    this scales the inner radius outward with sat_radius^0.75 (calibrated so
    that n_sat=37 → inner=25, matching recovery-test optimal params).  For
    marginally saturated sources the standard (15, 30) annulus is returned.

    The outer radius is always 2 × inner.

    Typical values for NIRCam SW at ~2 µm:
      n_sat=1–5   → (15, 30)
      n_sat=37    → (25, 50)
      n_sat=100   → (37, 74)
    """
    sat_radius = np.sqrt(n_sat_pixels / np.pi)
    inner = int(np.clip(np.round(10 * sat_radius ** 0.75), bkg_inner_min, bkg_inner_max))
    return inner, 2 * inner


def fit_saturated_source(
    sci,
    dq,
    psf_model,
    *,
    mask_buffer: int = 1,
    adaptive: bool = False,      # if True, mask_buffer is a minimum; actual value scales with sat area
    adaptive_bkg: bool = False,  # if True, bkg_inner/outer scale with sat area
    bkg_inner: float = 15.0,
    bkg_outer: float = 30.0,
    fit_shape: int = 81,
    fwhm_pix: float = 2.1,
    pad: int = 120,
):
    """
    Fit a single (dominant) saturated source in *sci* using PSF photometry.

    This replicates the logic in ``get_saturated_stars`` with configurable
    parameters so we can test their effect on flux recovery.

    Parameters
    ----------
    sci : 2-D float array
        Science image (arbitrary flux units; noise need not be calibrated).
    dq : 2-D uint32 array
        Data-quality array.  Pixels with bit 2 set are treated as saturated.
    psf_model : GriddedPSFModel
        Normalised PSF model.
    mask_buffer : int
        Binary-dilation iterations applied to the saturated mask before fitting.
        Larger values exclude more pixels around the saturated core.
    adaptive : bool
        If True, mask_buffer is a minimum and scales with sat area.
    adaptive_bkg : bool
        If True, bkg_inner/bkg_outer scale with sat area (wider for brighter
        sources to avoid PSF wing contamination).  bkg_inner/bkg_outer are used
        as starting values / fallback when adaptive_bkg is False.
    bkg_inner, bkg_outer : float
        Inner/outer radii (pixels) of the background annulus.  Ignored when
        adaptive_bkg=True (overridden by adaptive_bkg_annulus()).
    fit_shape : int
        PSF fit window size (pixels), passed to ``PSFPhotometry``.
    fwhm_pix : float
        Estimated FWHM in pixels (used for the aperture radius heuristic).
    pad : int
        Half-size of the cutout extracted around the saturated center.

    Returns
    -------
    dict with keys:
        flux_fit, flux_err, x_fit, y_fit, qfit, cfit, snr,
        n_sat_pixels, sat_radius_pix, mask_buffer, bkg_inner, bkg_outer,
        fit_shape, success (bool)
    """
    saturated = (dq & SATURATED_BIT) > 0
    n_sat = int(saturated.sum())

    # Resolve effective mask_buffer before building result_base
    if adaptive and n_sat > 0:
        effective_buffer = adaptive_mask_buffer(n_sat, mask_buffer_min=mask_buffer)
    else:
        effective_buffer = mask_buffer

    # Resolve background annulus — overrides bkg_inner/bkg_outer when adaptive_bkg=True
    if adaptive_bkg and n_sat > 0:
        bkg_inner, bkg_outer = adaptive_bkg_annulus(n_sat)

    result_base = dict(
        mask_buffer=mask_buffer,
        effective_mask_buffer=effective_buffer,
        adaptive=adaptive,
        bkg_inner=bkg_inner,
        bkg_outer=bkg_outer,
        fit_shape=fit_shape,
        flux_fit=np.nan,
        flux_err=np.nan,
        x_fit=np.nan,
        y_fit=np.nan,
        qfit=np.nan,
        cfit=np.nan,
        snr=np.nan,
        n_sat_pixels=n_sat,
        sat_radius_pix=float(np.sqrt(n_sat / np.pi)) if n_sat else 0.0,
        success=False,
    )

    if n_sat == 0:
        return result_base  # nothing saturated — skip

    # ── Find center of the saturated region ─────────────────────────────────
    sources_arr, nsource = label(saturated)
    if nsource == 0:
        return result_base

    sizes = sum_labels(saturated, sources_arr, np.arange(nsource) + 1)
    biggest = np.argmax(sizes) + 1
    coms = center_of_mass(saturated, labels=sources_arr, index=biggest)
    yf, xf = coms
    if not (np.isfinite(yf) and np.isfinite(xf)):
        return result_base

    ycen = int(round(yf))
    xcen = int(round(xf))

    # ── Extract cutout ───────────────────────────────────────────────────────
    ny, nx = sci.shape
    y0 = max(0, ycen - pad)
    y1 = min(ny,  ycen + pad)
    x0 = max(0, xcen - pad)
    x1 = min(nx,  xcen + pad)

    cutout = sci[y0:y1, x0:x1].copy().astype(float)
    cutout[np.isnan(cutout)] = 0.0

    # ── Build mask using effective (possibly adaptive) buffer ────────────────
    sat_cut = saturated[y0:y1, x0:x1]
    sat_expanded = binary_dilation(sat_cut, iterations=effective_buffer)
    mask = sat_expanded | (cutout == 0) | ~np.isfinite(cutout)

    # ── Position bounds (mirror original code) ───────────────────────────────
    size_saturated = max(3, int(np.sqrt(n_sat) / 2))
    x_init = float(xcen - x0)
    y_init = float(ycen - y0)

    init_params = QTable()
    init_params["x"] = [x_init]
    init_params["y"] = [y_init]

    lmfitter = LevMarLSQFitter()
    try:
        localbkg = LocalBackground(bkg_inner, bkg_outer)
    except Exception:
        localbkg = None

    psfphot = PSFPhotometry(
        localbkg_estimator=localbkg,
        fitter=lmfitter,
        psf_model=psf_model,
        fit_shape=fit_shape,
        aperture_radius=max(5.0, 15 * fwhm_pix),
    )

    # Set position bounds on the underlying model
    model = getattr(psfphot, "psf_model", None)
    if model is not None:
        for pname, bounds in (
            ("x_0", (x_init - size_saturated, x_init + size_saturated)),
            ("y_0", (y_init - size_saturated, y_init + size_saturated)),
        ):
            if hasattr(model, pname):
                try:
                    getattr(model, pname).bounds = bounds
                except Exception:
                    pass

    try:
        res = psfphot(cutout, init_params=init_params, mask=mask)
    except Exception as exc:
        result_base["error"] = str(exc)
        return result_base

    if res is None or len(res) == 0:
        return result_base

    row = res[0]
    flux     = float(row["flux_fit"])
    flux_err = float(row["flux_err"])

    if not (np.isfinite(flux_err) and flux_err > 0 and flux > 0):
        return result_base

    result_base.update(
        flux_fit=flux,
        flux_err=flux_err,
        x_fit=float(row["x_fit"]) + x0,
        y_fit=float(row["y_fit"]) + y0,
        qfit=float(row.get("qfit", np.nan)),
        cfit=float(row.get("cfit", np.nan)),
        snr=flux / flux_err,
        success=True,
    )
    return result_base
