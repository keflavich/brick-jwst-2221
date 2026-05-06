"""
Reproduce Bergner & Piacentino 2024 (ApJ 977, 27) Figure 3-style opacity
plot for the Polar-10-2-2 (H2O:CO2:CO=10:2:2) deposit at 30 K, with both
ice-only and ice+dust composite κ in cm² g⁻¹.

Mass-normalization convention
-----------------------------
All curves in the output PDF are reported as **cm² per gram of TOTAL grain
mass** (ice mantle + refractory dust core). Empirically this convention
reproduces Bergner+ Fig. 3 for small grains. In particular:

* ``κ_ice`` (single-component) is per gram of ice (the directly-measured
  Lambert-Beer quantity from the deposit's absorbance + thickness).
* ``κ_dust`` (single-component) is per gram of dust (OpTool's native
  output for astronomical silicate).
* ``κ_composite`` is the mass-weighted sum:

      κ_composite = f_ice · κ_ice  +  f_dust · κ_dust            (cm²/g_total)
              with    f_ice = M_ice / (M_ice + M_dust),
                     f_dust = M_dust / (M_ice + M_dust).

  This makes ``κ_composite`` per gram of **total** mass (so τ = κ_composite ·
  Σ_total). Using the canonical dense-ISM ratio M_ice/M_dust = 0.05 →
  f_ice ≈ 0.048, f_dust ≈ 0.952. Pure ice features (e.g., H2O 3 μm at
  ~10⁴ cm²/g_ice) appear in the composite at ~500 cm²/g_total, which is
  the level Bergner+ Fig. 3 shows.

Note: an alternative ``per gram of ice`` convention would write
κ_total/g_ice = κ_ice + (M_dust/M_ice)·κ_dust. Bergner+ Fig. 3 does NOT
appear to use this one (it would put the silicate plateau ~20× higher
than they show). We therefore plot the per-total-mass version.

Pipeline
--------
1. Load Bergner Polar-10-2-2 30 K from the icemodels cache (no baseline
   subtraction).
2. Compute ice κ_abs(λ) [cm²/g_ice] via icemodels' canonical
   absorbed_spectrum pipeline (matches the OpTool small-grain Rayleigh
   limit; verified by
   :func:`icemodels.tests.test_core.test_kappa_optool_benchmark_bergner_polar_10_2_2`).
3. Run OpTool on an astronomical-silicate dust grain (Draine 2003) at the
   same wavelengths to get the bare-dust κ_abs(λ) [cm²/g_dust].
4. Build the composite κ in cm²/g_total per the convention above.

Saves to:
    /orange/adamginsburg/ice/colors_of_ices_overleaf/figures/
        bergner_fig3_reproduction_polar_10_2_2.pdf
"""

import os
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
from astropy import units as u

from icemodels.core import (
    read_bergner_file, optical_constants_cache_dir,
    absorbed_spectrum, composition_to_molweight,
)


def _icemodels_kappa(tb):
    molwt = u.Quantity(composition_to_molweight(tb.meta['composition']),
                       u.Da)
    op_per_mol = absorbed_spectrum(
        xarr=tb['Wavelength'], ice_column=1,
        ice_model_table=tb, molecular_weight=molwt,
        return_tau=True).to(u.cm**2).value
    kappa = op_per_mol / molwt.to(u.g).value
    wl = np.asarray(tb['Wavelength'], dtype=float)
    so = np.argsort(wl)
    return wl[so], kappa[so]


def _run_optool_dust(material, wl_um_min, wl_um_max, nlam, optool_bin,
                     tmpdir, amin_um=0.1, amax_um=1.0, na=10):
    """Run OpTool on a built-in dust material (e.g., 'astrosil'). Returns
    (wl_um, kappa_abs_cm2_per_g)."""
    outdir = os.path.join(tmpdir, f'optool_{material}')
    cmd = [optool_bin, material,
           '-a', f"{amin_um}", f"{amax_um}", '-na', str(na),
           '-l', f"{wl_um_min:.4f}", f"{wl_um_max:.4f}", str(int(nlam)),
           '-o', outdir]
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    rows = []
    with open(os.path.join(outdir, 'dustkappa.dat')) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                rows.append([float(p) for p in parts[:4]])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def reproduce_fig3(savedir, sample='Polar-10-2-2', T=30,
                   ice_to_dust_mass=0.05):
    """Build the Bergner+ Fig 3 reproduction PDF."""
    matches = sorted(__import__('glob').glob(
        f'{optical_constants_cache_dir}/bergner_*_{sample}_{T}K.txt'))
    if not matches:
        raise FileNotFoundError(
            f"Bergner {sample} {T}K not in cache; run "
            f"download_all_bergner() first")
    tb = read_bergner_file(matches[0], baseline_subtract=False)
    if 'k' not in tb.colnames:
        raise RuntimeError("Bergner table has no derived k")

    wl_ice, kappa_ice = _icemodels_kappa(tb)
    keep = (wl_ice > 2.0) & (wl_ice < 20.0) & np.isfinite(kappa_ice)
    wl_ice, kappa_ice = wl_ice[keep], kappa_ice[keep]

    optool_bin = shutil.which('optool')
    if optool_bin is None:
        raise RuntimeError("optool CLI required on PATH for Fig 3 dust")

    with tempfile.TemporaryDirectory() as tmp:
        wl_d, kappa_d = _run_optool_dust(
            'astrosil', wl_ice.min(), wl_ice.max(), wl_ice.size,
            optool_bin, tmp,
            amin_um=0.1, amax_um=1.0, na=10)

    kappa_d_interp = np.interp(wl_ice, wl_d, kappa_d)

    # Composite ice + dust opacity (per gram of TOTAL grain mass; matches
    # Bergner+ 2024 Fig. 3 convention for small grains).
    f_ice = ice_to_dust_mass / (1.0 + ice_to_dust_mass)
    f_dust = 1.0 - f_ice
    kappa_composite = f_ice * kappa_ice + f_dust * kappa_d_interp

    fig, ax = pl.subplots(figsize=(8.5, 5.5))
    ax.semilogy(wl_ice, kappa_d_interp, color='gray', lw=1.2,
                label='astronomical silicate (Draine 2003), '
                      r'$0.1{-}1\,\mu$m DHS  (per g$_{dust}$)')
    ax.semilogy(wl_ice, kappa_ice, color='C0', lw=1.4,
                label=rf"Bergner+ {sample} {T} K (ice only, per g$_{{ice}}$)")
    ax.semilogy(wl_ice, kappa_composite, color='k', lw=1.6,
                label=rf"composite, $M_{{ice}}/M_{{dust}}={ice_to_dust_mass:.2f}$"
                rf"  (per g$_{{total}}$)")
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylabel(r'$\kappa_{abs}$ [cm$^{2}$ g$^{-1}$]'
                  '\n(ice & dust per their own mass; '
                  'composite per total grain mass)')
    ax.set_xlim(wl_ice.min(), 15)
    ax.set_ylim(1e-1, 1e5)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_title(
        rf"Reproduction of Bergner+Piacentino 2024 Fig. 3: "
        rf"{sample} ({tb.meta['composition']}) at {T} K + astrosilicate dust"
    )
    fig.tight_layout()
    out = os.path.join(savedir, f'bergner_fig3_reproduction_{sample}.pdf')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    pl.close(fig)
    print(f"saved {out}")
    return out


if __name__ == '__main__':
    savedir = '/orange/adamginsburg/ice/colors_of_ices_overleaf/figures'
    os.makedirs(savedir, exist_ok=True)
    reproduce_fig3(savedir)
