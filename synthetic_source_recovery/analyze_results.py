"""
Analyze and plot the results of the saturated-star recovery parameter sweep.

Reads results/recovery_results.ecsv produced by run_recovery_tests.py and
produces:
  results/plots/bias_vs_mag_<filter>.png       — bias vs magnitude, coloured by parameter
  results/plots/bias_vs_deltamag_all.png        — summary across all filters
  results/plots/heatmap_mb_bkg_<filter>.png     — 2-D heatmap: mask_buffer × bkg_inner
  results/recovery_summary.txt                 — plain-text summary table

The central question answered here:
  Which fitter parameters minimise the brightness-dependent flux bias
  (over-recovery at mag << sat_limit, under-recovery at mag ≈ sat_limit)?
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

sys.path.insert(0, os.path.dirname(__file__))
from utils import SAT_MAG_VEGA

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOT_DIR    = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

FILTERS = ["F200W", "F182M", "F212N"]
FILTER_COLOR = {"F200W": "#1f77b4", "F182M": "#ff7f0e", "F212N": "#2ca02c"}


def load_results():
    fpath = os.path.join(RESULTS_DIR, "recovery_results.ecsv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"No results file found at {fpath}.\n"
            "Run run_recovery_tests.py first."
        )
    return Table.read(fpath, format="ascii.ecsv")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: bias vs magnitude for each parameter, one panel per filter
# ──────────────────────────────────────────────────────────────────────────────

def plot_bias_vs_mag(tbl, filtername):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    axes[0].set_ylabel("Flux ratio (recovered / true)")

    sub = tbl[tbl["filtername"] == filtername]
    default = sub[sub["is_default"]]

    def _plot_panel(ax, vary_col, vary_vals, fixed_label, title):
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.axhline(1.05, color="k", lw=0.4, ls=":")
        ax.axhline(0.95, color="k", lw=0.4, ls=":")
        ax.fill_between(
            [sub["mag_vega"].min() - 0.2, sub["mag_vega"].max() + 0.2],
            [0.95, 0.95], [1.05, 1.05], alpha=0.08, color="gray"
        )
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(vary_vals)))
        for val, col in zip(vary_vals, cmap):
            mask = np.ones(len(sub), dtype=bool)
            for k, v in fixed_label.items():
                mask &= np.asarray(sub[k]) == v
            mask &= np.asarray(sub[vary_col]) == val
            rows = sub[mask]
            if len(rows) == 0:
                continue
            order = np.argsort(rows["mag_vega"])
            rows = rows[order]
            label = f"{vary_col}={val}"
            ax.plot(rows["mag_vega"], rows["flux_ratio_med"],
                    color=col, marker="o", ms=5, label=label)
            ax.fill_between(
                rows["mag_vega"],
                rows["flux_ratio_16"], rows["flux_ratio_84"],
                alpha=0.15, color=col
            )
        # Default highlight
        if len(default) > 0:
            d_order = np.argsort(default["mag_vega"])
            d = default[d_order]
            ax.plot(d["mag_vega"], d["flux_ratio_med"],
                    color="red", lw=2, ls="--", marker="s", ms=6,
                    label="default", zorder=10)
        ax.set_xlabel("Magnitude (Vega)")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0.6, 1.5)
        # mark saturation limit
        ax.axvline(SAT_MAG_VEGA[filtername], color="gray", ls=":", lw=1.2)
        ax.text(SAT_MAG_VEGA[filtername] + 0.05, 1.42, "sat limit",
                fontsize=7, color="gray")

    # Panel 1: vary mask_buffer
    bkg_default = sub[sub["is_default"]]["bkg_inner"][0] if len(default) else 15
    fs_default  = sub[sub["is_default"]]["fit_shape"][0] if len(default) else 81
    mask_vals = sorted(set(sub["mask_buffer"]))
    _plot_panel(axes[0], "mask_buffer", mask_vals,
                {"bkg_inner": bkg_default, "bkg_outer": bkg_default + 15, "fit_shape": fs_default},
                f"{filtername}: vary mask_buffer")

    # Panel 2: vary bkg_inner
    mb_default = sub[sub["is_default"]]["mask_buffer"][0] if len(default) else 1
    bkg_inner_vals = sorted(set(sub["bkg_inner"]))
    _plot_panel(axes[1], "bkg_inner", bkg_inner_vals,
                {"mask_buffer": mb_default, "fit_shape": fs_default},
                f"{filtername}: vary bkg_inner")

    # Panel 3: vary fit_shape
    fs_vals = sorted(set(sub["fit_shape"]))
    _plot_panel(axes[2], "fit_shape", fs_vals,
                {"mask_buffer": mb_default, "bkg_inner": bkg_default},
                f"{filtername}: vary fit_shape")

    fig.tight_layout()
    outpath = os.path.join(PLOT_DIR, f"bias_vs_mag_{filtername.lower()}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: all filters, default vs best, bias vs delta-mag
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary(tbl):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhline(1.05, color="k", lw=0.5, ls=":")
    ax.axhline(0.95, color="k", lw=0.5, ls=":")
    ax.fill_between([-5, 5], [0.95]*2, [1.05]*2, alpha=0.07, color="gray")

    for filtername in FILTERS:
        sub = tbl[tbl["filtername"] == filtername]
        color = FILTER_COLOR[filtername]

        # Default
        default = sub[sub["is_default"]]
        if len(default):
            order = np.argsort(default["delta_mag"])
            d = default[order]
            ax.plot(d["delta_mag"], d["flux_ratio_med"],
                    color=color, marker="o", ms=6, ls="--",
                    label=f"{filtername} default")

        # Best (minimum |bias| per mag)
        best_rows = []
        for mag in sorted(set(sub["mag_vega"])):
            msub = sub[np.asarray(sub["mag_vega"]) == mag]
            best_idx = np.argmin(np.abs(msub["flux_ratio_bias"]))
            best_rows.append(msub[best_idx])
        if best_rows:
            from astropy.table import vstack
            best = vstack(best_rows)
            order = np.argsort(best["delta_mag"])
            b = best[order]
            ax.plot(b["delta_mag"], b["flux_ratio_med"],
                    color=color, marker="^", ms=7, ls="-",
                    label=f"{filtername} best params")

    ax.set_xlabel("Δmag = sat_limit − mag_vega  (positive = brighter / more saturated)")
    ax.set_ylabel("Flux ratio (recovered / true)")
    ax.set_title("Flux recovery: default vs best parameters (all filters)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0.6, 1.6)
    ax.set_xlim(-0.5, 5)

    fig.tight_layout()
    outpath = os.path.join(PLOT_DIR, "bias_vs_deltamag_all.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: 2-D heatmap mask_buffer × bkg_inner at fixed fit_shape
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmap(tbl, filtername, fit_shape=81):
    sub = tbl[
        (np.asarray(tbl["filtername"]) == filtername) &
        (np.asarray(tbl["fit_shape"])  == fit_shape)
    ]
    if len(sub) == 0:
        return

    # Average |bias| over all magnitudes
    mb_vals  = sorted(set(sub["mask_buffer"]))
    bkg_vals = sorted(set(sub["bkg_inner"]))
    grid = np.full((len(bkg_vals), len(mb_vals)), np.nan)

    for i, bkg in enumerate(bkg_vals):
        for j, mb in enumerate(mb_vals):
            sel = sub[
                (np.asarray(sub["mask_buffer"]) == mb) &
                (np.asarray(sub["bkg_inner"])  == bkg)
            ]
            if len(sel):
                grid[i, j] = np.nanmean(np.abs(sel["flux_ratio_bias"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   vmin=0, vmax=0.20, cmap="RdYlGn_r")
    ax.set_xticks(range(len(mb_vals)))
    ax.set_xticklabels(mb_vals)
    ax.set_yticks(range(len(bkg_vals)))
    ax.set_yticklabels(bkg_vals)
    ax.set_xlabel("mask_buffer")
    ax.set_ylabel("bkg_inner (pixels)")
    ax.set_title(f"{filtername}: mean |flux bias| (fit_shape={fit_shape})\nGreen = low bias")
    plt.colorbar(im, ax=ax, label="|bias|")

    # Mark default
    if 1 in mb_vals and 15 in bkg_vals:
        ax.add_patch(plt.Rectangle(
            (mb_vals.index(1) - 0.5, bkg_vals.index(15) - 0.5),
            1, 1, fill=False, edgecolor="blue", lw=2, label="default"
        ))
        ax.legend(fontsize=8)

    fig.tight_layout()
    outpath = os.path.join(PLOT_DIR, f"heatmap_mb_bkg_{filtername.lower()}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


# ──────────────────────────────────────────────────────────────────────────────
# Text summary
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(tbl):
    lines = [
        "SATURATED STAR FITTER — RECOVERY TEST SUMMARY",
        "=" * 60,
        "",
        "Bias = (recovered / true) - 1.  Positive = over-recovery.",
        "Δmag = sat_limit - mag_vega  (larger = more deeply saturated).",
        "",
    ]

    # Default performance
    lines.append("DEFAULT PARAMETERS (mask_buffer=1, bkg=(15,30), fit_shape=81)")
    lines.append("-" * 60)
    default = tbl[tbl["is_default"]]
    for filt in FILTERS:
        sub = default[np.asarray(default["filtername"]) == filt]
        if len(sub) == 0:
            continue
        lines.append(f"\n  {filt}:")
        order = np.argsort(sub["mag_vega"])
        for row in sub[order]:
            lines.append(
                f"    mag={row['mag_vega']:.1f}  Δmag={row['delta_mag']:+.1f}"
                f"  ratio={row['flux_ratio_med']:.3f}"
                f"  bias={row['flux_ratio_bias']*100:+.1f}%"
                f"  qfit={row['qfit_med']:.3f}"
            )

    lines += ["", "BEST PARAMETERS (minimum |bias| per filter × mag)", "-" * 60]
    for filt in FILTERS:
        sub = tbl[np.asarray(tbl["filtername"]) == filt]
        lines.append(f"\n  {filt}:")
        for mag in sorted(set(sub["mag_vega"])):
            msub = sub[np.asarray(sub["mag_vega"]) == mag]
            best_idx = np.argmin(np.abs(msub["flux_ratio_bias"]))
            b = msub[best_idx]
            lines.append(
                f"    mag={b['mag_vega']:.1f}"
                f"  best: mb={b['mask_buffer']} bkg=({b['bkg_inner']},{b['bkg_outer']})"
                f" fs={b['fit_shape']}"
                f"  ratio={b['flux_ratio_med']:.3f}"
                f"  bias={b['flux_ratio_bias']*100:+.1f}%"
            )

    # Overall recommendation
    lines += ["", "RECOMMENDATIONS", "-" * 60]
    # Find the single (mb, bkg, fs) combo with best mean |bias| across all filters
    combos = {}
    for row in tbl:
        key = (row["mask_buffer"], row["bkg_inner"], row["bkg_outer"], row["fit_shape"])
        combos.setdefault(key, []).append(abs(row["flux_ratio_bias"]))
    best_combo = min(combos, key=lambda k: np.mean(combos[k]))
    lines.append(
        f"\n  Single best parameter set (minimises mean |bias| across all filters/mags):\n"
        f"    mask_buffer={best_combo[0]}\n"
        f"    LocalBackground({best_combo[1]}, {best_combo[2]})\n"
        f"    fit_shape={best_combo[3]}\n"
        f"  Mean |bias|: {np.mean(combos[best_combo])*100:.1f}%"
        f"  (default: {np.mean(combos.get((1,15,30,81),[0.1]))*100:.1f}%)"
    )

    text = "\n".join(lines)
    outpath = os.path.join(RESULTS_DIR, "recovery_summary.txt")
    with open(outpath, "w") as fh:
        fh.write(text)
    print(f"\nSaved {outpath}")
    print(text)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    tbl = load_results()
    print(f"Loaded {len(tbl)} rows from recovery_results.ecsv")

    for filt in FILTERS:
        n = (np.asarray(tbl["filtername"]) == filt).sum()
        if n > 0:
            plot_bias_vs_mag(tbl, filt)
            plot_heatmap(tbl, filt)

    plot_summary(tbl)
    write_summary(tbl)


if __name__ == "__main__":
    main()
