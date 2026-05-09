import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyspeckit
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.svo_fps import SvoFps
from astroquery.vizier import Vizier
from requests.exceptions import ConnectionError
from requests.exceptions import HTTPError
from requests.exceptions import ReadTimeout
from tqdm.auto import tqdm

from brick2221.analysis.JWST_Archive_Spectra import adjust_yaxis_for_legend_overlap
from icemodels import fluxes_in_filters

isodir = Path("/orange/adamginsburg/ice/iso/library/swsatlas/")
table_path_fits = isodir / "iso_spectra_as_fluxes.fits"
table_path_ecsv = isodir / "iso_spectra_as_fluxes.ecsv"

cmd_subdir = isodir / "cmds_sgrb2_f1minusf2_vs_f1"

SGRB2_FILTERS = [
    "F150W",
    "F182M",
    "F187N",
    "F210M",
    "F212N",
    "F300M",
    "F360M",
    "F405N",
    "F410M",
    "F466N",
    "F480M",
]


def transmission_wavelength_um(transmission_table):
    wave = np.array(transmission_table["Wavelength"], dtype=float)
    if np.nanmax(wave) > 1000:
        return wave / 1e4
    return wave


def get_jwst_filters_and_transmissions():
    jfilts = SvoFps.get_filter_list("JWST")
    jfilts.add_index("filterID")
    filter_ids = [fid for fid in jfilts["filterID"] if fid.startswith("JWST/")]
    filter_data = {fid: float(jfilts.loc[fid]["ZeroPoint"]) for fid in filter_ids}
    transdata = {fid: SvoFps.get_transmission_data(fid) for fid in filter_ids}
    return jfilts, filter_ids, filter_data, transdata


JFILTS, FILTER_IDS, FILTER_DATA, TRANSDATA = get_jwst_filters_and_transmissions()


def get_overlapping_filter_ids(wavelength_um):
    spec_min = np.nanmin(wavelength_um)
    spec_max = np.nanmax(wavelength_um)
    overlaps = []
    for fid in FILTER_IDS:
        twave = transmission_wavelength_um(TRANSDATA[fid])
        tmin = np.nanmin(twave)
        tmax = np.nanmax(twave)
        if tmax >= spec_min and tmin <= spec_max:
            overlaps.append(fid)
    return overlaps


def add_distance_columns(table):
    def first_existing_col(colnames, candidates):
        for cand in candidates:
            if cand in colnames:
                return cand
        return None

    simbad = Simbad()
    simbad.add_votable_fields("ra(d)", "dec(d)", "plx", "sp")
    vizier = Vizier(columns=["*"])

    nrows = len(table)
    ra_deg = np.full(nrows, np.nan)
    dec_deg = np.full(nrows, np.nan)
    simbad_parallax_mas = np.full(nrows, np.nan)
    simbad_distance_pc = np.full(nrows, np.nan)
    simbad_sp_type = np.full(nrows, "", dtype="U64")
    gaia_source_id = np.full(nrows, -1, dtype=np.int64)
    gaia_parallax_mas = np.full(nrows, np.nan)
    gaia_parallax_error_mas = np.full(nrows, np.nan)
    gaia_distance_pc = np.full(nrows, np.nan)
    vizier_bj_rgeo_pc = np.full(nrows, np.nan)
    vizier_bj_rpgeo_pc = np.full(nrows, np.nan)
    gaia_enabled = True
    gaia_http_failures = 0
    gaia_failure_limit = 2

    for ii, row in enumerate(tqdm(table, desc="Querying SIMBAD/Gaia/Vizier")):
        obj_name = row["Object"]
        simbad_tbl = simbad.query_object(obj_name)
        if simbad_tbl is None or len(simbad_tbl) == 0:
            continue

        ra_col = first_existing_col(simbad_tbl.colnames, ("RA_d", "ra", "RA"))
        dec_col = first_existing_col(simbad_tbl.colnames, ("DEC_d", "dec", "DEC"))
        plx_col = first_existing_col(simbad_tbl.colnames, ("PLX_VALUE", "plx_value", "PLX"))
        sp_col = first_existing_col(simbad_tbl.colnames, ("SP_TYPE", "sp_type", "SP"))
        if ra_col is None or dec_col is None:
            continue

        ra_val = float(simbad_tbl[ra_col][0])
        dec_val = float(simbad_tbl[dec_col][0])
        ra_deg[ii] = ra_val
        dec_deg[ii] = dec_val

        if plx_col is not None and np.isfinite(simbad_tbl[plx_col][0]):
            simbad_parallax_mas[ii] = float(simbad_tbl[plx_col][0])
            if simbad_parallax_mas[ii] > 0:
                simbad_distance_pc[ii] = 1000.0 / simbad_parallax_mas[ii]

        if sp_col is not None and simbad_tbl[sp_col][0] is not None:
            simbad_sp_type[ii] = str(simbad_tbl[sp_col][0]).strip()

        if not gaia_enabled:
            continue

        coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg)
        try:
            gaia_job = Gaia.cone_search_async(coord, radius=2.0 * u.arcsec)
            gaia_tbl = gaia_job.get_results()
        except HTTPError as ex:
            gaia_http_failures += 1
            print(f"Gaia query failed for {obj_name}: {ex}")
            if gaia_http_failures >= gaia_failure_limit:
                gaia_enabled = False
                print(
                    f"Disabling Gaia/Vizier lookups after {gaia_http_failures} HTTP failures; "
                    "continuing with SIMBAD-only distances."
                )
            continue
        if len(gaia_tbl) == 0:
            continue

        gcoords = SkyCoord(ra=np.array(gaia_tbl["ra"]) * u.deg, dec=np.array(gaia_tbl["dec"]) * u.deg)
        sep = coord.separation(gcoords)
        best = int(np.argmin(sep))

        source_id_col = first_existing_col(gaia_tbl.colnames, ("SOURCE_ID", "source_id"))
        if source_id_col is None:
            continue

        gaia_source_id[ii] = int(gaia_tbl[source_id_col][best])
        gaia_parallax_mas[ii] = float(gaia_tbl["parallax"][best])
        gaia_parallax_error_mas[ii] = float(gaia_tbl["parallax_error"][best])
        if gaia_parallax_mas[ii] > 0:
            gaia_distance_pc[ii] = 1000.0 / gaia_parallax_mas[ii]

        try:
            bj_tbls = vizier.query_constraints(catalog="I/352/gedr3dis", Source=str(gaia_source_id[ii]))
        except HTTPError as ex:
            print(f"Vizier query failed for {obj_name} (Gaia {gaia_source_id[ii]}): {ex}")
            continue
        if len(bj_tbls) > 0 and len(bj_tbls[0]) > 0:
            bj_row = bj_tbls[0][0]
            bj_colnames = bj_tbls[0].colnames
            if "rgeo" in bj_colnames and np.isfinite(bj_row["rgeo"]):
                vizier_bj_rgeo_pc[ii] = float(bj_row["rgeo"])
            if "rpgeo" in bj_colnames and np.isfinite(bj_row["rpgeo"]):
                vizier_bj_rpgeo_pc[ii] = float(bj_row["rpgeo"])

    table["simbad_ra_deg"] = ra_deg
    table["simbad_dec_deg"] = dec_deg
    table["simbad_parallax_mas"] = simbad_parallax_mas
    table["simbad_parallax_distance_pc"] = simbad_distance_pc
    table["simbad_sp_type"] = simbad_sp_type
    table["gaia_source_id"] = gaia_source_id
    table["gaia_parallax_mas"] = gaia_parallax_mas
    table["gaia_parallax_error_mas"] = gaia_parallax_error_mas
    table["gaia_parallax_distance_pc"] = gaia_distance_pc
    table["vizier_bj_rgeo_pc"] = vizier_bj_rgeo_pc
    table["vizier_bj_rpgeo_pc"] = vizier_bj_rpgeo_pc

    return table


def to_float_array(col):
    if np.ma.isMaskedArray(col):
        return np.asarray(col.filled(np.nan), dtype=float)
    return np.asarray(col, dtype=float)


def get_preferred_distance_pc(table):
    n = len(table)
    dist = np.full(n, np.nan)

    if "gaia_parallax_distance_pc" in table.colnames:
        gaia = to_float_array(table["gaia_parallax_distance_pc"])
        use = np.isfinite(gaia) & (gaia > 0)
        dist[use] = gaia[use]

    if "vizier_bj_rgeo_pc" in table.colnames:
        bj = to_float_array(table["vizier_bj_rgeo_pc"])
        use = ~np.isfinite(dist) & np.isfinite(bj) & (bj > 0)
        dist[use] = bj[use]

    if "simbad_parallax_distance_pc" in table.colnames:
        simbad = to_float_array(table["simbad_parallax_distance_pc"])
        use = ~np.isfinite(dist) & np.isfinite(simbad) & (simbad > 0)
        dist[use] = simbad[use]

    return dist


def fetch_spectral_types_for_objects(object_names):
    simbad = Simbad()
    simbad.add_votable_fields("sp")

    result = {}
    for name in object_names:
        tbl = simbad.query_object(name)
        if tbl is None or len(tbl) == 0:
            result[name] = ""
            continue
        col = "SP_TYPE" if "SP_TYPE" in tbl.colnames else "sp_type" if "sp_type" in tbl.colnames else None
        if col is None:
            result[name] = ""
            continue
        val = tbl[col][0]
        result[name] = "" if val is None else str(val).strip()
    return result


def print_absmag_targets(table, min_absmag=-1.0, max_absmag=1.0):
    preferred_distance_pc = get_preferred_distance_pc(table)
    valid_dist = np.isfinite(preferred_distance_pc) & (preferred_distance_pc > 0)
    if valid_dist.sum() == 0:
        print("No objects have usable parallax-based distances for absolute magnitudes.")
        return

    if "simbad_sp_type" in table.colnames:
        sp_types = np.array(table["simbad_sp_type"], dtype=str)
    else:
        sp_types = np.full(len(table), "", dtype="U64")

    rows = []
    for filt in SGRB2_FILTERS:
        mag_col = f"JWST/NIRCam.{filt}"
        if mag_col not in table.colnames:
            continue
        mag = to_float_array(table[mag_col])
        abs_mag = mag - 5.0 * np.log10(preferred_distance_pc / 10.0)
        sel = valid_dist & np.isfinite(abs_mag) & (abs_mag >= min_absmag) & (abs_mag <= max_absmag)
        for idx in np.where(sel)[0]:
            rows.append((str(table["Object"][idx]), filt, float(abs_mag[idx]), str(sp_types[idx]).strip()))

    if len(rows) == 0:
        print(f"No spectra found with absolute magnitudes in [{min_absmag}, {max_absmag}].")
        return

    missing_sp_objects = sorted({obj for obj, _, _, sp in rows if len(sp) == 0})
    if len(missing_sp_objects) > 0:
        fetched = fetch_spectral_types_for_objects(missing_sp_objects)
        rows = [
            (obj, filt, absmag, fetched[obj] if len(sp) == 0 else sp)
            for obj, filt, absmag, sp in rows
        ]

    rows = sorted(rows, key=lambda rr: (rr[1], rr[2], rr[0]))

    print("Spectra with -1 <= M <= 1 (using preferred parallax-based distance):")
    print("Object | Filter | AbsMag | SpectralType")
    for obj, filt, absmag, sp in rows:
        sp_print = sp if len(sp) > 0 else "(none reported)"
        print(f"{obj} | {filt} | {absmag: .3f} | {sp_print}")


def objects_in_absmag_window(table, min_absmag=-1.0, max_absmag=1.0):
    preferred_distance_pc = get_preferred_distance_pc(table)
    valid_dist = np.isfinite(preferred_distance_pc) & (preferred_distance_pc > 0)
    objects = set()

    for filt in SGRB2_FILTERS:
        mag_col = f"JWST/NIRCam.{filt}"
        if mag_col not in table.colnames:
            continue
        mag = to_float_array(table[mag_col])
        abs_mag = mag - 5.0 * np.log10(preferred_distance_pc / 10.0)
        sel = valid_dist & np.isfinite(abs_mag) & (abs_mag >= min_absmag) & (abs_mag <= max_absmag)
        for idx in np.where(sel)[0]:
            objects.add(str(table["Object"][idx]))

    return sorted(objects)


def _age_column_candidates(colnames):
    return [col for col in colnames if "age" in col.lower()]


def _age_value_to_gyr(value, colname, unit=None):
    if value is None or np.ma.is_masked(value):
        return np.nan

    # Handle astropy quantities/scalars and non-numeric table entries robustly.
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")

    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(v):
        return np.nan
    cl = colname.lower()

    unit_str = "" if unit is None else str(unit).lower()

    if "log" in cl and "age" in cl:
        return (10.0 ** v) / 1e9
    if "myr" in cl or "myr" in unit_str:
        return v / 1e3
    if "kyr" in cl or "kyr" in unit_str:
        return v / 1e6
    if "gyr" in cl or "gyr" in unit_str:
        return v
    if "yr" in cl or "yr" in unit_str:
        return v / 1e9

    # Ambiguous "Age" columns without explicit unit/context are skipped.
    return np.nan


def find_published_ages_for_objects(object_names):

    import pickle
    from pathlib import Path
    vizier_cache_dir = Path("/orange/adamginsburg/jwst/brick/icemodels_cache")
    vizier_cache_dir.mkdir(parents=True, exist_ok=True)
    vizier_cache_path = vizier_cache_dir / "vizier_age_cache.pkl"

    # Load cache from disk if it exists
    if vizier_cache_path.exists():
        try:
            with open(vizier_cache_path, "rb") as f:
                _vizier_age_cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load VizieR age cache: {e}")
            _vizier_age_cache = {}
    else:
        _vizier_age_cache = {}

    vizier = Vizier(columns=["**"], row_limit=20)
    rows = []

    updated = False
    for obj in tqdm(object_names, desc="Searching published ages in VizieR"):
        if obj in _vizier_age_cache:
            tables = _vizier_age_cache[obj]
        else:
            try:
                tables = vizier.query_object(obj, radius=5 * u.arcsec)
            except (HTTPError, ReadTimeout, ConnectionError) as ex:
                print(f"VizieR age lookup failed for {obj}: {ex}")
                _vizier_age_cache[obj] = []
                updated = True
                continue
            _vizier_age_cache[obj] = tables
            updated = True

        for table in tables:
            if len(table) == 0:
                continue
            age_cols = _age_column_candidates(table.colnames)
            if len(age_cols) == 0:
                continue

            first_row = table[0]
            for age_col in age_cols:
                age_unit = table[age_col].unit if age_col in table.colnames else None
                age_gyr = _age_value_to_gyr(first_row[age_col], age_col, unit=age_unit)
                if np.isfinite(age_gyr):
                    rows.append(
                        (
                            obj,
                            float(age_gyr),
                            age_col,
                            table.meta.get("name", "unknown_catalog"),
                        )
                    )

    # Save cache to disk if updated
    if updated:
        try:
            with open(vizier_cache_path, "wb") as f:
                pickle.dump(_vizier_age_cache, f)
        except Exception as e:
            print(f"Warning: Could not save VizieR age cache: {e}")

        for table in tables:
            if len(table) == 0:
                continue
            age_cols = _age_column_candidates(table.colnames)
            if len(age_cols) == 0:
                continue

            first_row = table[0]
            for age_col in age_cols:
                age_unit = table[age_col].unit if age_col in table.colnames else None
                age_gyr = _age_value_to_gyr(first_row[age_col], age_col, unit=age_unit)
                if np.isfinite(age_gyr):
                    rows.append(
                        (
                            obj,
                            float(age_gyr),
                            age_col,
                            table.meta.get("name", "unknown_catalog"),
                        )
                    )

    if len(rows) == 0:
        return Table(names=["Object", "age_gyr", "age_column", "age_catalog"], dtype=["U128", float, "U64", "U128"])

    return Table(rows=rows, names=["Object", "age_gyr", "age_column", "age_catalog"])


def print_sources_in_age_range(table, min_age_gyr=5.0, max_age_gyr=15.0, min_absmag=-1.0, max_absmag=1.0):
    candidates = objects_in_absmag_window(table, min_absmag=min_absmag, max_absmag=max_absmag)
    if len(candidates) == 0:
        print("No absolute-magnitude candidates found for age lookup.")
        return

    age_table = find_published_ages_for_objects(candidates)
    if len(age_table) == 0:
        print("No published ages found in queried VizieR catalogs for selected objects.")
        return

    inrange = (age_table["age_gyr"] >= min_age_gyr) & (age_table["age_gyr"] <= max_age_gyr)
    subset = age_table[inrange]
    if len(subset) == 0:
        print(f"No sources found with published ages in {min_age_gyr}-{max_age_gyr} Gyr.")
        return

    print(f"Sources with published ages in {min_age_gyr}-{max_age_gyr} Gyr (from VizieR):")
    print("Object | Age[Gyr] | Column | Catalog")
    for row in subset:
        print(f"{row['Object']} | {row['age_gyr']:.3f} | {row['age_column']} | {row['age_catalog']}")


def make_iso_spectra_as_fluxes_table(add_distances=True):
    iso_data = []
    for fn in tqdm(glob.glob(f"{isodir}/*sws.fit"), desc="Integrating ISO spectra"):
        with fits.open(fn) as fh:
            wave_um = np.array(fh[0].data[:, 0], dtype=float)
            flux_jy = np.array(fh[0].data[:, 1], dtype=float)

            overlaps = get_overlapping_filter_ids(wave_um)
            iso_flxd = fluxes_in_filters(
                wave_um * u.um,
                flux_jy * u.Jy,
                filterids=overlaps,
                transdata=TRANSDATA,
            )

            row = {
                "Object": fh[0].header["OBJECT"],
                "Filename": os.path.basename(fn),
                "spec_wave_min_um": float(np.nanmin(wave_um)),
                "spec_wave_max_um": float(np.nanmax(wave_um)),
            }

            for fid, flx in iso_flxd.items():
                flx_jy = float(flx.to(u.Jy).value)
                row[f"{fid}_flux_jy"] = flx_jy
                row[fid] = -2.5 * np.log10(flx_jy / FILTER_DATA[fid])

            iso_data.append(row)

    tbl = Table(iso_data)

    if add_distances:
        tbl = add_distance_columns(tbl)

    tbl.write(table_path_fits, overwrite=True)
    tbl.write(table_path_ecsv, overwrite=True)
    return tbl


def make_sgrb2_style_cmds(table, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    for f1 in SGRB2_FILTERS:
        col1 = f"JWST/NIRCam.{f1}"
        if col1 not in table.colnames:
            continue

        for f2 in SGRB2_FILTERS:
            if f1 == f2:
                continue

            col2 = f"JWST/NIRCam.{f2}"
            if col2 not in table.colnames:
                continue

            x = np.array(table[col1], dtype=float) - np.array(table[col2], dtype=float)
            y = np.array(table[col1], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                continue

            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            ax.scatter(x[valid], y[valid], s=14, alpha=0.6, linewidths=0, rasterized=True)
            ax.set_xlabel(f"{f1} - {f2}")
            ax.set_ylabel(f1)
            ax.set_title(f"CMD: {f1} vs {f1}-{f2}")
            ax.invert_yaxis()
            ax.grid(alpha=0.25)

            outname = output_dir / f"CMD_{f1}_minus_{f2}_vs_{f1}.png"
            fig.savefig(outname, dpi=200)
            plt.close(fig)


def make_reference_spectra_plots():
    for fn in tqdm(glob.glob(f"{isodir}/*sws.fit"), desc="Plotting spectra overlays"):
        with fits.open(fn) as fh:
            wave_um = np.array(fh[0].data[:, 0], dtype=float)
            flux_jy = np.array(fh[0].data[:, 1], dtype=float)
            finite_spec = np.isfinite(wave_um) & np.isfinite(flux_jy)
            overlaps = get_overlapping_filter_ids(wave_um)

            iso_flxd = fluxes_in_filters(
                wave_um * u.um,
                flux_jy * u.Jy,
                filterids=overlaps,
                transdata=TRANSDATA,
            )
            iso_mags = {
                key: -2.5 * np.log10(float(iso_flxd[key].to(u.Jy).value) / FILTER_DATA[key])
                for key in iso_flxd
            }

            mags = {}
            for filters, setname in (
                (
                    [
                        "JWST/NIRCam.F182M",
                        "JWST/NIRCam.F212N",
                        "JWST/NIRCam.F405N",
                        "JWST/NIRCam.F410M",
                        "JWST/NIRCam.F466N",
                    ],
                    "2221",
                ),
                (
                    [
                        "JWST/NIRCam.F115W",
                        "JWST/NIRCam.F200W",
                        "JWST/NIRCam.F356W",
                        "JWST/NIRCam.F444W",
                    ],
                    "1182",
                ),
            ):
                sp = pyspeckit.Spectrum(data=flux_jy * u.Jy, xarr=wave_um * u.um)
                sp.specname = fh[0].header["OBJECT"]

                # Avoid pyspeckit recursion errors by requiring finite samples in-window.
                finite_wave = wave_um[finite_spec]
                if finite_wave.size == 0:
                    print(f"Skipping {sp.specname} ({setname}): no finite spectrum samples")
                    continue

                spec_wmin = float(np.nanmin(finite_wave))
                spec_wmax = float(np.nanmax(finite_wave))

                candidate_windows = []
                if setname == "2221":
                    candidate_windows.append((3.5, 4.8))
                candidate_windows.append((2.3, 5.1))
                candidate_windows.append((spec_wmin, spec_wmax))

                plot_window = None
                for win_min, win_max in candidate_windows:
                    xmin = max(win_min, spec_wmin)
                    xmax = min(win_max, spec_wmax)
                    in_window = finite_spec & (wave_um >= xmin) & (wave_um <= xmax)
                    if xmax > xmin and in_window.any():
                        plot_window = (xmin, xmax)
                        break

                if plot_window is None:
                    print(f"Skipping {sp.specname} ({setname}): no finite data in any plotting window")
                    continue

                sp.plotter(xmin=plot_window[0], xmax=plot_window[1])

                ax = sp.plotter.axis
                ax.set_xlabel("Wavelength [$\\mu m$]")
                ax.set_title(f"{sp.specname}")

                # Twin axis for filter transmission (0-1) on the right
                ax_trans = ax.twinx()
                ax_trans.set_ylim(0, 2.1)
                ax_trans.set_ylabel("Transmission")

                for key in filters:
                    if (key not in iso_mags) and (key not in iso_flxd):
                        continue
                    mag = iso_mags[key]
                    mags[key.split(".")[-1]] = mag
                    if ax.get_ylim()[0] < 0:
                        ax.set_ylim(0, ax.get_ylim()[1])

                    twave = transmission_wavelength_um(TRANSDATA[key])
                    tcurve = np.array(TRANSDATA[key]["Transmission"], dtype=float)
                    filter_line, = ax_trans.plot(
                        twave,
                        tcurve / tcurve.max(),
                        linewidth=0.5,
                        label=f"{key[-5:]}: {mag:0.2f}" if np.isfinite(mag) else None,
                    )

                    # Overplot the computed transmission-weighted flux as a hollow
                    # square on the flux (left) axis at the filter's
                    # transmission-weighted mean wavelength.  Edge color matches
                    # the transmission line so the marker and curve are visually
                    # tied together; face is hollow so the line style remains
                    # readable.
                    tsum = tcurve.sum()
                    if tsum > 0:
                        marker_wave_um = float((twave * tcurve).sum() / tsum)
                    else:
                        marker_wave_um = float(np.nanmean(twave))
                    flux_value = iso_flxd[key].to(u.Jy).value if hasattr(iso_flxd[key], 'to') else iso_flxd[key]
                    if np.isfinite(flux_value) and flux_value != 0:
                        ax.scatter(
                            [marker_wave_um],
                            [flux_value],
                            marker='s',
                            s=36,
                            facecolor='none',
                            edgecolor=filter_line.get_color(),
                            linestyle=filter_line.get_linestyle(),
                            linewidth=1.2,
                            zorder=filter_line.get_zorder() + 1,
                        )
                        if flux_value < 0:
                            print(f"Warning: Negative flux value for filter {key} in {sp.specname}: {flux_value} Jy")
                    else:
                        raise ValueError(f"Invalid flux value for filter {key}: {flux_value}")

                if setname == "2221" and "F405N" in mags and "F466N" in mags and "F410M" in mags:
                    ax.plot(
                        [],
                        [],
                        label=f"[F405N] - [F466N] = {mags['F405N'] - mags['F466N']:0.2f}",
                        linestyle="none",
                        color="k",
                    )
                    ax.plot(
                        [],
                        [],
                        label=f"[F405N] - [F410M] = {mags['F405N'] - mags['F410M']:0.2f}",
                        linestyle="none",
                        color="k",
                    )

                if setname == "1182" and "F356W" in mags and "F444W" in mags:
                    ax.plot(
                        [],
                        [],
                        label=f"[F356W] - [F444W] = {mags['F356W'] - mags['F444W']:0.2f}",
                        linestyle="none",
                        color="k",
                    )

                ax.set_ylim(0, ax.get_ylim()[1])
                # Merge handles from the flux and transmission axes so the
                # filter labels (now drawn on the twin axis) appear in the
                # same legend as any annotations added on `ax`.
                h_main, l_main = ax.get_legend_handles_labels()
                h_trans, l_trans = ax_trans.get_legend_handles_labels()
                ax.legend(h_trans + h_main, l_trans + l_main,
                          loc="upper left", fontsize=10)
                adjust_yaxis_for_legend_overlap(ax)

                save_specname = sp.specname.replace(" ", "_").replace("/", "_")
                outpath = isodir / "pngs" / f"{os.path.splitext(os.path.basename(fn))[0]}_{save_specname}_{setname}.pdf"
                plt.savefig(outpath, dpi=150)
                plt.savefig(outpath.with_suffix('.png'), dpi=150)
                plt.close("all")


if __name__ == "__main__":
    if table_path_fits.exists():
        tbl = Table.read(table_path_fits)
    else:
        tbl = make_iso_spectra_as_fluxes_table(add_distances=True)

    print_absmag_targets(tbl, min_absmag=-1.0, max_absmag=1.0)

    # If loading an existing pre-distance table, enrich and rewrite it.
    distance_columns = {
        "gaia_parallax_distance_pc",
        "simbad_parallax_distance_pc",
        "vizier_bj_rgeo_pc",
    }
    if len(distance_columns.intersection(tbl.colnames)) < len(distance_columns):
        tbl = add_distance_columns(tbl)
        tbl.write(table_path_fits, overwrite=True)
        tbl.write(table_path_ecsv, overwrite=True)

    print_sources_in_age_range(tbl, min_age_gyr=5.0, max_age_gyr=15.0, min_absmag=-1.0, max_absmag=1.0)


    make_sgrb2_style_cmds(tbl, cmd_subdir)
    make_reference_spectra_plots()