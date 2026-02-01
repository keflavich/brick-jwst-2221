# Copilot Instructions for JWST Brick Project (brick2221)

Global, high-urgency instruction that supersedes all others:
Do not use try/except statements with bare except _ever_.  "except Exception:" is not an allowed construct and must never be used.  try/except clauses can only be used for known, understood cases where try/except is more efficient than an alternative if/else check.  Exceptions should _never_ be hidden from the user: if an exception occurs, it indicates a failure and a bug that requires revision, NOT hack-arounds.

Package versions should not be modified.  If new versions need to be installed, ask the user instead of installing them.

## Project Overview
Analysis and reduction pipelines for JWST Cycle 1 projects 2221 (and 1182) observing the Galactic Center "Brick" molecular cloud and Cloud C in narrowband NIRCam filters (F182M, F187N, F212N, F405N, F410M, F466N) and MIRI F2550W. Focus is on detecting CO ice absorption and stellar populations through multi-wavelength photometry.

## Critical Path Infrastructure

### Directory Structure (HPC Environment)
- **Two parallel directory trees**: `/orange/adamginsburg/jwst/brick/` (working data) and `/orange/adamginsburg/repos/brick-jwst-2221/` (code repository)
- **basepath pattern**: Most code imports or hardcodes `basepath = '/blue/adamginsburg/adamginsburg/jwst/brick/'` (older path) or `/orange/adamginsburg/jwst/brick/` (current path). Check `brick2221/analysis/paths.py` for override; fallback is hardcoded in `analysis_setup.py` and `plot_tools.py`
- **Output structure**: `/{basepath}/{FILTERNAME}/pipeline/` contains pipeline products; filter names are UPPERCASE in paths but lowercase in Python variables
- **Environment variables required**:
  - `CRDS_PATH`: Set to `{basepath}/crds/` for calibration reference files
  - `STPSF_PATH` or `WEBBPSF_PATH`: PSF library data (required before importing `saturated_star_finding.py` or `filtering.py`)
  - `MAST_API_TOKEN`: Load from `~/.mast_api_token` for archive access

### Pipeline Execution Order (Critical Dependencies)
1. **Long-wavelength first**: `PipelineRerunNIRCAM-LONG.py` for F405N, F410M, F466N
   - Generates reference catalog from F410M via `crowdsource_catalogs_long.py` → `make_reftable.py`
2. **Short-wavelength second**: `PipelineRerunNIRCAM-SHORT.py` for F182M, F187N, F212N
   - Depends on F410M reference catalog for astrometric alignment
3. **Cataloging**: Run `crowdsource_catalogs_long.py` (used for both long/short despite name) for photometry extraction
4. **Merging**: `merge_catalogs.py` combines multi-wavelength catalogs with cross-matching

### HPC Job Submission Pattern
All pipeline scripts run via `sbatch` on SLURM cluster. See [sbatch_commands.md](sbatch_commands.md) for templates:
```bash
sbatch --job-name=webb-long-pipeline --account=adamginsburg --qos=adamginsburg-b \
  --ntasks=8 --mem=128gb --time=96:00:00 \
  --wrap "ipython /path/to/PipelineRerunNIRCAM-LONG.py"
```
- Modular runs: Use `--filternames=F405N --modules=nrca` to process single filter/detector
- Skip early steps: Add `--skip_step1and2` to bypass ramp fitting if `_cal.fits` files exist

## Code Conventions & Patterns

### Module/Detector Naming
- **Detectors**: `nrca` (short for NRCALONG/nrca5) and `nrcb` (NRCBLONG/nrcb5) for long-wavelength; short-wavelength uses specific detector IDs
- **Special "merged" module**: Stitches nrca + nrcb detectors into single mosaic. Check `module == 'merged'` throughout code

### Custom Pipeline Modifications
The JWST official pipeline (`jwst.pipeline`) is heavily customized:
1. **Destreaking**: `destreak.py` applies percentile-based horizontal stripe removal with filter-specific `medfilt_size` dict (F410M: 15px, F187N: 512px, etc.)
2. **Saturated star handling**: `saturated_star_finding.py` uses PSF fitting on saturated sources. Iterative removal via `iteratively_remove_saturated_stars()`. Check for `BrightBrA_nrca_badstars_test.fits` exclusion lists
3. **Astrometric alignment**: `align_to_catalogs.py` realigns to VVV catalog or crowdsource-based F405N reference. Alignment happens in `tweakreg` step but often needs post-pipeline rerun via `realign_and_merge.py`

### Filter/Wavelength Utilities
- `filtering.py`: Provides `get_filtername(header)` and `get_fwhm(header)` to extract filter and PSF FWHM from FITS headers
- FWHM lookup: `reduction/fwhm_table.ecsv` maps filters → PSF properties (arcsec, pixels)
- Filter name normalization: Code uses lowercase internally (`f410m`) but file paths/FITS use uppercase (`F410M`)

### Catalog Handling
- **Table metadata**: Filter names stored in `tbl.meta['FILT001']`, `FILT002`, etc. Iterate with `[tbl.meta[key] for key in tbl.meta if 'FILT' in key]`
- **Masked columns**: FITS tables can't mask boolean columns. Saturation flags combined with flux masks: `basetable[f'near_saturated_{filtername}_{filtername}'] & ~basetable[f'flux_{filtername}'].mask`
- **Coordinate columns**: Tables use `skycoord` column (SkyCoord objects) and separate `x_{filtername}`, `y_{filtername}` pixel coords per filter

### Analysis Workflow
- **Start with**: `from brick2221.analysis.analysis_setup import basepath, filternames, plot_tools, distance_modulus`
- **Selections**: `selections.py` provides `load_table()` to downselect reliable sources (saturation checks, S/N cuts, field edge masking)
- **Plotting**: `plot_tools.py` contains reusable functions (`ccd`, `ccds`, `cmds`, `plot_extvec_ccd`) for color-color/magnitude diagrams with extinction vectors
- **Ice modeling**: `co_fundamental_modeling.py` generates CO ice absorption models; `co_ice_absorbance_models.py` for integration with photometry

## Common Pitfalls & Debugging

### Import Errors
- Photutils version conflicts: Code supports both 1.6.0 (`BasicPSFPhotometry`) and ≥1.7.0 (`PSFPhotometry`) via try/except imports. Check `photutils.__version__`
- `asdf` version: Requires `asdf < 3.0` for `AsdfInFits` functionality

### WCS Validation
Pipeline includes paranoid WCS checks: `assert ww.wcs.cdelt[1] != 1, "This is not a valid WCS!"` appears in multiple files. CDELT=1 indicates pipeline WCS failure

### Timestamp Logging
Custom `print()` override in pipeline scripts prepends ISO timestamps: `f"{datetime.datetime.now().isoformat()}: {message}"`

### Proposal/Field ID Conventions
- Brick project 2221 = field 001, project 1182 = field 004
- Cloud C project 2221 = field 002
- Filenames follow: `jw{proposal_id}-o{field}_t001_nircam_clear-{filtername}-{module}_i2d.fits`

## Testing & Validation
No formal test suite. Validation happens through:
1. Catalog sanity checks in `merge_catalogs.py`: Compare flux ↔ AB mag conversions against SVO Filter Profile Service zeropoints
2. Diagnostic plots: `crowdsource_diagnostic()` in `plot_tools.py` checks flux vs background-subtracted flux consistency
3. Manual notebook checks in `notebooks/` (note: README warns these contain "dead ends" and experiments)

## External Dependencies
- **CRDS**: JWST calibration reference data system (files in `{basepath}/crds/`)
- **WebbPSF/STPSF**: PSF library generators (mutually exclusive imports)
- **Crowdsource**: Third-party photometry package (not in this repo)
- **MAST**: Data archive queries via `astroquery.mast`
- **VVV**: External astrometric reference catalog (loaded via `retrieve_vvv()` in `align_to_catalogs.py`)
