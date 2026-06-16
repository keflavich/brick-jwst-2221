This repository contains notes & scripts for reduction and analysis of the JWST
Cycle 1 project 2221 GO that observed the Brick and Cloud C in NIRCAM
narrow-band filters and the MIRI F2550W filter.

While some aspects of this repository are well-organized, the notebooks are
not: they often include asides and experiments that lead to dead ends or just
got dropped.  There are a lot of 'notes in the margins'.  Most of them should
work if you have the directory structure set up right and the right files in
place, but I haven't put in the time needed to make clear what that structure
should be.

## Package split

As of 2026, the generic JWST photometry pipeline that was originally developed
in this repository has been extracted into a separate package,
[`jwst-gc-pipeline`](https://github.com/keflavich/jwst-gc-pipeline),
so that other Galactic Center JWST programs can share the same reduction and
crowded-field photometry code.

- **`brick2221`** (this repository) keeps the Brick / Cloud C science:
  ice analyses (`make_icecolumn_fig9*`, `icecolumn_*`, `make_ccd_with_icemodels`,
  `co_*`, etc.), foreground extinction modeling, paper-figure generators,
  selections, distance analysis, and `analysis_setup` / `paths` configuration.
- **`jwst-gc-pipeline`** holds the field-agnostic pipeline:
  `PipelineRerun*`, `destreak`, `align_to_catalogs`, `saturated_star_finding`,
  `filtering`, `make_merged_psf`, `merge_a_plus_b`, `crowdsource_catalogs_*`,
  `merge_catalogs`, `make_reftable`, generic `plot_tools`, and `isochrones`.

The old import paths inside `brick2221.reduction.*` and
`brick2221.analysis.*` continue to work as backward-compatibility shims —
they re-export the implementation from `jwst_gc_pipeline`. New code should
import directly from `jwst_gc_pipeline`.

## Iter1 / Iter2 / Iter3 / Iter4 cataloging cycle

After the JWST pipeline (`PipelineRerunNIRCAM-LONG.py` etc.) produces
per-frame `*_destreak_o<NNN>_crf.fits` images, photometry is built up
across four iterative passes.  Every pass runs the same script
(`jwst_gc_pipeline.photometry.crowdsource_catalogs_long` /
`brick2221.analysis.crowdsource_catalogs_long`) on a per-frame basis,
but with different inputs and goals.

> **New default photometry pipeline (2026-06-09):** the manual-iteration
> pipeline (`jwst_gc_pipeline.photometry.cataloging.run_manual_pipeline`)
> replaces `IterativePSFPhotometry` as **the** default PSF-photometry
> path.  It runs the m12 → m3 → m4 → m5 → m6 → m7 phases in-process as a
> single non-array job (phases are sequential, per-frame fits within a
> phase parallelize via `--parallel-workers`).  Output tokens are
> `_m1.._m7`, `_dao_basic` — disjoint from the legacy `_iter2/_iter3/
> _iter4` / `_daoiterative` products.  See
> `jwst-gc-pipeline/PHOTOMETRY_PIPELINE.md` for the algorithm.
>
> Use `brick2221/shellscripts/submit_manual_pipeline.sh` to launch the
> new path (see "Manual-iteration pipeline" section below).
>
> **Legacy scripts retired (2026-06-16):** the legacy iter1→iter2→merge→
> iter3→iter4 shell submitters (`submit_full_chain.sh`,
> `run_iter4resbgrefit.sh`, `run_merge_each*.sh`, `run_residbg_cataloging.sh`,
> the `run_all_cataloging*`/`sbatch_cataloging*` drivers, the sgrb2
> recovery/continuation scripts, `run_mosaicing_only_sickle.sh`, …) now
> live in `brick2221/shellscripts/legacy pipeline/` and `exit 1` with a
> deprecation banner if invoked.  `run_iter3_cataloging.sh` is **temporarily
> retained in place** (unmoved) only because 65 in-flight V9 `iter3-launch`
> jobs exec it by absolute path; retire it once those drain.  See
> `brick2221/shellscripts/README.md` for the full active-vs-legacy list.
>
> The legacy iter1/iter2/iter3/iter4 description below is retained for
> reference and remains accurate for the (now-retired) legacy path.

 1. **iter1 — basic per-frame DAO seed.** `crowdsource_catalogs_long.py
    --each-exposure --daophot --skip-crowdsource` (no `--iteration-label`).
    Finds peaks above noise on each per-frame destreak/crf image and runs
    one DAOPHOT PSF-fit pass.  Output is the seed catalog used downstream.
 2. **iter2 — residual-aware refit, per filter.** Same script with
    `--iteration-label=iter2 --postprocess-residuals`.  Uses the iter1
    catalog as a seed, fits the residual for missed sources, and writes a
    cleaned per-frame catalog + residual.  `merge_catalogs.py
    --merge-singlefields --indiv-merge-methods=daoiterative
    --iteration-label=iter2` then merges per-frame catalogs within each
    filter (cross-exposure merge) — the per-filter daoiterative output is
    what feeds the iter3 cross-band union seed.
 3. **iter3 — cross-band union-seeded fit.**
    `build_union_seed_catalog.py` unions the iter2 per-filter merged
    catalogs into a single `seed_union_iter3_<target>.fits`, then
    `crowdsource_catalogs_long.py --iteration-label=iter3
    --seed-catalog=<union> --postprocess-residuals` is run per frame.
    This is the canonical "all filters, all frames" simultaneous fit and
    produces the final per-frame iter3 photometry + residual mosaics.
    `merge_catalogs.py --iteration-label=iter3` produces the merged
    iter3 multiwavelength catalog.  Per-frame "forced photometry on
    satstar residuals" (`forced_photometry_residuals.py`) is gated on
    iter3 to recover sources undetected in individual filters.
 4. **iter4 — residual-background refit (optional refinement).**
    `run_iter4resbgrefit.sh` builds median-smoothed background images
    from the iter3 residuals, then refits the iter3 catalog as seeds
    with tight `xy_bounds` on the residual-bg-subtracted data.  Output
    `--iteration-label=iter4resbgrefit` is purely additive — iter1/2/3
    are not overwritten.  A parallel `iter2residbg` / `iter3residbg`
    cascade exists in `run_residbg_cataloging.sh` for early-stage
    residbg-driven refitting.

The cross-iteration dependency tree is

```
iter1 (per-filter, per-frame DAO)
  ↓ (per-filter merge)
iter2 (per-filter, per-frame, with residual postprocess)
  ↓ (per-filter merge → daoiterative)
seed_union_iter3 (cross-band union)
  ↓
iter3 (per-frame, all sources fit jointly, with residuals)
  ↓ (per-target merge)
iter4resbgrefit (residual-bg refinement, optional)
```

### Job runner coverage matrix

Active entry points in `brick2221/shellscripts/` (post-2026-06-16 cleanup).
**Cataloging is the manual-iteration pipeline (default).** Reduction
scripts are upstream (they produce the `*_crf` inputs) and remain required.

| Script                          | Stage | Pipeline | Refcat | Notes |
|---------------------------------|:-----:|:--------:|:------:|-------|
| `submit_manual_pipeline.sh`     | cataloging (m12→m7) |        |        | **THE cataloging submitter.** Single in-process job, `--manual-iterations` default; `_m1.._m7`/`_dao_basic` output |
| `sickle_miri_f770w_cataloging_*.sh` | cataloging (MIRI) |     |        | Hand-written manual-path MIRI F770W jobs (single in-process, not array) |
| `run_full_pipeline_common.sh`   | reduction + iter1 |    ✓    |   ✓   | First-pass + refcat + second-pass; its step-4 per-exposure cataloging now runs the manual path |
| `run_full_pipeline_<target>.sh` | reduction |    ✓    |   ✓   | Thin wrappers over `_common.sh` (arches, cloudef, gc2211, quintuplet, sgra, sgrb2, sgrc, w51, wd1, wd2) |
| `run_pipeline_long_sickle.sh` / `run_pipeline_miri_sickle.sh` | reduction | ✓ | | Sickle NIRCam-LONG / MIRI calibration |
| MIRI f2550w `*_image3/edgetrim/homogenize/colprofile`, `cloudc_miri_f2550w_pipelinemiri.sh` | reduction | ✓ | | MIRI f2550w mosaic post-processing |
| `sickle_build_miri_large_psf.sh` | utility |       |        | Builds large MIRI PSF grids |

**Retired to `legacy pipeline/` (deprecated, `exit 1`):** the legacy
iter1→iter2→merge→iter3→iter4 submitters — `submit_full_chain.sh`,
`run_iter4resbgrefit.sh`, `run_merge_each{,_brick,_cloudc}.sh`,
`run_residbg_cataloging.sh`, `run_all_cataloging{,_sgrb2,_sickle}.sh`,
`run_f115w_cataloging.sh`, `run_cataloging_eachexposure{,_f115w}.sh`,
`run_merge_f115w.sh`, `all_cataloging_merge{,_blur}.sh`,
`sbatch_cataloging.sh`, `sbatch_commands_nobgsub.sh`,
`run_{continuation,continuation2,recovery,recovery2}_sgrb2.sh`,
`resubmit_failed_brick_sickle_2026-04-20.sh`,
`run_mosaicing_only_sickle.sh`.
`run_iter3_cataloging.sh` stays in place until its 65 in-flight V9
`iter3-launch` jobs drain (it is exec'd by absolute path), then retire it too.

### Target / observation coverage

Each target's full per-frame set must be reduced *across all observations
that belong to it*.  Missing an observation here means a permanent hole
in the merged catalog.  Coverage as of 2026-06-07:

| Target / "name"   | Proposal | Obs (fields)         | Filters                                                            |
|-------------------|---------:|----------------------|--------------------------------------------------------------------|
| brick (2221 narrowband) | 2221 | 001                  | F182M F187N F212N F405N F410M F466N                                |
| brick-1182 (broadband)  | 1182 | 004                  | F115W F200W F356W F444W                                            |
| cloudc            |     2221 | 002                  | F182M F187N F212N F405N F410M F466N                                |
| sickle            |     3958 | 007                  | F187N F210M F335M F470N F480M                                      |
| sgrb2             |     5365 | 001                  | F150W F182M F187N F210M F212N F300M F360M F405N F410M F466N F480M  |
| sgra              |     1939 | 001                  | F115W F212N F405N                                                  |
| cloudef           |     2092 | **002 + 005**        | F162M F210M F360M F480M (Cloud E = 002, Cloud F = 005, merged)     |
| arches            |     2045 | 001                  | F212N F323N                                                        |
| quintuplet        |     2045 | 003                  | F212N F323N                                                        |
| sgrc              |     4147 | 012                  | F115W F162M F182M F212N F360M F405N F470N F480M                    |
| gc2211            |     2211 | **023, 028, 046, 049, 050** | obs 028: F150W,F277W; others: F200W,F277W                  |

Multi-obs targets (cloudef, gc2211) require the runner to loop over
every obs id; missing one silently drops part of the field.  Brick is
covered as two separate launcher targets (`brick` for proposal 2221 +
`brick-1182` for the broadband proposal 1182) because the per-frame
`destreak_o<NNN>_crf` suffix is obs-specific.

## Reduction process

 1. Pipeline files are in `reduction/`.
  * `PipelineRerunNIRCAM-LONG.py` run the JWST pipeline with modifications to the tweakwcs stage
  * `destreak.py` runs "Massimo's Destriper", which is a simple percentile-subtraction across the X-axis of each horizontal quadrant of each detector (this is run by the pipeline)
  * `align_to_catalogs.py` includes some post-facto re-alignment tools.  One function runs only on the final processed data & matches it to VVV. (this is run by the pipeline)
  * `saturated_star_finding.py` performs PSF fitting on saturated stars and removes them.  (this is run by the pipeline)
  * `crowdsource_catalogs_long.py` runs the crowsource extraction algorithm.  This must be run on the long-wavelength channels before running the short-wavelength pipeline to provide the reference catalog we use for the shortwave data.
  * `merge_catalogs.py` merges the multiwavelength catalogs.
  * `make_reftable.py` makes the reference table from the long-wavelength (F410M) data to be used on the short-wavelength data (this is run by merge_catalogs) - note that this is in analysis/
  * `PipelineRerunNIRCAM-SHORT.py` run the JWST pipeline with modifications to the tweakwcs stage including a reference catalog generated from F410M
  * `crowdsource_catalogs_long.py` on the short data (it has _long in the name, but I merged both into this one; the _short version is deprecated)  - note that this is in analysis/
  * `merge_catalogs.py` again to finally merge wavelengths - note that this is in analysis/

 2. Notebooks.  There are a lot of these.
  * `BrA_separation_nrc{a,b}.ipynb` should be run after the pipeline.  They subtract BrA from F410M, then subtract the rescaled continuum from the BrA image.
  * `F466_separation_nrc{a,b}.ipynb` does the same thing, but for the F466 filter.  It doesn't work as well because of the wavelength difference.
  * `StarKILLER_nrc{a,b}.ipynb` manually remove stars from the images, only for display purposes!
  * `Stitch_A_to_B.ipynb` pulls the two modules together.  It makes the "star-free" three-color merged image
  * `nrca_contsub_colorimages.ipynb` and `BrickImages.ipynb` make 3-color images.

  Most of the rest are test / WIP things.


## Analysis

The analysis has been done haphazardly in notebooks, but then I tried to reconcile different notebooks and merge them.

 * `analysis_setup.py` - loads the latest versions of 3-color images and catalogs.  Meant to be called before running plotting things.
 * `co_fundamental_modeling.py` - extracts from the CO2_PhoenixModel.ipynb, CO2_Phoenix_Example.ipynb, and COFundamentalModeling_and_IceModeling.ipynb notebooks.  This script produces a wider range of models than those notebooks
 * `plot_tools.py` - contains many plot templates for color-color, color-magnitude, etc.
 * `selections.py` - downselects from the full merged catalog of NIRCAM data of the brick, in all 6 filters, to the subset(s) that are reliable
 * `ColorVsCOIceAnalysis.ipynb` - makes several color-color plots



### Manual-iteration pipeline (new default, 2026-06-09)

The new default PSF-photometry pipeline (`run_manual_pipeline` in
`jwst_gc_pipeline.photometry.cataloging`) is launched via
`brick2221/shellscripts/submit_manual_pipeline.sh`.  Phases m12 → m3 →
m4 → m5 → m6 (→ m7 if multi-filter) run sequentially in one process;
per-frame fits inside each phase parallelize via
`--parallel-workers=<N>` (defaults to `--cpus-per-task`).

#### Defaults

| env var | default | meaning |
|---------|---------|---------|
| `MANUAL_CPUS` | 32 | `--cpus-per-task` AND `--parallel-workers` |
| `MANUAL_MEM`  | 256gb | `--mem` per node |
| `MANUAL_TIME` | 96:00:00 | walltime |

Override at launch, e.g.:

```bash
MANUAL_CPUS=64 MANUAL_MEM=512gb \
  bash submit_manual_pipeline.sh sgrb2 F212N nrcb V9
```

#### Usage

```
submit_manual_pipeline.sh <target> <filter[,filter,...]> <module> [tag] [extra_dep]
```

**Module token rules** (see `PHOTOMETRY_PIPELINE.md`):

- A single token `nrcb` (or `nrca`) auto-expands to `nrcb1..nrcb4` for
  SW filters and `nrcblong` for LW filters within one run.
- Use `merged` to process all detectors via the module-merging codepath.

**The `<filter>` argument is passed verbatim to `--filternames`.**  A
single filter token = single-filter run (no m7 cross-band seed).
COMMA-SEPARATED filters engage the m7 cross-band seed across all given
filters.  The full-coverage call for each target is **multi-filter**;
single-filter calls are only useful for smoke tests on one band.

Multi-filter calls must share the same `each_suffix` family — sgrb2
LW/SW mixes are rejected by the script (split into two calls; see
"Special-case targets" below).

#### Per-target copy-pastable examples

The first line for each target is the canonical full-coverage call
(multi-filter, engages m7).  The second line is a single-filter
smoke-test.

```bash
# brick (proposal 2221 narrowband, 6 filters)
bash submit_manual_pipeline.sh brick F182M,F187N,F212N,F405N,F410M,F466N merged
bash submit_manual_pipeline.sh brick F405N merged

# brick-1182 (proposal 1182 broadband, 4 filters; outputs under /jwst/brick/)
bash submit_manual_pipeline.sh brick-1182 F115W,F200W,F356W,F444W merged
bash submit_manual_pipeline.sh brick-1182 F115W merged

# cloudc (proposal 2221 obs 002, 6 filters)
bash submit_manual_pipeline.sh cloudc F182M,F187N,F212N,F405N,F410M,F466N merged
bash submit_manual_pipeline.sh cloudc F410M merged

# sickle (proposal 3958 obs 007, 5 filters)
bash submit_manual_pipeline.sh sickle F187N,F210M,F335M,F470N,F480M nrcb
bash submit_manual_pipeline.sh sickle F470N nrcb

# sgra (proposal 1939, 3 filters)
bash submit_manual_pipeline.sh sgra F115W,F212N,F405N merged
bash submit_manual_pipeline.sh sgra F212N merged

# arches (proposal 2045 obs 001, 2 filters)
bash submit_manual_pipeline.sh arches F212N,F323N merged
bash submit_manual_pipeline.sh arches F212N merged

# quintuplet (proposal 2045 obs 003, 2 filters)
bash submit_manual_pipeline.sh quintuplet F212N,F323N merged
bash submit_manual_pipeline.sh quintuplet F212N merged

# sgrc (proposal 4147 obs 012, 8 filters)
bash submit_manual_pipeline.sh sgrc F115W,F162M,F182M,F212N,F360M,F405N,F470N,F480M merged
bash submit_manual_pipeline.sh sgrc F212N merged
```

#### Special-case targets

**sgrb2** (proposal 5365) — LW filters use `align_o001_crf`, SW filters
use `destreak_o001_crf`.  Multi-filter calls that mix LW + SW are
rejected by the script; split into two calls per family:

```bash
# SW family (canonical full coverage, 5 filters)
bash submit_manual_pipeline.sh sgrb2 F150W,F182M,F187N,F210M,F212N nrcb
# LW family (canonical full coverage, 6 filters)
bash submit_manual_pipeline.sh sgrb2 F300M,F360M,F405N,F410M,F466N,F480M nrcb
# Single-filter smoke tests
bash submit_manual_pipeline.sh sgrb2 F212N nrcb   # SW
bash submit_manual_pipeline.sh sgrb2 F360M nrcb   # LW
```

**cloudef** (proposal 2092) — two obs (`002` Cloud E, `005` Cloud F)
that must be reduced together.  `submit_manual_pipeline.sh cloudef`
auto-loops the `fields` array and submits ONE sbatch per obs.  Each
obs's `run_manual_pipeline` produces per-obs outputs under
`/orange/adamginsburg/jwst/cloudef/`.  Cross-obs catalog union is
a separate manual step (call `merge_catalogs.py` over the two output
trees once both jobs complete; cross-obs auto-union in the manual
pipeline is a TODO).

```bash
bash submit_manual_pipeline.sh cloudef F210M nrcb
bash submit_manual_pipeline.sh cloudef F162M,F210M,F360M,F480M nrcb
```

**gc2211** (proposal 2211, asteroid survey, 5 GC pointings) — each obs
is its own launcher target (`gc2211-023`, `gc2211-028`, `gc2211-046`,
`gc2211-049`, `gc2211-050`).  Filter sets vary per obs.  Submit one
call per obs:

```bash
bash submit_manual_pipeline.sh gc2211-023 F200W,F277W merged
bash submit_manual_pipeline.sh gc2211-028 F150W,F277W merged   # obs 028 is F150W not F200W
bash submit_manual_pipeline.sh gc2211-046 F200W,F277W merged
bash submit_manual_pipeline.sh gc2211-049 F200W,F277W merged
bash submit_manual_pipeline.sh gc2211-050 F200W,F277W merged
```

#### Output locations

Per-frame and merged outputs land under
`<basepath>/<filter>/pipeline/` and `<basepath>/catalogs/` with tokens
`_m1.._m7`, `_dao_basic`, plus `_vetted` and `_allcols` catalog
variants (see PHOTOMETRY_PIPELINE.md §Outputs).  These are disjoint
from the legacy `_iter2/_iter3/_iter4` tokens — both paths coexist in
one tree without collision.

### Setup for Reduction

There are a few basic setup requirements before reduction of a new target:

 * The target data root must exist at `/orange/adamginsburg/jwst/{target}/`.
 * A `crds/` directory must exist under that target root.
 * A `reduction/fwhm_table.ecsv` file must exist under that target root.
 * A `catalogs/` directory should exist under that target root (it will be created automatically by helper scripts if missing).
 * If you want VVV-based alignment for that target, provide a field-of-view region file under `regions_/` and wire it into `fov_regname` in `brick2221/reduction/PipelineRerunNIRCAM-LONG.py`.

Validated mapping locations in the codebase (these must be coherent for any new target):

 * `brick2221/reduction/PipelineRerunNIRCAM-LONG.py`
   * proposal/field -> target mapping in `field_to_reg_mapping`
   * proposal/field -> reference catalog mapping in `REFERENCE_ASTROMETRIC_CATALOG_BY_FIELD`
   * optional target -> fov region mapping in `fov_regname`
 * `brick2221/analysis/crowdsource_catalogs_long.py`
   * proposal/field/target mapping in `field_to_reg_mapping`
   * visit-count policy in `nvisits`
 * `brick2221/analysis/merge_catalogs.py`
   * filter inventory per target/proposal in `obs_filters`
   * target/proposal -> field mapping in `project_obsnum`

Current first-pass/second-pass behavior:

 * If the configured reference catalog does not exist yet, `PipelineRerunNIRCAM-LONG.py` now runs a first pass and skips reference-catalog realignment steps.
 * After building a reference catalog, rerun with `--skip_step1and2` to reuse existing `_cal` products and perform the aligned second pass.

### Full Pipeline runners

The "full pipeline" for one target is the end-to-end dependency tree:

 1. First-pass pipeline jobs for all configured filters (no refcat).
 2. Reference-catalog build via `make_reference_from_pipeline_catalogs.py`.
 3. Second-pass pipeline jobs with `--skip_step1and2`.
 4. Per-exposure iter1 DAO cataloging arrays via `crowdsource_catalogs_long.py`.
 5. Cross-filter merge via `merge_catalogs.py --merge-singlefields`.

Iter2/iter3/iter4 are NOT included — run `submit_full_chain.sh` /
`run_iter3_cataloging.sh` / `run_iter4resbgrefit.sh` afterwards.

**Entry points (one per target):**

 * `run_full_pipeline_common.sh <target>` — generic per-target runner.
   Supports `cloudef sgrc sgrb2 arches quintuplet sgra gc2211`.
 * `run_full_pipeline_<target>.sh` — thin per-target wrappers
   (`arches`, `cloudef`, `quintuplet`, `sgra`, `sgrb2`, `sgrc`).
 * `run_full_pipeline_gc2211.sh` — wraps `_common.sh` and loops over the
   5 gc2211 obs IDs (one full pipeline per obs).  Required because each
   gc2211 obs is a separate field with its own filter set.
 * `run_full_pipeline_cloudef_sgrc.sh` — multi-target launcher.  Despite
   the name, runs the full pipeline for whatever target list you pass
   on the command line, OR a default list of
   `cloudef sgrc sgrb2 arches quintuplet sgra` when called with no
   args.  **Does NOT include brick or gc2211 in its default list** —
   brick is reduced separately (see below), and gc2211 needs its own
   per-obs wrapper.

**Brick is handled separately.**  Proposal 2221 (narrowband) and
proposal 1182 (broadband) have historical per-frame outputs already on
disk under `/orange/adamginsburg/jwst/brick/`.  There is no
`run_full_pipeline_brick.sh`; bring brick through `submit_full_chain.sh`
(handles iter1/iter2/merge/iter3) once per (target, filter, module),
where `target` is `brick` (2221) or `brick-1182` (1182).  See the
**iter1+ cataloging** example below.

#### Copy-pastable run examples

From `brick2221/shellscripts/`:

```bash
# --- Per-target full pipeline (steps 1-5: pipeline + refcat + iter1 merge) ---

bash run_full_pipeline_cloudef.sh        # cloudef (auto-loops obs 002 + 005)
bash run_full_pipeline_sgrc.sh           # sgrc
bash run_full_pipeline_sgrb2.sh          # sgrb2
bash run_full_pipeline_arches.sh         # arches
bash run_full_pipeline_quintuplet.sh     # quintuplet
bash run_full_pipeline_sgra.sh           # sgra
bash run_full_pipeline_gc2211.sh         # gc2211 (auto-loops obs 023 028 046 049 050)

# Single gc2211 obs only:
GC2211_OBS=028 bash run_full_pipeline_gc2211.sh

# Multi-target convenience launcher (default list: cloudef sgrc sgrb2 arches quintuplet sgra):
bash run_full_pipeline_cloudef_sgrc.sh
# Or a custom subset:
bash run_full_pipeline_cloudef_sgrc.sh cloudef sgrc

# --- iter1 + iter2 + iter2-merge + iter3-launch chain (per filter+module) ---
# submit_full_chain.sh <target> <filter> <module> [tag] [extra_dep]

bash submit_full_chain.sh brick        F405N merged V8   # proposal 2221 narrowband
bash submit_full_chain.sh brick-1182   F115W merged V8   # proposal 1182 broadband
bash submit_full_chain.sh cloudc       F410M merged V8
bash submit_full_chain.sh cloudef      F480M merged V8   # auto-loops obs 002 + 005
bash submit_full_chain.sh sickle       F470N nrcb  V8
bash submit_full_chain.sh sgrb2        F212N merged V8
bash submit_full_chain.sh sgra         F212N merged V8
bash submit_full_chain.sh arches       F212N merged V8
bash submit_full_chain.sh quintuplet   F212N merged V8
bash submit_full_chain.sh sgrc         F212N merged V8
bash submit_full_chain.sh gc2211-028   F277W merged V8   # per-obs (one of 023/028/046/049/050)

# --- iter3 launch only (assumes iter2 done) ---
bash run_iter3_cataloging.sh --target brick
bash run_iter3_cataloging.sh --target brick-1182
bash run_iter3_cataloging.sh --target cloudef           # loops obs 002 + 005
bash run_iter3_cataloging.sh --target gc2211-046

# --- iter4 refinement (assumes iter3 done) ---
bash run_iter4resbgrefit.sh --target sickle
bash run_iter4resbgrefit.sh --target brick
bash run_iter4resbgrefit.sh --target sgrb2
```

#### Defaults (full pipeline runners + `submit_full_chain.sh`)

Proposal id / field(s) / filters / reference filter used when no env-var
override is given.  Multi-obs targets are marked **bold**.

 * **brick**:       proposal `2221`, field `001`, filters `F182M,F187N,F212N,F405N,F410M,F466N`, reference filter `F405N`
 * **brick-1182**:  proposal `1182`, field `004`, filters `F115W,F200W,F356W,F444W`, reference filter `F405N` (cross-proposal — brick narrowband seed)
 * **cloudc**:      proposal `2221`, field `002`, filters `F182M,F187N,F212N,F405N,F410M,F466N`, reference filter `F405N`
 * **cloudef**:     proposal `2092`, fields **`002,005`** (Cloud E + Cloud F), filters `F162M,F210M,F360M,F480M`, reference filter `F210M` (`merge_ref_filter=f360m`)
 * **sgrc**:        proposal `4147`, field `012`, filters `F115W,F162M,F182M,F212N,F360M,F405N,F470N,F480M`, reference filter `F212N`
 * **sgrb2**:       proposal `5365`, field `001`, filters `F150W,F182M,F187N,F210M,F212N,F300M,F360M,F405N,F410M,F466N,F480M`, reference filter `F210M`
 * **sgra**:        proposal `1939`, field `001`, filters `F115W,F212N,F405N`, reference filter `F212N`
 * **arches**:      proposal `2045`, field `001`, filters `F212N,F323N`, reference filter `F212N`
 * **quintuplet**:  proposal `2045`, field `003`, filters `F212N,F323N`, reference filter `F212N`
 * **sickle**:      proposal `3958`, field `007`, filters `F187N,F210M,F335M,F470N,F480M`, reference filter `F470N`
 * **gc2211**:      proposal `2211`, obs **`023, 028, 046, 049, 050`** (each is its own field; launched per-obs by `run_full_pipeline_gc2211.sh`).
   - obs 028: filters `F150W,F277W`, reference filter `F150W`
   - obs 023/046/049/050: filters `F200W,F277W`, reference filter `F200W`
   - `merge_ref_filter=F277W` (only filter common to all five obs)

Override defaults with env vars before launching, e.g.:

 * `CLOUDEF_FIELD`, `CLOUDEF_FIELDS` (CSV for multi-obs), `CLOUDEF_FILTERS`, `CLOUDEF_REF_FILTER`
 * `SGRC_FIELD`, `SGRC_FILTERS`, `SGRC_REF_FILTER`
 * `SGRB2_FIELD`, `SGRB2_FILTERS`, `SGRB2_REF_FILTER`
 * `ARCHES_FIELD`, `ARCHES_FILTERS`, `ARCHES_REF_FILTER`
 * `QUINTUPLET_FIELD`, `QUINTUPLET_FILTERS`, `QUINTUPLET_REF_FILTER`
 * `SGRA_FIELD`, `SGRA_FILTERS`, `SGRA_REF_FILTER`
 * `GC2211_OBS` (CSV subset of `023,028,046,049,050`)
 * `MODULES` (default `merged`)
 * `ARRAY_RANGE` (default `0-23`; auto-sized from destreak file count once those exist)