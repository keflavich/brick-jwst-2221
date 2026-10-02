# shellscripts — job submitters

As of **2026-06-16** the cataloging stage is the **manual-iteration
pipeline** (`jwst_gc_pipeline.photometry.cataloging.run_manual_pipeline`),
which is the default of `catalog_long.py` (pass
`--legacy-iterations` for the old `IterativePSFPhotometry` path). The legacy
iter1→iter2→merge→iter3→iter4 shell chain has been retired to
[`legacy pipeline/`](./legacy%20pipeline/); those scripts `exit 1` with a
deprecation banner if invoked.

## Active — new cataloging pipeline

| Script | What it does |
|--------|--------------|
| `submit_manual_pipeline.sh` | **The cataloging submitter.** `submit_manual_pipeline.sh <target> <filter[,filter,...]> <module> [tag] [extra_dep]`. Launches one in-process job that runs phases m12→m3→m4→m5→m6→m7 (per-frame fits parallelize via `--parallel-workers`; NOT a SLURM array). Output tokens `_m1.._m7`, `_dao_basic`, merged internally. |
| `sickle_miri_f770w_cataloging_o001.sh`, `..._o001_mirituned.sh`, `..._o002_pruned.sh` | Hand-written manual-path MIRI F770W cataloging jobs (single in-process job, `--each-exposure`, no `--array`, no `--legacy-iterations`). |

## Active — reduction / calibration (upstream; produce the `*_crf` inputs)

These are a **different stage** from cataloging and are still required —
they generate the per-frame `*_destreak_o<NNN>_crf.fits` / mosaics that the
cataloging pipeline consumes.

| Script | What it does |
|--------|--------------|
| `run_full_pipeline_common.sh` | End-to-end reduction engine: first-pass pipeline → build refcat → second-pass pipeline → per-exposure cataloging (now the manual path). |
| `run_full_pipeline_<target>.sh` | Thin wrappers over `_common.sh`: arches, cloudef, cloudef_sgrc, gc2211, quintuplet, sgra, sgrb2, sgrc, w51, wd1, wd2. |
| `run_pipeline_long_sickle.sh` | Sickle NIRCam-LONG calibration (`PipelineRerunNIRCAM-LONG.py`). |
| `run_pipeline_miri_sickle.sh` | Sickle MIRI calibration (`PipelineMIRI.py`). |
| `brick_miri_f2550w_image3_v2.sh`, `..._edgetrim_v4.sh`, `..._tile_homogenize_v3.sh`, `..._colprofile_v5.sh` | MIRI F2550W Image3 + mosaic post-processing. |
| `cloudc_miri_f2550w_pipelinemiri.sh` | Cloud C MIRI F2550W calibration. |
| `sickle_build_miri_large_psf.sh` | Build large MIRI PSF grids (utility). |

## Retired — `legacy pipeline/`

Legacy iter1→iter2→merge→iter3→iter4 cataloging submitters. Each `exit 1`s
with a deprecation banner pointing at `submit_manual_pipeline.sh`. Recover
from git history if ever needed.

`submit_full_chain.sh`, `run_iter4resbgrefit.sh`,
`run_merge_each.sh`, `run_merge_each_brick.sh`, `run_merge_each_cloudc.sh`,
`run_residbg_cataloging.sh`, `run_all_cataloging.sh`,
`run_all_cataloging_sgrb2.sh`, `run_all_cataloging_sickle.sh`,
`run_f115w_cataloging.sh`, `run_merge_f115w.sh`,
`run_cataloging_eachexposure.sh`, `run_cataloging_eachexposure_f115w.sh`,
`all_cataloging_merge.sh`, `all_cataloging_merge_blur.sh`,
`sbatch_cataloging.sh`, `sbatch_commands_nobgsub.sh`,
`run_continuation_sgrb2.sh`, `run_continuation2_sgrb2.sh`,
`run_recovery_sgrb2.sh`, `run_recovery2_sgrb2.sh`,
`resubmit_failed_brick_sickle_2026-04-20.sh`,
`run_mosaicing_only_sickle.sh`.

### Not yet moved

`run_iter3_cataloging.sh` is the legacy iter3 launcher. It is **still in
place and runnable** because 65 in-flight V9 `iter3-launch` SLURM jobs
(w51/wd1/gc2211) exec it by absolute path — moving or guarding it would
break those pending jobs. Retire it (guard + move) once those drain.
