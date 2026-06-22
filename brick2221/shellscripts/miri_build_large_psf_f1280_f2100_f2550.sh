#!/bin/bash
#SBATCH --job-name=miri-largepsf-extra
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=8:00:00
#SBATCH --array=0-2
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/miri-largepsf-extra_%A_%a.log

# Large (fovp1024_samp2) MIRI PSF grids for the satstar fitter, for the filters
# the other MIRI fields need but sickle never built: F1280W, F2100W, F2550W.
# (F770W/F1130W/F1500W already exist in sickle/psfs.)  These are detector+filter
# dependent, NOT field dependent, so one canonical set lives in sickle/psfs and
# is symlinked into brick/cloudc/sgrb2/w51/cloudc-2526 psf dirs.
# Date = a representative MIRI WSS OPD epoch (same convention as the sickle build).
FILTERS=(F1280W F2100W F2550W)
F=${FILTERS[$SLURM_ARRAY_TASK_ID]}
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/scripts/miri_reduction/build_large_psf_grids_miri.py \
    /orange/adamginsburg/jwst/sickle/psfs $F 2024-08-23T12:03:43
