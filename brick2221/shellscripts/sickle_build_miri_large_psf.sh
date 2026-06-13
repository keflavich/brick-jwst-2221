#!/bin/bash
#SBATCH --job-name=sickle-miri-largepsf
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=8:00:00
#SBATCH --array=0-2
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/sickle-miri-largepsf_%A_%a.log

FILTERS=(F770W F1130W F1500W)
F=${FILTERS[$SLURM_ARRAY_TASK_ID]}
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/scripts/miri_reduction/build_large_psf_grids_miri.py \
    /orange/adamginsburg/jwst/sickle/psfs $F 2024-08-23T12:03:43
