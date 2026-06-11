#!/bin/bash
#SBATCH --job-name=brick-F2550W-image3-v2
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/brick-F2550W-image3-v2_%j.log

/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/jwst/brick/reduction_scripts/miri_f2550w_image3_rerun_v2.py
