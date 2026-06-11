#!/bin/bash
#SBATCH --job-name=brick-F2550W-v4-edgetrim
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/brick-F2550W-v4-edgetrim_%j.log

# Detector edge-trim + image3 rebuild for brick F2550W
# (fixes the residual tile-boundary seam at x=1609 reported 2026-06-11).

/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/scripts/miri_reduction/miri_f2550w_edgetrim_v4.py
