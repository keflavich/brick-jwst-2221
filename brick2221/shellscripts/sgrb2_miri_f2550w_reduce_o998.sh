#!/bin/bash
#SBATCH --job-name=sgrb2-F2550W-o998-PipelineMIRI
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/sgrb2-F2550W-o998-PipelineMIRI_%j.log

# Reduce sgrb2 (prop 5365) obs 998 ("Sgr B2 MIRI_skipped_redo") F2550W -> crf.
# obs998 (tiles 06101+12101) is the MISSING HALF of the sgrb2 MIRI mosaic: obs002
# (0210b+02105) was already reduced+cataloged, but obs998 sat at cal with no crf.
# -s reuses the existing 10 cal files (Detector1/Image2 already done).  East
# trim is adaptive (per-frame glow).  After this -> joint-catalog --field=002-998.
cd /orange/adamginsburg/jwst/sgrb2
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/PipelineMIRI.py \
    -f F2550W -d 998 -p 5365 -s
