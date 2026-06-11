#!/bin/bash
#SBATCH --job-name=webb-cat-F770W-mirimage-o001
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-F770W-mirimage-o001_%j.log

# Full-frame MIRI F770W photometry (sickle, prop 3958 obs 001) using the
# manual m1-m6 pipeline. Single in-process job (NOT an array): phases are
# strictly sequential. Validated on the miri_f770w_smoketest cutout 2026-06-10.

cd /orange/adamginsburg/jwst/sickle
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py \
    --filternames=F770W --modules=mirimage --each-exposure \
    --proposal_id=3958 --field=001 --target=sickle \
    --each-suffix=o001_crf \
    --daophot --skip-crowdsource \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
