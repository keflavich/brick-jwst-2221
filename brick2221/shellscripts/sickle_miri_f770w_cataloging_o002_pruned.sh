#!/bin/bash
#SBATCH --job-name=webb-cat-F770W-miri-o002-pruned
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-F770W-miri-o002-pruned_%j.log

# MIRI-tuned full-frame F770W cataloging (sickle prop 3958 obs 001).
# MIRI tuning auto-applies (higher early thresholds, aggressive bg-sub rounds,
# relaxed qfit, no cross-band) because F770W is a MIRI filter.
# NOTE: program 3958 MIRI obs 001/002 are the sickle; obs 003 is the BRICK
# MIRI field -- catalog o003 with --target=brick --field=003 (lands in brick/,
# not sickle/).  There is intentionally no sickle o003 cataloging script.
cd /orange/adamginsburg/jwst/sickle
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py \
    --filternames=F770W --modules=mirimage --each-exposure \
    --proposal_id=3958 --field=002 --target=sickle \
    --each-suffix=o002_crf \
    --daophot --skip-crowdsource \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
