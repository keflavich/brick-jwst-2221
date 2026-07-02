#!/bin/bash
#SBATCH --job-name=webb-cat-sgrb2-F2550W-joint-o002o998
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-sgrb2-F2550W-joint-o002o998_%j.log

# JOINT MIRI F2550W cataloging of sgrb2 (prop 5365) obs 002 + obs 998 together.
# The 4 mosaic tiles (002: 0210b/02105, 998: 06101/12101) tile the full Sgr B2
# field; --field=002-998 globs BOTH obs's crf so the merged data_i2d / residual /
# model span all four -> the correct combined product (the per-obs run showed
# only half).  Worktree miri-joint-satstar (satstar + fake-bright gate + data_i2d
# on reduction-mosaic grid).  Reuses crf from the standard-tree reduction.
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint
export PYTHONPATH="$WT:$PYTHONPATH"
source /blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/miri_cataloging_gains.sh  # MIRI cataloging gains (2026-07)
cd /orange/adamginsburg/jwst/sgrb2
rm -f F2550W/pipeline/jw05365*_mirimage_*o[09][09][28]_crf*satstar_catalog.fits \
      F2550W/pipeline/jw05365*_mirimage_*o[09][09][28]_crf*satstar_model*.fits \
      F2550W/pipeline/jw05365*_mirimage_*o[09][09][28]_crf*satstar_flags*.fits \
      F2550W/pipeline/jw05365*_mirimage_*o[09][09][28]_crf*satstar_residual*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py" \
    --filternames=F2550W --modules=mirimage --each-exposure \
    --proposal_id=5365 --field=002-998 --target=sgrb2 \
    --each-suffix=o002_crf \
    --daophot --skip-crowdsource \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
