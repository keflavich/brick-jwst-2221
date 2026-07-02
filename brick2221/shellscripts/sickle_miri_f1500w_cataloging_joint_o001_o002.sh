#!/bin/bash
#SBATCH --job-name=webb-cat-F1500W-miri-joint-o001o002
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-cat-F1500W-miri-joint-o001o002_%j.log

# JOINT MIRI F1500W cataloging of sickle prop 3958 obs 001 + obs 002 (2026-06-20).
# Same joint pipeline + A/B satstar fixes as F770W (worktree miri-joint-satstar).
# crf are SYMLINKS mapping the level-3 image3 crf to the per-exposure mirimage
# naming get_filenames globs (made by EXPSTART match; reduction produced only
# level-3-named crf for these filters).  seed_core_min default is F770W MJy/sr-
# calibrated; the oversub gate uses the adaptive core so bright A/B are robust,
# but the faint-star core threshold may need recal for F1500W.
WT=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-miri-joint
export PYTHONPATH="$WT:$PYTHONPATH"
source /blue/adamginsburg/adamginsburg/repos/brick-jwst-2221/brick2221/shellscripts/miri_cataloging_gains.sh  # MIRI cataloging gains (2026-07)
cd /orange/adamginsburg/jwst/sickle
rm -f F1500W/pipeline/jw03958*_mirimage_o00[12]_crf*satstar_catalog.fits \
      F1500W/pipeline/jw03958*_mirimage_o00[12]_crf*satstar_model*.fits \
      F1500W/pipeline/jw03958*_mirimage_o00[12]_crf*satstar_flags*.fits 2>/dev/null
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python \
    "$WT/jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py" \
    --filternames=F1500W --modules=mirimage --each-exposure \
    --proposal_id=3958 --field=001-002 --target=sickle \
    --each-suffix=o001_crf \
    --daophot --skip-crowdsource \
    --parallel-workers=4 \
    --group --max-group-size=10 --manual-group-min-sep-fwhm=3.0
