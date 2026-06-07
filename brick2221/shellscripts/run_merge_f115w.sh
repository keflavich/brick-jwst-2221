#!/usr/bin/env bash
# OBSOLETE: F115W-only one-off merge driver using python310 paths.
# Superseded by submit_full_chain.sh + run_full_pipeline_<target>.sh
# (see README "Iter1 / Iter2 / Iter3 / Iter4 cataloging cycle").
# Set ALLOW_OBSOLETE=1 to bypass.
if [[ "${ALLOW_OBSOLETE:-0}" != "1" ]]; then
    echo "this code is obsolete; see README iter1-4 section" >&2
    exit 0
fi
# just F115W
sbatch --array=9 --job-name=webb-cat-merge-singlefields-dao --output=webb-cat-merge-singlefields-dao_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource"
sbatch --array=9 --job-name=webb-cat-merge-singlefields-crowdsource --output=webb-cat-merge-singlefields-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=crowdsource --skip-dao"
sbatch --array=9 --job-name=webb-cat-merge-singlefields-iterative --output=webb-cat-merge-singlefields-iterative%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=iterative --skip-crowdsource"
