#!/usr/bin/env bash
set -euo pipefail

# Resubmit only failed array indices from earlier failed runs.

sbatch --parsable --array=22 --job-name=webb-cat-sickle-F210M-nrcb1-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F210M-nrcb1-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F210M --modules=nrcb1 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --bgsub'

sbatch --parsable --array=10 --job-name=webb-cat-sickle-F210M-nrcb2-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F210M-nrcb2-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F210M --modules=nrcb2 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --bgsub'

sbatch --parsable --array=1,13 --job-name=webb-cat-sickle-F210M-nrcb4-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F210M-nrcb4-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F210M --modules=nrcb4 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource'

sbatch --parsable --array=6 --job-name=webb-cat-sickle-F187N-nrcb3-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F187N-nrcb3-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F187N --modules=nrcb3 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --bgsub'

sbatch --parsable --array=4 --job-name=webb-cat-sickle-F187N-nrcb3-eachexp-iter2-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F187N-nrcb3-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F187N --modules=nrcb3 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --iteration-label=iter2 --postprocess-residuals'

sbatch --parsable --array=0,6 --job-name=webb-cat-F200W-nrca2-eachexp-brick-rerun \
  --output=/blue/adamginsburg/adamginsburg/brick_logs/webb-cat-F200W-nrca2-eachexp-brick_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F200W --modules=nrca2 --each-exposure --daophot --skip-crowdsource --proposal_id=1182 --target=brick --each-suffix=destreak_o004_crf'
