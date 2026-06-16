#!/usr/bin/env bash
# ============================================================
# DEPRECATED -- legacy iter1-4 cataloging pipeline.
# Superseded by the manual-iteration pipeline (default):
#   submit_manual_pipeline.sh  -> jwst_gc_pipeline.photometry.cataloging
# Retired to 'legacy pipeline/'. Kept for reference only.
# ============================================================
echo "DEPRECATED: $(basename "$0") belongs to the legacy iter1-4 cataloging" >&2
echo "pipeline, superseded by submit_manual_pipeline.sh (manual-iteration path)." >&2
echo "This script has been retired and no longer runs. Recover from git if needed." >&2
exit 1
set -euo pipefail

# OBSOLETE: date-stamped one-off rerun script from 2026-04-20.  The job
# IDs and indices it references are stale.  Use the normal iter1->iter3
# chain (submit_full_chain.sh / run_iter3_cataloging.sh) with
# --skip-if-done to recover missing per-frame outputs.
# Set ALLOW_OBSOLETE=1 to bypass.
if [[ "${ALLOW_OBSOLETE:-0}" != "1" ]]; then
    echo "this code is obsolete (one-off from 2026-04-20); use submit_full_chain.sh with --skip-if-done" >&2
    exit 0
fi

# Resubmit only failed array indices from earlier failed runs.

sbatch --parsable --array=22 --job-name=webb-cat-sickle-F210M-nrcb1-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F210M-nrcb1-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F210M --modules=nrcb1 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --bgsub'

sbatch --parsable --array=10 --job-name=webb-cat-sickle-F210M-nrcb2-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F210M-nrcb2-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F210M --modules=nrcb2 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --bgsub'

sbatch --parsable --array=1,13 --job-name=webb-cat-sickle-F210M-nrcb4-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F210M-nrcb4-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F210M --modules=nrcb4 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource'

sbatch --parsable --array=6 --job-name=webb-cat-sickle-F187N-nrcb3-eachexp-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F187N-nrcb3-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F187N --modules=nrcb3 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --bgsub'

sbatch --parsable --array=4 --job-name=webb-cat-sickle-F187N-nrcb3-eachexp-iter2-rerun \
  --output=/blue/adamginsburg/adamginsburg/logs/sickle_jwst/webb-cat-sickle-F187N-nrcb3-eachexp_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F187N --modules=nrcb3 --each-exposure --proposal_id=3958 --target=sickle --each-suffix=destreak_o007_crf --daophot --skip-crowdsource --iteration-label=iter2 --postprocess-residuals'

sbatch --parsable --array=0,6 --job-name=webb-cat-F200W-nrca2-eachexp-brick-rerun \
  --output=/blue/adamginsburg/adamginsburg/brick_logs/webb-cat-F200W-nrca2-eachexp-brick_%j-%A_%a.log \
  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 \
  --wrap '/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /orange/adamginsburg/repos/brick-jwst-2221/brick2221/analysis/crowdsource_catalogs_long.py --filternames=F200W --modules=nrca2 --each-exposure --daophot --skip-crowdsource --proposal_id=1182 --target=brick --each-suffix=destreak_o004_crf'
