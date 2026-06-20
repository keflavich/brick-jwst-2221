#!/bin/bash
#SBATCH --job-name=webb-reduce-F1130W-miri-o001
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/webb-reduce-F1130W-miri-o001_%j.log

# Re-reduce sickle (prop 3958) MIRI F1130W obs 001 alignment + mosaic.
# WHY: o001 came out shifted 20-62" southward (per-frame divergent) because the
# absolute reference catalog (f210m/GNS/VVV, all northern-strip only, Dec >=
# -28.808) does not cover o001's southern half (extends to -28.828); with the
# old abs_minobj=2 the under-covered frames latched onto spurious 2-star matches.
# FIX (PipelineMIRI.py): abs_minobj raised 2->5 so under-covered frames fail the
# absolute fit and fall back to their good raw guide-star pointing (~0.1")
# instead of a tens-of-arcsec blunder.  o002 (in the dense refcat band) unaffected.
# skip_step1and2=True: reuse existing _cal images; redo tweakreg + resample only.
cd /orange/adamginsburg/jwst/sickle
/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python -c "
from astroquery.mast import Observations
from jwst_gc_pipeline.reduction.PipelineMIRI import main
main('F1130W', Observations=Observations, regionname='sickle', field='001', proposal_id='3958', skip_step1and2=True, skip_download_for_existing=True)
"
