#!/bin/bash
#SBATCH --job-name=sgrb2-F770W-o002o998-PipelineMIRI
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=48:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/logs/miri_phot/sgrb2-F770W-o002o998-PipelineMIRI_%j.log

# Full re-reduction of sgrb2 (prop 5365) F770W obs 002 + obs 998 from scratch
# in the STANDARD tree (sgrb2/F770W/pipeline).  F770W/F1280W previously existed
# only under nbudaiev's sgrb2/NB/ tree (old daophot, no satstar); user opted to
# re-reduce cleanly in our pipeline.  No -s: downloads uncal + image3 asn from
# MAST, then Detector1 -> Image2 -> Image3 (adaptive east trim, relative align,
# no abs refcat -- same as F2550W obs002).  obs002 first, then obs998, sequential
# (shared filter dir -> avoid mastDownload relocate race).  Then joint-catalog
# --field=002-998.
cd /orange/adamginsburg/jwst/sgrb2
PY=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
echo "===== F770W obs002 ====="
$PY /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/PipelineMIRI.py \
    -f F770W -d 002 -p 5365
echo "===== F770W obs998 ====="
$PY /orange/adamginsburg/repos/jwst-gc-pipeline/jwst_gc_pipeline/reduction/PipelineMIRI.py \
    -f F770W -d 998 -p 5365
