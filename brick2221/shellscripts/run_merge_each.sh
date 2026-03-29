logdir=/blue/adamginsburg/adamginsburg/brick_logs/

sbatch --array=0-4,6-7 --job-name=webb-cat-merge-singlefields-dao-brick --output=${logdir}/webb-cat-merge-singlefields-dao-brick_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource"
sbatch --array=0-4,6-7 --job-name=webb-cat-merge-singlefields-dao-cloudc --output=${logdir}/webb-cat-merge-singlefields-dao-cloudc_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource --target=cloudc"
#sbatch --array=0-4,6-7 --job-name=webb-cat-merge-singlefields-crowdsource --output=${logdir}/webb-cat-merge-singlefields-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=128gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=crowdsource --skip-dao"
#sbatch --array=0-4,6-7 --job-name=webb-cat-merge-singlefields-iterative --output=${logdir}/webb-cat-merge-singlefields-iterative%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=iterative --skip-crowdsource"
# F182M, F200W, F115W
sbatch --array=5,8-9 --job-name=webb-cat-merge-singlefields-dao-brick --output=${logdir}/webb-cat-merge-singlefields-dao-brick_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource"
sbatch --array=5,8-9 --job-name=webb-cat-merge-singlefields-dao-cloudc --output=${logdir}/webb-cat-merge-singlefields-dao-cloudc_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource --target=cloudc"
#sbatch --array=5,8-9 --job-name=webb-cat-merge-singlefields-crowdsource --output=${logdir}/webb-cat-merge-singlefields-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=crowdsource --skip-dao"
#sbatch --array=5,8-9 --job-name=webb-cat-merge-singlefields-iterative --output=${logdir}/webb-cat-merge-singlefields-iterative%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=iterative --skip-crowdsource"

# #!/bin/bash
# #SBATCH --job-name=webb-cat-merge-singlefields
# #SBATCH --output=webb-cat-merge-singlefields_%j_%A_%a.out
# #SBATCH --error=webb-cat-merge-singlefields_%j_%A_%a.err
# #SBATCH --array=0-9
# #SBATCH --account=astronomy-dept
# #SBATCH --qos=astronomy-dept-b
# #SBATCH --ntasks=1
# #SBATCH --nodes=1
# #SBATCH --time=96:00:00
# #SBATCH --mem=16gb
# 
# # filter order
# filternames=(f410m f212n f466n f405n f187n f182m f444w f356w f200w f115w)
# memory=(16gb 128gb 16gb 16gb 32gb 128gb 16gb 16gb 256gb 256gb)
# MEM=${memory[$SLURM_ARRAY_TASK_ID]}
# echo $SLURM_ARRAY_TASK_ID $MEM
# 
# #SBATCH --mem=${MEM}
# 
# srun --mem=$MEM /blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged
# 
# 
# # sbatch --array=0-9 --job-name=webb-cat-merge-singlefields-dao --output=webb-cat-merge-singlefields-dao_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource"
# # sbatch --array=0-9 --job-name=webb-cat-merge-singlefields-crowdsource --output=webb-cat-merge-singlefields-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=crowdsource --skip-dao"
# # sbatch --array=0-9 --job-name=webb-cat-merge-singlefields-iterative --output=webb-cat-merge-singlefields-iterative%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=iterative --skip-crowdsource"