#dao="--daophot --skip-crowdsource"
# enables modifying globally whether you're doing just crowdsource or both (" " = crowdsource only)
# daoloop=("--daophot --skip-crowdsource")
daoloop=("--daophot --skip-crowdsource" " ")


mem=64gb
for filter in F115W; do
    for modnum in 1 2 3 4; do
        for module in nrca${modnum} nrcb${modnum}; do
            for dao in "${daoloop[@]}"; do
                sbatch --array=0-23 --job-name=webb-cat-${filter}-${module}-eachexp --output=webb-cat-${filter}-${module}-eachexp_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --modules=${module} --each-exposure ${dao}  --proposal_id=1182 --target=brick --each-suffix=destreak_o004_crf"
            done
        done
    done
done

# for filter in F410M F405N F466N; do
#     for module in nrca nrcb; do
#         for dao in "--daophot --skip-crowdsource" " "; do
#             sbatch --array=0-23 --job-name=webb-cat-${filter}-${module}-eachexp-cloudc --output=webb-cat-${filter}-${module}-eachexp-cloudc_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --modules=${module} --each-exposure ${dao} --target=cloudc --each-suffix=destreak_o002_crf"
#         done
#     done
# done
# 
# for filter in F212N F182M F187N; do
#     for modnum in 1 2 3 4; do
#         for module in nrca${modnum} nrcb${modnum}; do
#             for dao in "--daophot --skip-crowdsource" " "; do
#                 sbatch --array=0-23 --job-name=webb-cat-${filter}-${module}-eachexp-cloudc --output=webb-cat-${filter}-${module}-eachexp-cloudc_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python312/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --modules=${module} --each-exposure ${dao} --target=cloudc --each-suffix=destreak_o002_crf"
#             done
#         done
#     done
# done
