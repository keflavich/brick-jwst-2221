
for filter in F212N F187N F182M; do
for module in nrca nrcb merged; do
for desaturated in "" "--desaturated"; do
if [ $module = merged ]; then mem="256gb"; else mem="256gb"; fi
if [ ${filter:0:2} = 'F4' ]; then ls='long'; else ls='short'; fi
sbatch --job-name=webb-cat-${filter}-${module/nrc/} --output=webb-cat-${filter}-${module/nrc/}%j.log  --account=adamginsburg --qos=adamginsburg-b --ntasks=8 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python39/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_${ls}.py --filternames=${filter} --modules=${module} ${desaturated}"
done
done 
done
