logdir=/blue/adamginsburg/adamginsburg/brick_logs/
python_exe=/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python
analysis_dir=/blue/adamginsburg/adamginsburg/jwst/brick/analysis
basepath=/blue/adamginsburg/adamginsburg/jwst/brick

submit_residual_mosaic_backstop() {
	local filter="$1"
	local module="$2"
	local proposal_id="$3"
	local target="$4"

	local field=""
	if [[ "${proposal_id}" == "2221" && "${target}" == "brick" ]]; then
		field="001"
	elif [[ "${proposal_id}" == "2221" && "${target}" == "cloudc" ]]; then
		field="002"
	elif [[ "${proposal_id}" == "1182" && "${target}" == "brick" ]]; then
		field="004"
	else
		echo "Skipping backstop for unknown proposal/target: proposal_id=${proposal_id} target=${target}"
		return
	fi

	sbatch --job-name=webb-mosaic-backstop-${filter}-${module}-${target} --output=${logdir}/webb-mosaic-backstop-${filter}-${module}-${target}_%j.log --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=24gb --time=24:00:00 --wrap "FILTER=${filter} MODULE=${module} PROPOSAL_ID=${proposal_id} FIELD=${field} BASEPATH=${basepath} ANALYSIS_DIR=${analysis_dir} ${python_exe} -c \"import glob, os, sys; sys.path.insert(0, os.environ['ANALYSIS_DIR']); import crowdsource_catalogs_long as c; pipeline_dir=f'{os.environ['BASEPATH']}/{os.environ['FILTER']}/pipeline';
for iteration_label in (None, 'iter2'):
	iter_suffix = '' if iteration_label is None else f'_{iteration_label}'
	iter_name = 'iter0' if iteration_label is None else iteration_label
	for kind in ('basic', 'iterative'):
		pattern=(f'{pipeline_dir}/jw0{os.environ['PROPOSAL_ID']}-o{os.environ['FIELD']}_t001_nircam_clear-{os.environ['FILTER'].lower()}-{os.environ['MODULE']}_visit*_vgroup*_exp*{iter_suffix}_daophot_{kind}_residual.fits')
		files=sorted(glob.glob(pattern))
		out=(f'{pipeline_dir}/jw0{os.environ['PROPOSAL_ID']}-o{os.environ['FIELD']}_t001_nircam_clear-{os.environ['FILTER'].lower()}-{os.environ['MODULE']}{iter_suffix}_daophot_{kind}_residual_i2d.fits')
		if files and (not os.path.exists(out)):
			print(f'Residual backstop: mosaicking {kind} {iter_name} for {os.environ[\'FILTER\']} {os.environ[\'MODULE\']} target field {os.environ[\'FIELD\']}')
			c.mosaic_each_exposure_residuals(basepath=os.environ['BASEPATH'], filtername=os.environ['FILTER'], proposal_id=os.environ['PROPOSAL_ID'], field=os.environ['FIELD'], module=os.environ['MODULE'], residual_kind=kind, desat=False, bgsub=False, epsf=False, blur=False, group=False, pupil='clear', iteration_label=iteration_label)
		elif files:
			print(f'Residual backstop: output already exists for {kind} {iter_name}: {out}')
		else:
			print(f'Residual backstop: no per-exposure residual inputs for {kind} {iter_name} matching {pattern}')\""
}

for filter in F410M F405N F466N; do
	for module in nrca nrcb; do
		submit_residual_mosaic_backstop "${filter}" "${module}" "2221" "brick"
	done
done

for filter in F212N F182M F187N; do
	for modnum in 1 2 3 4; do
		for module in nrca${modnum} nrcb${modnum}; do
			submit_residual_mosaic_backstop "${filter}" "${module}" "2221" "brick"
		done
	done
done

for filter in F410M F405N F466N; do
	for module in nrca nrcb; do
		submit_residual_mosaic_backstop "${filter}" "${module}" "2221" "cloudc"
	done
done

for filter in F212N F182M F187N; do
	for modnum in 1 2 3 4; do
		for module in nrca${modnum} nrcb${modnum}; do
			submit_residual_mosaic_backstop "${filter}" "${module}" "2221" "cloudc"
		done
	done
done

for filter in F356W F444W; do
	for module in nrca nrcb; do
		submit_residual_mosaic_backstop "${filter}" "${module}" "1182" "brick"
	done
done

for filter in F115W F200W; do
	for modnum in 1 2 3 4; do
		for module in nrca${modnum} nrcb${modnum}; do
			submit_residual_mosaic_backstop "${filter}" "${module}" "1182" "brick"
		done
	done
done

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