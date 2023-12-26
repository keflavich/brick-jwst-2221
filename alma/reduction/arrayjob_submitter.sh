#!/bin/sh

export FIELDNAME=${FIELDNAME:-'cloudc_2828'}
export FIELD=${FIELD:-'CloudC'} # CASA / MS field name
export MOUS=${MOUS:-'uid___A001_X1590_X282a'}
export SOUS=${SOUS:-'uid___A001_X1590_X2828'}
export GOUS=${GOUS:-'uid___A001_X1590_X2829'}

export WORK_DIR=${WORK_DIR:-"/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.${SOUS}/group.${GOUS}/member.${MOUS}/calibrated/working"}
export MSES=${MSES:-"uid___A002_Xf287d3_Xcd1e.ms uid___A002_Xfbe192_X54c.ms uid___A002_Xfbf8a1_Xfe1.ms"}
export SPW=${SPW:-25}
export NCHAN=${NCHAN:-4}
export TOTALNCHAN=${TOTALNCHAN:-1960}
export START=${START:-0}
export NARRAY=$(($TOTALNCHAN / $NCHAN))

echo "FIELDNAME=${FIELDNAME} FIELD=${FIELD} MOUS=${MOUS} SOUS=${SOUS} GOUS=${GOUS} WORK_DIR=${WORK_DIR} MSES=${MSES} SPW=${SPW} NCHAN=${NCHAN} TOTALNCHAN=${TOTALNCHAN} START=${START} NARRAY=${NARRAY}"

export DOMERGE=0

jobid=$(sbatch --qos=astronomy-dept-b \
    --account=astronomy-dept \
    --array=0-$NARRAY \
    --output=/blue/adamginsburg/adamginsburg/brick_logs/${FIELDNAME}_spw${SPW}_%A_%a.out \
    --mail-type=NONE \
    --nodes=1 \
    --ntasks=16 \
    --mem-per-cpu=4gb \
    --time=96:00:00 \
    --job-name=${FIELDNAME}_spw${SPW} \
    /orange/adamginsburg/jwst/brick/alma/reduction/slurm_arrayjob.sh)

echo ${jobid##* }

export DOMERGE=1

sbatch --qos=astronomy-dept-b \
    --account=astronomy-dept \
    --output=/blue/adamginsburg/adamginsburg/brick_logs/${FIELDNAME}_spw${SPW}_merge.out \
    --mail-type=NONE \
    --nodes=1 \
    --ntasks=16 \
    --mem-per-cpu=4gb \
    --time=96:00:00 \
    --dependency=afterok:${jobid##* } \
    --job-name=${FIELDNAME}_spw${SPW}_merge \
    /orange/adamginsburg/jwst/brick/alma/reduction/slurm_arrayjob.sh
