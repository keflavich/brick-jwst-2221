
export FIELDNAME='cloudc_2828'
export FIELD='CloudC' # CASA / MS field name
export MOUS=uid___A001_X1590_X282a

export ORIG_WORK_DIR='/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.uid___A001_X1590_X2828/group.uid___A001_X1590_X2829/member.uid___A001_X1590_X282a/calibrated/working'
export WORK_DIR='/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.uid___A001_X1590_X2828/group.uid___A001_X1590_X2829/member.uid___A001_X1590_X282a/calibrated/working'
export MSES="uid___A002_Xf287d3_Xcd1e.ms uid___A002_Xfbe192_X54c.ms uid___A002_Xfbf8a1_Xfe1.ms"
export SPW=25
export NCHAN=4
export TOTALNCHAN=3960

#for STARTCHAN in seq 0 32 3960; do
for STARTCHAN in `seq 0 $NCHAN 7`; do
    echo ${FIELDNAME}_spw${SPW}_ch${STARTCHAN}
    echo $(bash -c 'echo "Key environmental variables: startchan=$STARTCHAN, nchan=$NCHAN, workdir=$WORK_DIR, mses=$MSES, spw=$SPW, MOUS=$MOUS, field=$FIELD, fieldname=$FIELDNAME"')

    export STARTCHAN

    if [ ! $(bash -c 'echo $SPW') ]; then echo "SPW not exported"; exit 1; fi

    sbatch --job-name=${FIELDNAME}_spw${SPW}_ch${STARTCHAN} \
        --output="/blue/adamginsburg/adamginsburg/brick_logs/${FIELDNAME}_spw${SPW}_ch${STARTCHAN}_%j.log" \
        --export=ALL \
        /orange/adamginsburg/jwst/brick/alma/reduction/slurm_runner_splitjobs.sh
done
