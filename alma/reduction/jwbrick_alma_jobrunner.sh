export FIELDNAME='cloudc_2828'
export FIELD='CloudC' # CASA / MS field name
export MOUS=uid___A001_X1590_X282a
export SOUS=uid___A001_X1590_X2828
export GOUS=uid___A001_X1590_X2829

export ORIG_WORK_DIR='/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.uid___A001_X1590_X2828/group.uid___A001_X1590_X2829/member.uid___A001_X1590_X282a/calibrated/working'
export WORK_DIR="/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.${SOUS}/group.${GOUS}/member.${MOUS}/calibrated/working"
export MSES="uid___A002_Xf287d3_Xcd1e.ms uid___A002_Xfbe192_X54c.ms uid___A002_Xfbf8a1_Xfe1.ms"
export SPW=27
export NCHAN=4
export TOTALNCHAN=1960
export START=0


#for STARTCHAN in seq 0 32 3960; do
for STARTCHAN in `seq $START $NCHAN $TOTALNCHAN`; do

    fnbase="${MOUS}.${FIELD}_sci.spw${SPW}.$(printf %04d $STARTCHAN)+$(printf %03d $NCHAN).cube.I.manual"
    fullfn="${WORK_DIR}/${fnbase}.image"

    export STARTCHAN

    if [ ! $(bash -c 'echo $SPW') ]; then echo "SPW not exported"; exit 1; fi

    JOBNAME=${FIELDNAME}_spw${SPW}_ch${STARTCHAN}

    if [ -e $fullfn ]; then
        echo SKIPPING: ${fnbase} is done!
    else
        # use sacct to check for jobname
        job_running=$(sacct --format="JobID,JobName%45,Account%15,QOS%17,State" | grep RUNNING | grep $JOBNAME)
        if [[ $job_running ]]; then
            echo -n "SKIPPING: ${fnbase} job $jobname because it's running"
            echo $JOBNAME $fnbase
        else
            echo "RUNNING ${fnbase}: ${fullfn} does not exist.  Running."
            echo ${FIELDNAME}_spw${SPW}_ch${STARTCHAN} $fnbase
            echo $(bash -c 'echo "Key environmental variables: startchan=$STARTCHAN, nchan=$NCHAN, workdir=$WORK_DIR, mses=$MSES, spw=$SPW, MOUS=$MOUS, field=$FIELD, fieldname=$FIELDNAME"')
            echo $fnbase $fullfn
            sbatch --job-name=$JOBNAME \
                --output="/blue/adamginsburg/adamginsburg/brick_logs/${FIELDNAME}_spw${SPW}_ch${STARTCHAN}_%j.log" \
                --export=ALL \
                /orange/adamginsburg/jwst/brick/alma/reduction/slurm_runner_splitjobs.sh
        fi
    fi
done
