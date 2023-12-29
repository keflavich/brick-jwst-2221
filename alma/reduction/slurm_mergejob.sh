#!/bin/sh

if [ $DOMERGE != 1 ]; then
    echo "Should only use this for merging"
    exit 0
fi

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

echo $(bash -c 'echo "Key environmental variables: startchan=$STARTCHAN, nchan=$NCHAN, workdir=$WORK_DIR, mses=$MSES, spw=$SPW, MOUS=$MOUS, field=$FIELD, fieldname=$FIELDNAME"')

# CASA setup
CASAVERSION=casa-6.5.5-21-py3.8
export CASAPATH=/orange/adamginsburg/casa/${CASAVERSION}
export MPICASA=${CASAPATH}/bin/mpicasa
export CASA=${CASAPATH}/bin/casa

export OMPI_COMM_WORLD_SIZE=$SLURM_NTASKS
if [ -z $OMP_NUM_THREADS ]; then
    export OMP_NUM_THREADS=1
fi
export INTERACTIVE=0
export LD_LIBRARY_PATH=${CASAPATH}/lib/:$LD_LIBRARY_PATH
export OPAL_PREFIX="${CASAPATH}/lib/mpi"

export IPYTHONDIR=$SLURM_TMPDIR
export IPYTHON_DIR=$IPYTHONDIR
cp ~/.casa/config.py $SLURM_TMPDIR

mkdir -v $WORK_DIR
cd ${WORK_DIR} || exit 314
echo ${WORK_DIR}
ls

export SCRIPT_DIR=/orange/adamginsburg/jwst/brick/alma/reduction/
export PYTHONPATH=$SCRIPT_DIR

export script=${SCRIPT_DIR}/slurm_subjob_jwbrick.py

export LOG_DIR=/blue/adamginsburg/adamginsburg/brick_logs/
export LOGFILENAME="${LOG_DIR}/casa_log${mpistr}_2021.1.00363.S_${FIELDNAME}_spw${SPW}_ch${STARTCHAN}_${SLURM_JOB_ID}_$(date +%Y-%m-%d_%H_%M_%S).log"
echo logfilename=$LOGFILENAME
cwd=$(pwd)

echo "Key environmental variables: startchan=$STARTCHAN, nchan=$NCHAN, workdir=$WORK_DIR, mses=$MSES, spw=$SPW, MOUS=$MOUS, field=$FIELD, fieldname=$FIELDNAME"

echo xvfb-run -d ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1
xvfb-run -d ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1
echo "Completed CASA run (no MPI)"
