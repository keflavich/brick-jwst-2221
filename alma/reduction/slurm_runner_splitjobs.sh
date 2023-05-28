#!/bin/bash
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adamginsburg@ufl.edu     # Where to send mail
#SBATCH --ntasks=4                    # Run on a single CPU
#SBATCH --nodes=1
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept

env

pwd; hostname; date


## not used; specified by caller
###SBATCH --job-name=cloudc00363_2828_red
###SBATCH --output=/blue/adamginsburg/adamginsburg/brick_logs/red_cloudc00363_2828_%j.log

module load git

which python
which git

git --version
echo $?


CASAVERSION=casa-6.5.0-9-py3.8
CASAVERSION=casa-6.5.5-21-py3.8
export CASA=/orange/adamginsburg/casa/${CASAVERSION}/bin/casa
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

if [ -z $SLURM_NTASKS ]; then
    echo "FAILURE: SLURM_NTASKS was not specified"
    exit 1
fi

mkdir -v $WORK_DIR
cd ${WORK_DIR} || exit 314
echo ${WORK_DIR}
ls

export SCRIPT_DIR=/orange/adamginsburg/jwst/brick/alma/reduction/
export PYTHONPATH=$SCRIPT_DIR

export script=${SCRIPT_DIR}/slurm_subjob_jwbrick.py

if $dompi; then
    mpistr="_mpi"
else
    mpistr=""
fi

export LOG_DIR=/blue/adamginsburg/adamginsburg/brick_logs/
export LOGFILENAME="${LOG_DIR}/casa_log${mpistr}_2021.1.00363.S_${FIELDNAME}_spw${SPW}_ch${STARTCHAN}_${SLURM_JOB_ID}_$(date +%Y-%m-%d_%H_%M_%S).log"
echo logfilename=$LOGFILENAME
cwd=$(pwd)

# handled inside the script
#cp -rv ${ORIG_WORK_DIR}/uid___A002_Xf287d3_Xcd1e.ms ${WORK_DIR}/
#cp -rv ${ORIG_WORK_DIR}/uid___A002_Xfbe192_X54c.ms ${WORK_DIR}/
#cp -rv ${ORIG_WORK_DIR}/uid___A002_Xfbf8a1_Xfe1.ms ${WORK_DIR}/

echo "Key environmental variables: startchan=$STARTCHAN, nchan=$NCHAN, workdir=$WORK_DIR, mses=$MSES, spw=$SPW, MOUS=$MOUS, field=$FIELD, fieldname=$FIELDNAME"

if [ -z $SPW ]; then
    echo "SPW='${SPW}', i.e., it is not defined; this is not allowed"
    exit 99
fi

echo "Running CASA commands now.  dompi='${dompi}'"
if [ ! -z $dompi ]; then
    echo xvfb-run -d ${MPICASA} -n $SLURM_NTASKS ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')"
    xvfb-run -d ${MPICASA} -n $SLURM_NTASKS ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1
    echo "Completed MPICASA run"
else
    echo xvfb-run -d ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1
    xvfb-run -d ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1
    echo "Completed CASA run (no MPI)"
fi

