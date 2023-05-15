#!/bin/bash
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adamginsburg@ufl.edu     # Where to send mail
#SBATCH --ntasks=64                    # Run on a single CPU
#SBATCH --nodes=1
#SBATCH --mem=512gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --qos=adamginsburg-b
#SBATCH --account=adamginsburg
#SBATCH --job-name=brick00363_2828_red
#SBATCH --output=/blue/adamginsburg/adamginsburg/brick_logs/brick00363_2828_%j.log
pwd; hostname; date

ORIG_WORK_DIR='/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.uid___A001_X1590_X2828/group.uid___A001_X1590_X2829/member.uid___A001_X1590_X282a/calibrated/working'
WORK_DIR='/red/adamginsburg/brick_2828'

module load git

which python
which git

git --version
echo $?


CASAVERSION=casa-6.5.0-9-py3.8
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

export SCRIPT_DIR=/orange/adamginsburg/jwst/brick/alma/reduction/
export PYTHONPATH=$SCRIPT_DIR

export script=${SCRIPT_DIR}/red_first_pass_imaging_science_goal.uid___A001_X1590_X2828.py

export LOG_DIR=/blue/adamginsburg/adamginsburg/brick_logs/
export LOGFILENAME="${LOG_DIR}/casa_log_mpi_2021.1.00363.S_2828_${SLURM_JOB_ID}_$(date +%Y-%m-%d_%H_%M_%S).log"
echo logfilename=$LOGFILENAME
cwd=$(pwd)

# handled inside the script
#cp -rv ${ORIG_WORK_DIR}/uid___A002_Xf287d3_Xcd1e.ms ${WORK_DIR}/
#cp -rv ${ORIG_WORK_DIR}/uid___A002_Xfbe192_X54c.ms ${WORK_DIR}/
#cp -rv ${ORIG_WORK_DIR}/uid___A002_Xfbf8a1_Xfe1.ms ${WORK_DIR}/

echo xvfb-run -d ${MPICASA} -n $SLURM_NTASKS ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')"
xvfb-run -d ${MPICASA} -n $SLURM_NTASKS ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1
#echo xvfb-run -d ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1
#xvfb-run -d ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1

echo "Completed MPICASA run"
