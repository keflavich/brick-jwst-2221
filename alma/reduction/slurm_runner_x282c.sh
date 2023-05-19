#!/bin/bash
#SBATCH --mail-type=NONE          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adamginsburg@ufl.edu     # Where to send mail
#SBATCH --ntasks=64                    # Run on a single CPU
#SBATCH --nodes=1
#SBATCH --mem=512gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --qos=adamginsburg-b
#SBATCH --account=adamginsburg
#SBATCH --job-name=cloudc00363_282c
#SBATCH --output=/blue/adamginsburg/adamginsburg/brick_logs/cloudc00363_282c_%j.log
pwd; hostname; date

WORK_DIR='/orange/adamginsburg/jwst/brick/alma/2021.1.00363.S/science_goal.uid___A001_X1590_X282c/group.uid___A001_X1590_X282d/member.uid___A001_X1590_X282e/calibrated/working'

module load git

which python
which git

git --version
echo $?


# LOGFILENAME=X3a43_lineimaging.log VIS=$PWD/science_goal.uid___A001_X1465_X3a41/group.uid___A001_X1465_X3a42/member.uid___A001_X1465_X3a43/calibrated//calibrated_final.ms sbatch run_line_imaging_slurm_brick.sh
# LOGFILENAME=X3a63_lineimaging.log VIS=$PWD/science_goal.uid___A001_X1465_X3a61/group.uid___A001_X1465_X3a62/member.uid___A001_X1465_X3a63/calibrated/uid___A002_Xe7b231_X1edc.ms.split.cal sbatch run_line_imaging_slurm_brick.sh
# LOGFILENAME=X3a5b_lineimaging.log VIS=$PWD/science_goal.uid___A001_X1465_X3a59/group.uid___A001_X1465_X3a5a/member.uid___A001_X1465_X3a5b/calibrated/uid___A002_Xe7b231_X89e8.ms.split.cal sbatch run_line_imaging_slurm_brick.sh


export CASA=/orange/adamginsburg/casa/casa-6.4.3-2-pipeline-2021.3.0.17/bin/casa
CASAVERSION=casa-6.4.3-2-pipeline-2021.3.0.17
CASAVERSION=casa-6.5.0-9-py3.8
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

cd ${WORK_DIR}
echo ${WORK_DIR}

export SCRIPT_DIR=/orange/adamginsburg/jwst/brick/alma/reduction/
export PYTHONPATH=$SCRIPT_DIR

export script=${SCRIPT_DIR}/first_pass_imaging_science_goal.uid___A001_X1590_X282c.py

export LOG_DIR=/blue/adamginsburg/adamginsburg/brick_logs/
export LOGFILENAME="${LOG_DIR}/casa_log_mpi_2021.1.00363.S_282c_${SLURM_JOB_ID}_$(date +%Y-%m-%d_%H_%M_%S).log"
echo logfilename=$LOGFILENAME
cwd=$(pwd)

echo xvfb-run -d ${MPICASA} -n $SLURM_NTASKS ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')"
xvfb-run -d ${MPICASA} -n $SLURM_NTASKS ${CASA} --logfile=${LOGFILENAME} --nogui --nologger --rcdir=$SLURM_TMPDIR -c "execfile('${script}')" || exit 1

echo "Completed MPICASA run"
