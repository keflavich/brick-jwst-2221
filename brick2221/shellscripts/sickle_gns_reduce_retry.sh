#!/bin/bash
# Submit the sickle GNS re-reduction array when the group QOS frees (it is
# currently saturated -> submit fails with QOSGrpCpuLimit).  On success: chain
# the already-submitted per-filter cataloging jobs behind their filter's
# reduction task (afterok) and RELEASE them (they were held to prevent the crf
# read/write collision + premature raw-frame cataloging).  Reduction applies the
# GNS fix_alignment branch + SW=destreak/LW=align policy; crf names match what
# cataloging --each-suffix consumes.
set -u
cd /blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
PR=/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline
declare -A CAT=( [0]=35364681 [1]=35364682 [2]=35364683 [3]=35364684 [4]=35364685 )
FIN=35364686
LOG=$PR/_bench/sickle_gns_reduce_retry.log
echo "RETRY start $(date -u +%FT%TZ)" | tee -a "$LOG"
while true; do
    OUT=$(sbatch --parsable --array=0-4 \
        --account=astronomy-dept --qos=astronomy-dept-b \
        --export=ALL,PROPOSAL=3958,FIELD=007,MODULES=nrcb,FILTERS="F187N F210M F335M F470N F480M",PIPE_ROOT=$PR \
        scripts/reduction/submit_reduction.sbatch 2>&1)
    if [[ "$OUT" =~ ^[0-9]+$ ]]; then
        R=${OUT%%;*}
        echo "REDUCTION SUBMITTED R=$R at $(date -u +%FT%TZ)" | tee -a "$LOG"
        for i in 0 1 2 3 4; do
            scontrol update jobid=${CAT[$i]} dependency=afterok:${R}_${i} 2>&1 | tee -a "$LOG"
            scontrol release ${CAT[$i]} 2>&1 | tee -a "$LOG"
        done
        scontrol release $FIN 2>&1 | tee -a "$LOG"
        echo "RETRY done: reduction=$R, cataloging chained+released" | tee -a "$LOG"
        break
    fi
    echo "submit blocked ($(echo "$OUT" | tr '\n' ' ' | cut -c1-80)); retry in 10m $(date -u +%FT%TZ)" >> "$LOG"
    sleep 600
done
