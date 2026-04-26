#!/usr/bin/env bash
# Legacy 'merged' mosaic cataloging sweep.  The full (group, blur, bgsub)
# combinatorial explosion is OFF by default (it submitted 120 identical
# 256GB jobs; we now only run on individual frames, so this sweep is rarely
# useful).  Set SWEEP_VARIANTS=1 to restore the old behaviour; otherwise the
# baseline variant (no group / no blur / no bgsub) is submitted and the
# crowdsource block is skipped unless INCLUDE_CROWDSOURCE=1.

: "${SWEEP_VARIANTS:=0}"
: "${INCLUDE_CROWDSOURCE:=0}"

if [[ "${SWEEP_VARIANTS}" == "1" ]]; then
    group_opts=(" " "--group")
    blur_opts=(" " "--blur")
    bgsub_opts=(" " "--bgsub")
else
    group_opts=(" ")
    blur_opts=(" ")
    bgsub_opts=(" ")
fi


for filter in F115W F200W F356W F444W; do
    for group in "${group_opts[@]}"; do
        for blur in "${blur_opts[@]}"; do
            for bgsub in "${bgsub_opts[@]}"; do
                sbatch --job-name=webb-cat-dao-${filter}mrg${blur:2}${bgsub:2} --output=webb-cat-dao-${filter}-mrg${blur:2}${bgsub:2}%j.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --proposal_id=1182 --modules=merged $blur $bgsub --daophot --skip-crowdsource $group"
            done
        done
    done
done

for filter in F212N F182M F187N F410M F405N F466N; do
    for group in "${group_opts[@]}"; do
        for blur in "${blur_opts[@]}"; do
            for bgsub in "${bgsub_opts[@]}"; do
                sbatch --job-name=webb-cat-dao-${filter}mrg${blur:2}${bgsub:2} --output=webb-cat-dao-${filter}-mrg${blur:2}${bgsub:2}%j.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --modules=merged $blur $bgsub --daophot --skip-crowdsource $group"
            done
        done
    done
done


if [[ "${INCLUDE_CROWDSOURCE}" == "1" ]]; then
for filter in F115W F200W F356W F444W; do
    for blur in "${blur_opts[@]}"; do
        for bgsub in "${bgsub_opts[@]}"; do
            sbatch --job-name=webb-cat-crowd-${filter}mrg${blur:2}${bgsub:2} --output=webb-cat-crowd-${filter}-mrg${blur:2}${bgsub:2}%j.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --proposal_id=1182 --modules=merged $blur $bgsub"
        done
    done
done

for filter in F212N F182M F187N F410M F405N F466N; do
    for blur in "${blur_opts[@]}"; do
        for bgsub in "${bgsub_opts[@]}"; do
            sbatch --job-name=webb-cat-crowd-${filter}mrg${blur:2}${bgsub:2} --output=webb-cat-crowd-${filter}-mrg${blur:2}${bgsub:2}%j.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/crowdsource_catalogs_long.py --filternames=${filter} --modules=merged $blur $bgsub"
        done
    done
done
fi
