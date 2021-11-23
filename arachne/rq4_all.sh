#!/bin/bash

datatype=$1
datadir=$2
key='rq4'

dest="results/rq4/on_test"
logdir="logs/rq4/on_test"

for pa in 1 2 4 6 8 
do
	for seed in {0..29}
	do
		echo "==$pa =$seed==="
		./rq4_pa.sh $datatype $datadir $key $dest $seed $pa > $logdir/$datatype/patch_aggr.$pa.$seed.out &
		wait $!
	done
done
