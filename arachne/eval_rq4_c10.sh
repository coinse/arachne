#!/bin/bash

#index_file="indices/cm/cifar10.init_pred.indices.csv"
index_file="indices/cm/test/cifar10.init_pred.indices.csv"

for pa in 1 2 4 6 8 
do
	for seed in {0..29}
	do
		# localiser
		python3 run_mdl.py -datadir data/cm -init_mdl data/models/cifar_simple_90p.h5 -patch results/rq4/on_test/cm/$pa/model.misclf-rq4.${seed}-*.pkl  -which_data cifar10 -num_label 10 -rq 4 -index_file $index_file -on_both > logs/eval/rq4/test/cifar10/patch_aggr.$pa.$seed.out &
		wait $!
	done
done
