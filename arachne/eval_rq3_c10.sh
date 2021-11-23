#!/bin/bash

#index_file="indices/cm/cifar10.init_pred.indices.csv"
index_file="indices/cm/test/cifar10.init_pred.indices.csv"
for seed in {0..29}
do
	# localiser
	python3 run_mdl.py -datadir data/cm -init_mdl data/models/cifar_simple_90p.h5 -patch results/rq3/on_test/cm/model.misclf-rq3.${seed}-*.pkl -top_n $seed -which_data cifar10 -num_label 10 -rq 3 -index_file $index_file -on_both > logs/eval/rq3/test/cifar10/$seed.out &
	wait $!
done
