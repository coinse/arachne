#!/bin/bash

#index_file="indices/fm/fashion_mnist.init_pred.indices.csv"
index_file="indices/fm/test/fashion_mnist.init_pred.indices.csv"

for pa in 1 2 4 6 8 
do
	for seed in {0..29}
	do
		# localiser
		python3 run_mdl.py -datadir data/fm -init_mdl data/models/fmnist_simple.h5 -patch results/rq4/on_test/fm/$pa/model.misclf-rq4.${seed}-*.pkl  -which_data fashion_mnist -num_label 10 -batch_size 10192 -rq 4 -index_file $index_file -on_both > logs/eval/rq4/test/fm/patch_aggr.$pa.$seed.out &
		wait $!
	done
done
