#!/bin/bash

#index_file="indices/fm/fashion_mnist.init_pred.indices.csv"
index_file="indices/fm/test/fashion_mnist.init_pred.indices.csv"
for seed in {0..29}
do
	# localiser
	python3 run_mdl.py -datadir data/fm -init_mdl data/models/fmnist_simple.h5 -patch results/rq3/on_test/fm/model.misclf-rq3.${seed}-*.pkl -top_n $seed -which_data fashion_mnist -num_label 10 -batch_size 10192 -rq 3 -index_file $index_file -on_both > logs/eval/rq3/test/fm/$seed.out &
	wait $!
done
