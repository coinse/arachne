#!/bin/bash

index_file="indices/fm/fashion_mnist.init_pred.indices.csv"
for seed in {0..29}
do
	# localiser
	python3 run_mdl.py -init_mdl data/models/fmnist_simple.h5 -patch results/rq2/on_train/fm/localiser/model.loc.$seed.pkl -seed $seed -which_data fashion_mnist -num_label 10 -batch_size 10192 -rq 2 -index_file $index_file -on_both > logs/eval/rq2/train/fm/$seed.loc.out &
	wait $!

	python3 run_mdl.py -init_mdl data/models/fmnist_simple.h5 -patch results/rq2/on_train/fm/gradient_loss/model.gl.$seed.pkl -seed $seed -which_data fashion_mnist -num_label 10 -batch_size 10192 -rq 2 -index_file $index_file -on_both > logs/eval/rq2/train/fm/$seed.gl.out &
	wait $!

	python3 run_mdl.py -init_mdl data/models/fmnist_simple.h5 -patch results/rq2/on_train/fm/random/model.rd.$seed.pkl -seed $seed -which_data fashion_mnist -num_label 10 -batch_size 10192 -rq 2 -index_file $index_file -on_both > logs/eval/rq2/train/fm/$seed.rd.out &
	wait $!
done
