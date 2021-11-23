#!/bin/bash

index_file="indices/cm/cifar10.init_pred.indices.csv"
for seed in {0..29}
do
	# localiser
	python3 run_mdl.py -init_mdl data/models/cifar_simple_90p.h5 -patch results/rq2/on_train/cm/localiser/model.loc.$seed.pkl -seed $seed -which_data cifar10 -num_label 10 -rq 2 -index_file $index_file -datadir data/gtsrb/prepared/ -on_both > logs/eval/rq2/train/cifar10/$seed.loc.out &
	wait $!

	python3 run_mdl.py -init_mdl data/models/cifar_simple_90p.h5 -patch results/rq2/on_train/cm/gradient_loss/model.gl.$seed.pkl -seed $seed -which_data cifar10 -num_label 10 -rq 2 -index_file $index_file -datadir data/gtsrb/prepared/ -on_both > logs/eval/rq2/train/cifar10/$seed.gl.out &
	wait $!

	python3 run_mdl.py -init_mdl data/models/cifar_simple_90p.h5 -patch results/rq2/on_train/cm/random/model.rd.$seed.pkl -seed $seed -which_data cifar10 -num_label 10 -rq 2 -index_file $index_file -datadir data/gtsrb/prepared/ -on_both > logs/eval/rq2/train/cifar10/$seed.rd.out &
	wait $!
done
