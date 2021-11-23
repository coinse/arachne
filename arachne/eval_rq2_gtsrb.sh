#!/bin/bash

index_file="indices/GTSRB/rq1/wo_hist/GTSRB.init_pred.indices.csv"

for seed in {0..29}
do
	# localiser
	python3 run_mdl.py -init_mdl data/models/GTSRB_NEW/simple/gtsrb.model.0.wh.0.h5 -patch results/rq2/on_train/gtsrb/localiser/model.loc.$seed.pkl -seed $seed -which_data GTSRB -num_label 43 -batch_size 10192 -rq 2 -index_file $index_file -datadir data/gtsrb/prepared/ -on_both > logs/eval/rq2/train/gtsrb/$seed.loc.out &
	wait $!

	python3 run_mdl.py -init_mdl data/models/GTSRB_NEW/simple/gtsrb.model.0.wh.0.h5 -patch results/rq2/on_train/gtsrb/gradient_loss/model.gl.$seed.pkl -seed $seed -which_data GTSRB -num_label 43 -batch_size 10192 -rq 2 -index_file $index_file -datadir data/gtsrb/prepared/ -on_both > logs/eval/rq2/train/gtsrb/$seed.gl.out &
	wait $!

	python3 run_mdl.py -init_mdl data/models/GTSRB_NEW/simple/gtsrb.model.0.wh.0.h5 -patch results/rq2/on_train/gtsrb/random/model.rd.$seed.pkl -seed $seed -which_data GTSRB -num_label 43 -batch_size 10192 -rq 2 -index_file $index_file -datadir data/gtsrb/prepared/ -on_both > logs/eval/rq2/train/gtsrb/$seed.rd.out &
	wait $!
done
