#!/bin/bash

#index_file="indices/GTSRB/rq1/wo_hist/GTSRB.init_pred.indices.csv"
index_file="indices/GTSRB/rq1/wo_hist/test/GTSRB.init_pred.indices.csv"
for seed in {0..29}
do
	# localiser
	python3 run_mdl.py -datadir data/gtsrb/prepared/ -init_mdl data/models/GTSRB_NEW/simple/gtsrb.model.0.wh.0.h5 -patch results/rq3/on_test/gtsrb/model.misclf-on_test.${seed}-*.pkl -top_n $seed -which_data GTSRB -num_label 43 -batch_size 10192 -rq 3 -index_file $index_file -on_both > logs/eval/rq3/test/gtsrb/$seed.out &
	wait $!
done
