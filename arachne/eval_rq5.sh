#!/bin/bash

which=$1
datadir=$2

if [ $which == 'cnn1' ]
then
	patch_aggr=10
        path_to_keras_model="data/models/ApricotCNN1_full.h5"
        indices_file="indices/cnn1/test/cifar10.init_pred.indices.csv"
        which_data='cifar10'
        num_label=10
elif [ $which == 'cnn2' ]
then
	patch_aggr=10
        path_to_keras_model="data/models/ApricotCNN2_full.h5"
        indices_file="indices/cnn2/test/cifar10.init_pred.indices.csv"
        which_data='cifar10'
        num_label=10
elif [ $which == 'cnn3' ]
then
	patch_aggr=10
        path_to_keras_model="data/models/ApricotCNN3_full.h5"
        indices_file="indices/cnn3/test/cifar10.init_pred.indices.csv"
        which_data='cifar10'
        num_label=10
elif [ $which == 'GTSRB' ]
then
	patch_aggr=10
        path_to_keras_model="data/models/GTSRB_NEW/gtsrb.model.0.wh.0.h5"
        #indices_file="indices/GTSRB/test/wo_hist/GTSRB.init_pred.indices.csv"
	indices_file="indices/GTSRB/test/GTSRB.init_pred.indices.csv"
        which_data='GTSRB'
        num_label=43
elif [ $which == 'lstm' ]
then
	patch_aggr=1
	path_to_keras_model="data/models/us_airline/n_64_dnn1_v2/tweets.sa.mdl.best.h5"
	indices_file="indices/lstm/us_airline/n_64_dnn1_v2/test/us_airline.init_pred.indices.csv"
	num_label=3
	which_data='us_airline'
elif [ $which == 'fm' ]
then
	patch_aggr=10
	path_to_keras_model="data/models/saved_models/fmnist/fm_v2.h5"
	num_label=10
	indices_file='indices/fm/for_rq5/val/fashion_mnist.init_pred.indices.csv'
	which_data='fm_for_rq5'
else # should implement a complex model for fashion_mnist
        echo "Not supported ".$which
        exit 0
fi

for top_n in 0 1 2
do
	for seed in {0..29}
	do
		# localiser # 10192
		if [ $which == 'lstm' ]
		then
			python3 run_mdl.py -datadir $datadir -init_mdl $path_to_keras_model -patch results/rq5/$which/us_airline/model.misclf-rq5.$top_n.$seed-*.pkl  -which_data $which_data -num_label $num_label -batch_size 512 -rq 5 -index_file $indices_file -on_both -top_n $top_n > logs/eval/rq5/lstm/us_airline/test/$patch_aggr.$seed.out &
		else
			#python3 run_mdl.py -datadir $datadir -init_mdl $path_to_keras_model -patch results/rq5/$which/pa_10/model.misclf-rq5.$top_n.$seed-*.pkl  -which_data $which_data -num_label $num_label -batch_size 512 -rq 5 -index_file $indices_file -on_both -top_n $top_n > logs/eval/rq5/test/$which/pa_10/$top_n.$seed.out &
			python3 run_mdl.py -datadir $datadir -init_mdl $path_to_keras_model -patch results/rq5/$which/pa_10/model.misclf-rq5.$top_n.$seed-*.pkl  -which_data $which_data -num_label $num_label -batch_size 512 -rq 5 -index_file $indices_file -on_both -top_n $top_n > logs/eval/rq5/test/$which/pa_10/train/int.$top_n.$seed.out &
		fi
		wait $!
	done
done
