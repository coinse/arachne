#!/bin/bash

which="lfw_vgg"
datadir="data/lfw_data"
indices_file="indices/lfw/v2/test/lfw.init_pred.indices.csv"
path_to_keras_model="data/models/saved_models/version_2/LFW_gender_classifier_best.h5"

pa=$1

for top_n in 0 
do
	for seed in {0..29}
	do
		#python3 run_mdl.py -datadir $datadir -init_mdl $path_to_keras_model -patch results/rq6/pa1/all/use_both/model.misclf-rq6.$top_n.$seed-*.pkl  -which_data lfw -num_label 2 -batch_size 512 -rq 5 -index_file $indices_file -on_both -top_n $top_n -female_lst_file data/lfw_np/female_names_lfw.txt > logs/eval/rq6/$top_n.$seed.out &
		# correct one
		#python3 run_mdl.py -datadir $datadir -init_mdl $path_to_keras_model -patch results/rq6/v2/all/use_both/pa$pa/model.misclf-rq6.$top_n.$seed-*.pkl  -which_data lfw -num_label 2 -batch_size 512 -rq 5 -index_file $indices_file -on_both -top_n $top_n -female_lst_file data/lfw_np/female_names_lfw.txt > logs/eval/rq6/v2/pa$pa.$top_n.$seed.out &

		python3 run_mdl.py -datadir $datadir -init_mdl $path_to_keras_model -patch results/rq6/v2/all/use_both/new/pa1/model.misclf-rq6.$top_n.$seed-*.pkl  -which_data lfw -num_label 2 -batch_size 512 -rq 5 -index_file $indices_file -on_both -top_n $top_n -female_lst_file data/lfw_np/female_names_lfw.txt > logs/eval/rq6/v2/new/pa$pa.$top_n.$seed.out &
		wait $!
	done
done
