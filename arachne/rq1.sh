#!/bin/bash

datadir=$1
which_data=$2
dest=$3
loc=$4

seed=0 # can be between 0 - 29

if [ $which_data == 'cifar10' ]
then
	init_pred_file="indices/cm/cifar10.init_pred.indices.csv"
	python3 main_rq1.py -init_pred_file $init_pred_file -aft_pred_file indices/cm/rq1/$seed/cifar10.init_pred.indices.csv -num_label 10 -datadir $datadir -which simple_cm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method $loc -path_to_keras_model data/models/faulty_models/c10/cifar10/cifar10_simple_seed${seed}.h5 -seed ${seed} -dest $dest/rq1/cm/$loc
else
	init_pred_file="indices/fm/FM.init_pred.indices.csv"	
	python3 main_rq1.py -init_pred_file $init_pred_file -aft_pred_file indices/fm/rq1/$seed/FM.init_pred.indices.csv -num_label 10 -datadir $datadir -which simple_fm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method $loc -path_to_keras_model data/models/faulty_models/fm/fmnist_simple_seed${seed}.h5 -seed ${seed} -dest $dest/rq1/fm/$loc
fi
	


