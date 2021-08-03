#!/bin/bash

datadir=$1
which_data=$2
dest=$3
loc=$4
target_all=$5
seed=$6 # can be between 0 - 29

if [ $which_data == 'cifar10' ]
then
	init_pred_file="indices/cm/cifar10.init_pred.indices.csv"
	python3 main_rq1.py -init_pred_file $init_pred_file -aft_pred_file indices/cm/rq1/all_layers/$seed/cifar10.init_pred.indices.csv -num_label 10 -datadir $datadir -which simple_cm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method $loc -path_to_keras_model data/models/faulty_models/c10/cifar10/all_layers/cifar10_allLayers_seed${seed}.h5 -seed ${seed} -target_all ${target_all}
elif [ $which_data == 'fashion_mnist' ]
then
	init_pred_file="indices/fm/fashion_mnist.init_pred.indices.csv"	
	python3 main_rq1.py -init_pred_file $init_pred_file -aft_pred_file indices/fm/rq1/all_layers/$seed/FM.init_pred.indices.csv -num_label 10 -datadir $datadir -which simple_fm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method $loc -path_to_keras_model data/models/faulty_models/fm/all_layers/fmnist_simple_seed${seed}.h5 -seed ${seed} -target_all ${target_all}
else
	init_pred_file="indices/GTSRB/train/wo_hist/GTSRB.init_pred.indices.csv"
	python3 main_rq1.py -init_pred_file $init_pred_file -aft_pred_file indices/GTSRB/rq1/all_layers/$seed/GTSRB.init_pred.indices.csv -num_label 43 -datadir $datadir -which gtsrb -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method $loc -path_to_keras_model data/models/faulty_models/GTSRB/all_layers/gtsrb.model.0.wh.0_seed${seed}.h5 -seed ${seed} -w_hist 0 -target_all ${target_all}
fi
	


