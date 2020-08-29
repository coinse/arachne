#!/bin/bash

datadir=$1
which_data=$2
loc=$3
patch_target_key=$4
dest=$5

iter_num=100
seed=0 # can be 0 .. 29
patch_aggr=10
if [ $which_data == 'fashion_mnist' ]
then
	indices_file="indices/fm/fashion_mnist.misclf.indices.csv"
	python3 main_rq2.py -datadir $datadir -which simple_fm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method $loc -patch_target_key $patch_target_key -path_to_keras_model data/models/fmnist_simple.h5 -seed $seed -iter_num $iter_num -target_indices_file $indices_file -dest $dest/fm/$loc -patch_aggr $patch_aggr
else
	indices_file="indices/cm/cifar10.misclf.indices.csv"
	python3 main_rq2.py -datadir $datadir -which simple_cm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method $loc -patch_target_key $patch_target_key -path_to_keras_model data/models/cifar_simple_90p.h5 -seed $seed -iter_num $iter_num -target_indices_file $indices_file -dest $dest/cm/$loc -patch_aggr $patch_aggr
fi
