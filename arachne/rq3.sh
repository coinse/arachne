#!/bin/bash

which_data=$1
datadir=$2
patch_key=$3
dest=$4

seed=0 # can be 0 .. 29
patch_aggr=10
if [ $which_data == 'fashion_mnist' ]
then
	indices_file="indices/fm/fashion_mnist.misclf.indices.csv"
	python3 main_rq3.py -datadir $datadir -which simple_fm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key $patch_key -seed $seed -path_to_keras_model data/models/fmnist_simple.h5 -iter_num 100 -target_indices_file $indices_file -dest $dest/rq3/fm -patch_aggr $patch_aggr
else # cifar10
	indices_file="indices/cm/cifar10.misclf.indices.csv"
	python3 main_rq3.py -datadir $datadir -which simple_cm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key $patch_key -seed $seed -path_to_keras_model data/models/cifar_simple_90p.h5 -iter_num 100 -target_indices_file $indices_file -dest $dest/rq3/cm -patch_aggr $patch_aggr
fi
