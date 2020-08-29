#!/bin/bash

which_data=$1
datadir=$2
patch_key=$3
dest=$4

iter_num=100
seed=0 # can be 0..29
if [ $which_data == 'fashion_mnist' ]
then
	indices_file="indices/fm/fashion_mnist.misclf.indices.csv"
	for pa in 1 #2 4 8 10
	do
		python3 main_rq4.py -datadir $datadir -which simple_fm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key $patch_key -seed $seed -iter_num $iter_num -path_to_keras_model data/models/fmnist_simple.h5 -target_indices_file $indices_file -dest $dest/fm -patch_aggr $pa &
		wait $!
	done
else
	indices_file="indices/cm/cifar10.misclf.indices.csv"
	for pa in 1 #2 4 8 10
	do
		python3 main_rq4.py -datadir $datadir -which simple_cm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key $patch_key -seed $seed -iter_num $iter_num -path_to_keras_model data/models/cifar_simple_90p.h5 -target_indices_file $indices_file -dest $dest/cm/ -patch_aggr $pa &
		wait $!
	done
fi
