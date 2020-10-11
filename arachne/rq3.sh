#!/bin/bash

which_data=$1
datadir=$2
patch_key=$3
dest=$4
n=$5

iter_num=100
seed=$n # can be 0 .. 29
patch_aggr=10

use_ewc=0

if [ $which_data == 'fashion_mnist' ]
then
	indices_file="indices/fm/fashion_mnist.misclf.indices.csv"
	python3 main_rq3.py -datadir $datadir -which simple_fm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key ${patch_key}.${seed} -seed $seed -path_to_keras_model data/models/fmnist_simple.h5 -iter_num $iter_num -target_indices_file $indices_file -dest $dest/fm -patch_aggr $patch_aggr -use_ewc ${use_ewc}
else # cifar10
	indices_file="indices/cm/cifar10.misclf.indices.csv"
	python3 main_rq3.py -datadir $datadir -which simple_cm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key ${patch_key}.${seed} -seed $seed -path_to_keras_model data/models/cifar_simple_90p.h5 -iter_num $iter_num -target_indices_file $indices_file -dest $dest/cm -patch_aggr $patch_aggr -use_ewc ${use_ewc}
fi
