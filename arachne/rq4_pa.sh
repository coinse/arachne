#!/bin/bash

which_data=$1
datadir=$2
patch_key=$3
dest=$4
n=$5
patch_aggr=$6


iter_num=100
seed=$n # can be 0 .. 29
if [ $which_data == 'fashion_mnist' ]
then
	#indices_file="indices/fm/fashion_mnist.misclf.indices.csv"
	indices_file="indices/fm/test/fashion_mnist.init_pred.indices.csv"
	python3 main_rq3.py -datadir $datadir -which simple_fm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key ${patch_key}.${seed} -seed $seed -path_to_keras_model data/models/fmnist_simple.h5 -iter_num $iter_num -target_indices_file $indices_file -dest $dest/fm/$patch_aggr -patch_aggr $patch_aggr -batch_size 10192 -top_n 0
elif [ $which_data == 'cifar10' ]
then
	#indices_file="indices/cm/cifar10.misclf.indices.csv"
	indices_file="indices/cm/test/cifar10.init_pred.indices.csv"
	python3 main_rq3.py -datadir $datadir -which simple_cm -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key ${patch_key}.${seed} -seed $seed -path_to_keras_model data/models/cifar_simple_90p.h5 -iter_num $iter_num -target_indices_file $indices_file -dest $dest/cm/$patch_aggr -patch_aggr $patch_aggr -top_n 0
else
	#indices_file="indices/GTSRB/rq1/wo_hist/GTSRB.misclf.indices.csv"
	indices_file="indices/GTSRB/rq1/wo_hist/test/GTSRB.init_pred.indices.csv"
	python3 main_rq3.py -datadir $datadir -which GTSRB -which_data $which_data -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key ${patch_key}.${seed} -seed $seed -path_to_keras_model data/models/GTSRB_NEW/simple/gtsrb.model.0.wh.0.h5 -iter_num $iter_num -target_indices_file $indices_file -dest $dest/gtsrb/$patch_aggr -batch_size 10192  -patch_aggr $patch_aggr -num_label 43 -w_hist 0 -top_n 0
fi
