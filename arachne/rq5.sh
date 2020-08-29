#!/bin/bash

which=$1
datadir=$2
patch_key=$3
dest=$4
org_label=$5
pred_label=$6

seed=0 # can be 0..29
patch_aggr=10
if [ $which == 'cnn1' ]
then
	path_to_keras_model="data/models/ApricotCNN1_full.h5"
	indices_file="indices/cnn1/test/cifar10.misclf.indices.csv"

elif [ $which == 'cnn2' ]
then
	path_to_keras_model="data/models/ApricotCNN2_full.h5"
	indices_file="indices/cnn2/test/cifar10.misclf.indices.csv"
elif [ $which == 'cnn3' ]
then
	path_to_keras_model="data/models/ApricotCNN3_full.h5"
	indices_file="indices/cnn3/test/cifar10.misclf.indices.csv"
else
	echo "Not supported ".$which	
	exit 0
fi

python3 main_rq5.py -datadir $datadir -which $which -which_dat cifar10 -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key $patch_key -path_to_keras_model $path_to_keras_model -seed $seed -iter_num 100 -target_indices_file $indices_file -dest $dest/rq5/cnn1 -patch_aggr $patch_aggr -org_label $org_label -pred_label $pred_label
