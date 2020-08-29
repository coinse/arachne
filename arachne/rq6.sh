#!/bin/bash

datadir=$1
patch_key=$2
dest=$3

seed=0 # can be 0 .. 29
patch_aggr=1
indices_file="indices/lfw/test/lfw.misclf.indices.csv"

python3 main_rq6.py -datadir $datadir -which lfw_vgg -which_data lfw -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key $patch_key -path_to_keras_model data/models/saved_models/LFW_GenderClassifier.pth -seed 0 -iter_num 100 -target_indices_file $indices_file -dest $dest/rq6 -patch_aggr $patch_aggr
