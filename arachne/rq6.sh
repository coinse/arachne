#!/bin/bash

datadir=$1
patch_key=$2
dest=$3

seed=0 # can be 0 .. 29
patch_aggr=1
indices_file="indices/lfw/test/lfw.misclf.indices.csv"
female_lst_file='data/lfw_np/female_names_lfw.txt' # store a list of females (for labeling purpose)
iter_num=100

use_ewc=0

python3 main_rq6.py -datadir $datadir -tensor_name_file data/tensor_names/tensor.lastLayer.names -patch_key ${patch_key}.${seed} -path_to_keras_model data/models/saved_models/LFW_GenderClassifier.pth -seed $seed -iter_num $iter_num -target_indices_file $indices_file -dest $dest -patch_aggr $patch_aggr -female_lst_file $female_lst_file -use_ewc ${use_ewc}
