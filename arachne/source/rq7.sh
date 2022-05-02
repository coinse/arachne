#!/bin/bash

datadir=$1 
dest=$2 

seed=0 # 0 .. 29
patch_key=rq7
iter_num=100
patch_aggr=1
model_dir="../final_data/models/rq7"
iter_num=100
index_dir="../final_data/indices"
path_to_keras_model="$model_dir/tweets.sa.mdl.best.h5"
num_label=3
which='lstm'
which_data='us_airline'
indices_file="$index_dir/lstm/test/us_airline.init_pred.indices.csv"

logdir="logs/rq7"
if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi

python3 main.py \
-datadir $datadir/us_airline \
-which $which \
-which_data $which_data \
-patch_key ${patch_key}.${top_n}.${seed} \
-seed $seed \
-path_to_keras_model $path_to_keras_model \
-iter_num $iter_num \
-target_indices_file $indices_file \
-dest $dest \
-patch_aggr $patch_aggr \
-batch_size 512 \
-top_n 0 \
-num_label $num_label > $logdir/us_airline.pa$patch_aggr.rq7.$top_n.$seed.out &
wait $!
