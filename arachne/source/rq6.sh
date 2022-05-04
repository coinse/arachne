#!/bin/bash

datadir=$1
dest=$2
pa=2
seed=0 # 0 .. 29

model_dir="final_data/models/rq6"
iter_num=100
index_dir="final_data/indices"
logdir="logs/rq6"

if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi

python3 main_rq3_7.py \
-datadir $datadir/lfw/lfw_data \
-which lfw_vgg \
-which_data lfw \
-patch_key rq6.0.$seed \
-seed $seed \
-path_to_keras_model $model_dir/LFW_gender_classifier_best.h5 \
-iter_num 100 \
-target_indices_file $index_dir/lfw/test/lfw.init_pred.indices.csv \
-dest $dest \
-patch_aggr $pa \
-batch_size 512 \
-female_lst_file $datadir/lfw/lfw_np/female_names_lfw.txt \
-top_n 0 \
-num_label 2 > $logdir/pa$pa.$seed.out &
wait $!
