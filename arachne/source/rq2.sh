#!/bin/bash

datadir=$1
loc_method=$2
which_data=$3
dest=$4
patch_target_key="rq2"

model_dir="../final_data/models/simple"
seed=0 # 0 ~ 29 (int)
iter_num=100
patch_aggr=10

index_dir="../final_data/indices"
logdir="logs/rq2"

if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi

if [ $which_data -eq 'fashion_mnist' ]
then
    if [ ! -d "$logdir/fm" ]; then 
        mkdir $logdir/fm 
    fi
    indices_file="$index_dir/fm/test/fashion_mnist.init_pred.indices.csv"

    python3 main_rq2.py \
    -datadir $datadir/fm \
    -which simple_fm \
    -which_data $which_data \
    -loc_method $loc_method \
    -patch_target_key ${patch_target_key}.${seed} \
    -path_to_keras_model $model_dir/fmnist_simple.h5 \
    -seed $seed \
    -iter_num $iter_num \
    -target_indices_file $indices_file \
    -dest $dest/fm/$loc_method \
    -patch_aggr $patch_aggr \
    -batch_size 10192 > $logdir/fm/$loc_method.$seed.fm.out &
elif [ $which_data -eq 'cifar10' ]
then
    if [ ! -d "$logdir/c10" ]; then 
        mkdir $logdir/c10
    fi
    indices_file="$index_dir/cm/test/cifar10.init_pred.indices.csv"

    python3 main_rq2.py \
    -datadir $datadir/cm \
    -which simple_cm \
    -which_data $which_data \
    -loc_method $loc_method \
    -patch_target_key ${patch_target_key}.${seed} \
    -path_to_keras_model $model_dir/cifar_simple_90p.h5 \
    -seed $seed \
    -iter_num $iter_num \
    -target_indices_file $indices_file \
    -dest $dest/cm/$loc_method \
    -patch_aggr $patch_aggr > $logdir/c10/$loc_method.$seed.c10.out &
elif [ $which_data -eq 'GTSRB' ]
then
    # GTSRB
    if [ ! -d "$logdir/gtsrb" ]; then 
        mkdir $logdir/gtsrb
    fi 

    indices_file="$index_dir/GTSRB/simple/test/GTSRB.init_pred.indices.csv"
    
    python3 main_rq2.py \
    -datadir $datadir/gtsrb/prepared \
    -which GTSRB \
    -which_data $which_data \
    -loc_method $loc_method\
    -patch_target_key ${patch_target_key}.${seed} \
    -path_to_keras_model $model_dir/gtsrb.model.0.wh.0.h5 \
    -seed $seed \
    -iter_num $iter_num \
    -target_indices_file $indices_file \
    -dest $dest/gtsrb/$loc_method \
    -patch_aggr $patch_aggr \
    -num_label 43 \
    -batch_size 10192 > $logdir/gtsrb/$loc_method.$seed.gtsrb.out &
else
    echo "unsupported data: $which_data"
fi
