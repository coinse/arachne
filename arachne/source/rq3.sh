#!/bin/bash

datadir=$1
which_data=$2
dest=$3
patch_key="rq3"

model_dir="../final_data/models/simple"
iter_num=100
seed=0 # 0 ~ 29 (top 1(=0) ~ 30(=29))
patch_aggr=10

index_dir="../final_data/indices"
logdir="logs/rq3"

if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi

if [ $which_data == 'fashion_mnist' ]
then
    if [ ! -d "$logdir/fm" ]; then 
        mkdir $logdir/fm 
    fi
    indices_file="$index_dir/fm/test/fashion_mnist.init_pred.indices.csv"

    python3 main.py \
    -datadir $datadir \
    -which simple_fm \
    -which_data $which_data \
    -patch_key ${patch_key}.${seed} \
    -seed $seed \
    -path_to_keras_model $model_dir/fmnist_simple.h5 \
    -iter_num $iter_num \
    -target_indices_file $indices_file \
    -dest $dest/fm \
    -patch_aggr $patch_aggr \
    -batch_size 10192 > $logdir/fm/$seed.fm.out &
elif [ $which_data == 'cifar10' ]
then
    if [ ! -d "$logdir/c10" ]; then 
        mkdir $logdir/c10
    fi
    indices_file="$index_dir/cm/test/cifar10.init_pred.indices.csv"
    
    python3 main.py \
    -datadir $datadir \
    -which simple_cm \
    -which_data $which_data \
    -patch_key ${patch_key}.${seed} \
    -seed $seed \
    -path_to_keras_model $model_dir/cifar_simple_90p.h5 \
    -iter_num $iter_num \
    -target_indices_file $indices_file \
    -dest $dest/cm \
    -patch_aggr $patch_aggr > $logdir/c10/$seed.c10.out &
elif [ $which_data == 'GTSRB' ]
then
    if [ ! -d "$logdir/gtsrb" ]; then 
        mkdir $logdir/gtsrb
    fi 
    indices_file="$index_dir/GTSRB/simple/test/GTSRB.init_pred.indices.csv"
    
    python3 main.py \
    -datadir $datadir \
    -which GTSRB \
    -which_data $which_data \
    -patch_key ${patch_key}.${seed} \
    -seed $seed \
    -path_to_keras_model $model_dir/gtsrb.model.0.wh.0.h5 \
    -iter_num $iter_num \
    -target_indices_file $indices_file \
    -dest $dest/gtsrb \
    -batch_size 10192  \
    -patch_aggr $patch_aggr \
    -num_label 43 > $logdir/gtsrb/$seed.gtsrb.out &
else
    echo "unsupported data: $which_data"
fi
