#!/bin/bash

datadir=$1
which=$2 
dest=$3

iter_num=100
patch_aggr=10 #
patch_key="rq5"
seed=0 # 0 ~ 29

model_dir="../final_data/models/rq5"
iter_num=100
index_dir="../final_data/indices"
logdir="logs/rq5"

if [ ! -d "$logdir" ]; then
    mkdir $logdir
fi

if [ $which == 'cnn1' ]
then
    if [ ! -d "$logdir/cnn1" ]; then 
        mkdir $logdir/cnn1
    fi
    path_to_keras_model="$model_dir/ApricotCNN1_full.h5"
    indices_file="$index_dir/cnn1/test/cifar10.init_pred.indices.csv"
    which_data='cifar10'
    num_label=10
    datadir=$datadir/cm
elif [ $which == 'cnn2' ]
then
    if [ ! -d "$logdir/cnn2" ]; then 
        mkdir $logdir/cnn2 
    fi
    path_to_keras_model="$model_dir/ApricotCNN2_full.h5"
    indices_file="$index_dir/cnn2/test/cifar10.init_pred.indices.csv"
    which_data='cifar10'
    num_label=10
    datadir=$datadir/cm
elif [ $which == 'cnn3' ]
then
    if [ ! -d "$logdir/cnn3" ]; then 
        mkdir $logdir/cnn3 
    fi
    path_to_keras_model="$model_dir/ApricotCNN3_full.h5"
    indices_file="$index_dir/cnn3/test/cifar10.init_pred.indices.csv"
    which_data='cifar10'
    num_label=10
    datadir=$datadir/cm
elif [ $which == 'GTSRB' ]
then
    if [ ! -d "$logdir/GTSRB" ]; then 
        mkdir $logdir/GTSRB 
    fi
    path_to_keras_model="$model_dir/gtsrb.model.0.wh.0.h5"
    indices_file="$index_dir/GTSRB/test/GTSRB.init_pred.indices.csv"
    which_data='GTSRB'
    num_label=43
    datadir=$datadir/gtsrb
elif [ $which == 'fm' ]
then
    if [ ! -d "$logdir/fm" ]; then 
        mkdir $logdir/fm 
    fi
    path_to_keras_model="$model_dir/fm_v2.h5"
    num_label=10
    which_data='fm_for_rq5'
    indices_file="$index_dir/fm/for_rq5/val/fashion_mnist.init_pred.indices.csv"
    datadir=$datadir/fm/for_rq5
else # should implement a complex model for fashion_mnist
    echo "Not supported ".$which
    exit 0
fi

for top_n in 0 1 2
do
    python3 main.py \
    -datadir $datadir \
    -which $which \
    -which_data $which_data \
    -patch_key ${patch_key}.${top_n}.${seed} \
    -seed $seed \
    -path_to_keras_model $path_to_keras_model \
    -iter_num $iter_num \
    -target_indices_file $indices_file \
    -dest $dest/$which \
    -patch_aggr $patch_aggr \
    -batch_size 512 \
    -top_n $top_n \
    -num_label $num_label > $logdir/$which/$patch_aggr.rq5.$top_n.$seed.out &
    wait $!
done
