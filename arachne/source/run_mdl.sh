#!/bin/bash

rq=$1
datadir=$2
which_data=$3
path_to_patch=$4
top_n=$5 # should match with path_to_patch
which=$6 # only needed for rq5
batch_size=512

#logdir="logs/eval"
#if [ ! -d "$logdir" ]; then
#    mkdir $logdir
#    mkdir $logdir/$rq
#elif [ ! -d "$logdir/$rq" ]; then 
#    mkdir $logdir/$rq
#fi
index_dir="../final_data/indices"

if [ $rq -eq 'rq2' ] || [ $rq -eq 'rq3' ] || [ $rq -eq 'rq4' ]; then 
    # set an index & model file 
    model_dir="../final_data/models/simple"

    if [ $which_data == 'fashion_mnist' ]; then
        model="$model_dir/fmnist_simple.h5"
        datadir="$datadir/fm"
        indices_file="$index_dir/fm/test/fashion_mnist.init_pred.indices.csv"
        num_label=10
    elif [ $which_data -eq 'cifar10' ]; then 
        model="$model_dir/cifar_simple_90p.h5"
        datadir="$datadir/cm"
        indices_file="$index_dir/cm/test/cifar10.init_pred.indices.csv"
        num_label=10
    elif [ $which_data -eq 'GTSRB' ]; then
        model="$model_dir/gtsrb.model.0.wh.0.h5"
        datadir="$datadir/gtsrb/prepared"
        indices_file="$index_dir/GTSRB/simple/test/GTSRB.init_pred.indices.csv"
        num_label=43
    else
        echo "unsupported data: $which_data"
        exit 1 
    fi
    # 
    if [ $rq -eq 'rq4' ]; then top_n=0; fi
    
    python3 run_mdl.py 
    -init_mdl $model \
    -patch $path_to_patch \ 
    -datadir $datadir \
    -index_file $indices_file \
    -which_data $which_data \
    -num_label $num_label \
    -batch_size $batch_size \
    -rq $rq \
    -top_n $top_n #> $logdir/$rq/log.out &
elif [ $rq -eq 'rq5' ]; then
    model_dir="../final_data/models/rq5"

    if [ $which == 'cnn1' ]
    then
        model="$model_dir/ApricotCNN1_full.h5"
        indices_file="$index_dir/cnn1/test/cifar10.init_pred.indices.csv"
        which_data='cifar10'
        num_label=10
        datadir=$datadir/cm
    elif [ $which == 'cnn2' ]
    then
        model="$model_dir/ApricotCNN2_full.h5"
        indices_file="$index_dir/cnn2/test/cifar10.init_pred.indices.csv"
        which_data='cifar10'
        num_label=10
        datadir=$datadir/cm
    elif [ $which == 'cnn3' ]
    then
        model="$model_dir/ApricotCNN3_full.h5"
        indices_file="$index_dir/cnn3/test/cifar10.init_pred.indices.csv"
        which_data='cifar10'
        num_label=10
        datadir=$datadir/cm
    elif [ $which == 'GTSRB' ]
    then
        model="$model_dir/gtsrb.model.0.wh.0.h5"
        indices_file="$index_dir/GTSRB/test/GTSRB.init_pred.indices.csv"
        which_data='GTSRB'
        num_label=43
        datadir=$datadir/gtsrb/prepared
    elif [ $which == 'fm' ]
    then
        model="$model_dir/fm_v2.h5"
        num_label=10
        which_data='fm_for_rq5'
        indices_file="$index_dir/fm/for_rq5/val/fashion_mnist.init_pred.indices.csv"
        datadir=$datadir/fm/for_rq5
    else # should implement a complex model for fashion_mnist
        echo "Not supported ".$which
        exit 0
    fi

    python3 run_mdl.py 
    -init_mdl $model \
    -patch $path_to_patch \ 
    -datadir $datadir \
    -index_file $indices_file \
    -which_data $which_data \
    -num_label $num_label \
    -batch_size $batch_size \
    -rq $rq \
    -top_n $top_n #> $logdir/$rq/log.out &

elif [ $rq -eq 'rq6' ]; then 
    model_dir="../final_data/models/rq6"
    model="$model_dir/LFW_gender_classifier_best.h5"
    which_data='lfw'
    num_label=2
    indices_file="$index_dir/lfw/test/lfw.init_pred.indices.csv"
    female_lst_file="$datadir/lfw_np/female_names_lfw.txt"

    python3 run_mdl.py 
    -init_mdl $model \
    -patch $path_to_patch \ 
    -datadir $datadir/lfw_data \
    -index_file $indices_file \
    -which_data $which_data \
    -num_label $num_label \
    -batch_size $batch_size \
    -rq $rq \
    -top_n $top_n \
    -female_lst_file female_lst_file  #> $logdir/$rq/log.out &

elif [ $rq -eq 'rq7' ]; then 
    model_dir="../final_data/models/rq7"
    model="$model_dir/tweets.sa.mdl.best.h5"
    datadir="$datadir/us_airline"
    which_data='us_airline'
    num_label=3
    indices_file="$index_dir/lstm/test/us_airline.init_pred.indices.csv"

    python3 run_mdl.py 
    -init_mdl $model \
    -patch $path_to_patch \ 
    -datadir $datadir \
    -index_file $indices_file \
    -which_data $which_data \
    -num_label $num_label \
    -batch_size $batch_size \
    -rq $rq \
    -top_n $top_n  #> $logdir/$rq/log.out &
else
    echo "Not supported ".$rq
    exit 0
fi