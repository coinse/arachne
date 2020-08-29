#!/bin/bash

rq=$1
w=$2
datadir=$3
dest=$4
which_data=$5
which=$6
val_index_file=$7

echo $rq
if [ $rq != 'rq6' ] && [ $rq != 'rq5' ]
then
	if [ $which_data == 'fashion_mnist' ]
	then
		model='data/models/fmnist_simple.h5'
		which='simple_fm'
	else
		model='data/models/cifar_simple_90p.h5'
		which='simple_cm'
	fi
	python3 run_model_for_fm_c10.py -model $model -datadir $datadir -dest $dest -w $w -which $which -which_data $which_data
elif [ $rq == 'rq5' ]
then
	if [ $which == 'cnn1' ]
	then
		model='data/models/ApricotCNN1_full.h5'
	elif [ $which == 'cnn2' ]
	then
		model='data/models/ApricotCNN2_full.h5'
	else
		model='data/models/ApricotCNN3_full.h5'
	fi
	python3 run_model_for_fm_c10.py -model $model -datadir $datadir -dest $dest -w $w -which $which -which_data $which_data -val_index_file $val_index_file
else
	python3 run_model_for_lfw.py -model data/models/saved_models/LFW_GenderClassifier.pth -datadir $datadir -dest $dest -w $w
fi
