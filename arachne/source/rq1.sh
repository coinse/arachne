#!/bin/bash

datadir=$1
loc_method=$2 # localiser (BL), gradient_loss (GL), random (Random)
which_data=$3
dest=$4
seed=0 #$3
index_dir="final_data/indices" #
fid_filedir="final_data/rq1/fault_ids/"
logdir="logs/rq1"

if [ ! -d "$logdir" ]; then
	mkdir $logdir
	mkdir $logdir/fm
	mkdir $logdir/c10
	mkdir $logdir/gtsrb
fi

if [ $which_data == 'cifar10' ]; then
	if [ ! -d "$logdir/c10" ]; then 
		mkdir $logdir/c10
	fi

	CIFAR10_indices="$index_dir/cm/test/cifar10.init_pred.indices.csv"
	#for seed in {0..39} 
	#do
	python3 main_rq1.py \
	-init_pred_file $CIFAR10_indices \
	-num_label 10 \
	-datadir $datadir/cm \
	-which simple_cm \
	-which_data cifar10 \
	-loc_method $loc_method \
	-seed ${seed} \
	-target_all 1 \
	-fid_file $fid_filedir/cifar10.target.fault_ids.csv \
	-dest $dest \
	-on_test > $logdir/c10/$loc_method.$seed.c10.out &
	wait $!
	#done
elif [ $which_data == 'fashion_mnist' ]; then 
	if [ ! -d "$logdir/fm" ]; then 
		mkdir $logdir/fm 
	fi
	
	FM_indices="$index_dir/fm/test/fashion_mnist.init_pred.indices.csv"
	#for seed in {0..30}
	#do	
	python3 main_rq1.py \
	-init_pred_file $FM_indices \
	-num_label 10 \
	-datadir $datadir/fm \
	-which simple_fm \
	-which_data fashion_mnist \
	-loc_method $loc_method \
	-seed ${seed} \
	-target_all 1 \
	-fid_file $fid_filedir/fm.target.fault_ids.csv \
	-dest $dest \
	-on_test  > $logdir/fm/$loc_method.$seed.fm.out &
	wait $!
	#done
else
	# GTSRB
	if [ ! -d "$logdir/gtsrb" ]; then 
		mkdir $logdir/gtsrb
	fi 

	GTSRB_indices="$index_dir/GTSRB/simple/test/GTSRB.init_pred.indices.csv"
	#for seed in {0..34}
	#do
	python3 main_rq1.py \
	-init_pred_file $GTSRB_indices \
	-num_label 43 \
	-datadir $datadir/gtsrb \
	-which GTSRB \
	-which_data GTSRB \
	-loc_method $loc_method \
	-seed ${seed} \
	-target_all 1 \
	-fid_file $fid_filedir/gtsrb.target.fault_ids.csv \
	-dest $dest \
	-on_test > $logdir/gtsrb/$loc_method.${seed}.gtsrb.out &
	wait $!
	#done
fi
