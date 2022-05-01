#!/bin/bash

which_data=$1
drange=$2

for i in {0..29}
do
	if [ $which_data == 'cifar10' ]
	then 
		#python3.6 record_misclf_indices.py -model data/models/faulty_models/c10/cifar10/all_layers/cifar10_allLayers_seed${i}.h5 -datadir data/cm/ -dest indices/cm/rq1/all_layers/${i} -which_data cifar10 -is_train 1 -w_hist 0 > logs/c10.f.$i.out &
		python3.6 record_misclf_indices.py -model data/models/faulty_models/c10/cifar10/all_layers/cifar10_allLayers_seed${i}.h5 -datadir data/cm/ -dest indices/cm/rq1/all_layers/${i} -which_data cifar10 -is_train 1 -w_hist 0 > logs/c10.f.$i.out &
		#data/models/faulty_models/by_tweak/cifar10/3/cifar_simple_90p_seed25.h5
		python3.6 record_misclf_indices.py -model data/models/faulty_models/by_tweak/cifar10/3/cifar_simple_90p_seed${i}.h5 -datadir data/cm/ -dest indices/cm/rq1/by_tweak/all_layers/${i} -which_data cifar10 -is_train 1 -w_hist 0 > logs/c10.f.$i.tweak.out &
		wait $!
	elif [ $which_data == 'fashion_mnist' ]
	then
		python3.6 record_misclf_indices.py -model data/models/faulty_models/fm/$drange/fmnist_simple_seed${i}.h5 -datadir data/fm -dest indices/fm/rq1/$drange/${i} -which_data $which_data -is_train 1 -w_hist 0 > logs/fm.f.$i.$drange.out &
		wait $! 
	else
		python3.6 record_misclf_indices.py -model data/models/faulty_models/GTSRB/all_layers/gtsrb.model.0.wh.0_seed${i}.h5 -datadir data/GTSRB/ -dest indices/GTSRB/rq1/wo_hist/all_layers/${i} -which_data GTSRB -is_train 1 -w_hist 0 > logs/gtsrb.f.$i.out &
		wait $!
	fi
done
