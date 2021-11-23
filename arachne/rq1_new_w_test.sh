#!/bin/bash

CIFAR10_indices="indices/cm/test/cifar10.init_pred.indices.csv"
#for seed in {1..39} #30..39}
#do
#	#echo "python3 main_rq1.py -init_pred_file indices/cm/cifar10.init_pred.indices.csv -num_label 10 -datadir data/cm -which simple_cm -which_data cifar10 -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method localiser -path_to_keras_model data/models/faulty_models/by_tweak/only_brk/mv/cifar10/1/cifar_simple_90p_seed${seed}.h5 -path_to_faulty_model data/models/faulty_models/by_tweak/only_brk/mv/cifar10/1/cifar_simple_90p_seed${seed}.h5 -seed ${seed} -target_all 1 -gt_file data/models/faulty_models/by_tweak/only_brk/mv/cifar10/1/faulty_nws.${seed}.pkl > logs/rq1_test_run/loc.$seed.out &"
#	python3 main_rq1.py -init_pred_file $CIFAR10_indices -num_label 10 -datadir data/cm -which simple_cm -which_data cifar10 -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method localiser -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/cifar10.target.fault_ids.csv -on_test > logs/rq1_new_fault/test_run/loc.$seed.c10.out &
#	wait $!
#	
#	python3 main_rq1.py -init_pred_file $CIFAR10_indices -num_label 10 -datadir data/cm -which simple_cm -which_data cifar10 -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method gradient_loss -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/cifar10.target.fault_ids.csv  -on_test > logs/rq1_new_fault/test_run/gl.$seed.c10.out &
#	wait $!
#	
#	python3 main_rq1.py -init_pred_file $CIFAR10_indices -num_label 10 -datadir data/cm -which simple_cm -which_data cifar10 -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method random -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/cifar10.target.fault_ids.csv -on_test > logs/rq1_new_fault/test_run/rd.$seed.c10.out &
#	wait $!
#done

FM_indices="indices/fm/test/fashion_mnist.init_pred.indices.csv"
#for seed in {1..31}
#do
        #echo "python3 main_rq1.py -init_pred_file indices/cm/cifar10.init_pred.indices.csv -num_label 10 -datadir data/cm -which simple_cm -which_data cifar10 -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method localiser -path_to_keras_model data/models/faulty_models/by_tweak/only_brk/mv/cifar10/1/cifar_simple_90p_seed${seed}.h5 -path_to_faulty_model data/models/faulty_models/by_tweak/only_brk/mv/cifar10/1/cifar_simple_90p_seed${seed}.h5 -seed ${seed} -target_all 1 -gt_file data/models/faulty_models/by_tweak/only_brk/mv/cifar10/1/faulty_nws.${seed}.pkl > logs/rq1_test_run/loc.$seed.out &"
#        echo "python3 main_rq1.py -init_pred_file $FM_indices -num_label 10 -datadir data/fm -which simple_fm -which_data fashion_mnist -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method localiser -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/fm.target.fault_ids.csv -on_test  > logs/rq1_new_fault/test_run/loc.$seed.fm.out &"
#        wait $!

        #python3 main_rq1.py -init_pred_file $FM_indices -num_label 10 -datadir data/fm -which simple_fm -which_data fashion_mnist -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method gradient_loss -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/fm.target.fault_ids.csv -on_test  > logs/rq1_new_fault/test_run/gl.$seed.fm.out &
        #wait $!

        #python3 main_rq1.py -init_pred_file $FM_indices -num_label 10 -datadir data/fm -which simple_fm -which_data fashion_mnist -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method random -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/fm.target.fault_ids.csv -on_test > logs/rq1_new_fault/test_run/rd.$seed.fm.out &
        #wait $!
#done


# GTSRB
GTSRB_indices="indices/GTSRB/rq1/wo_hist/test/GTSRB.init_pred.indices.csv"
for seed in 32 33 34 #{32..34} #{1..34}
do
	python3 main_rq1.py -init_pred_file $GTSRB_indices -num_label 43 -datadir data/gtsrb/prepared/ -which GTSRB -which_data GTSRB -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method localiser -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/gtsrb.target.fault_ids.csv -w_hist 0 -on_test > logs/rq1_new_fault/test_run/loc.${seed}.gtsrb.out &
	wait $!
	
	python3 main_rq1.py -init_pred_file $GTSRB_indices -num_label 43 -datadir data/gtsrb/prepared/ -which GTSRB -which_data GTSRB -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method gradient_loss -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/gtsrb.target.fault_ids.csv -w_hist 0 -on_test > logs/rq1_new_fault/test_run/gl.${seed}.gtsrb.out &
	wait $!
#	
	python3 main_rq1.py -init_pred_file $GTSRB_indices -num_label 43 -datadir data/gtsrb/prepared/ -which GTSRB -which_data GTSRB -tensor_name_file data/tensor_names/tensor.lastLayer.names -loc_method random -seed ${seed} -target_all 1 -fid_file data/rq1_fault_ids/gtsrb.target.fault_ids.csv -w_hist 0 -on_test > logs/rq1_new_fault/test_run/rd.${seed}.gtsrb.out &
	wait $!
done

