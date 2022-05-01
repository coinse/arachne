import argparse
import os, sys
import utils.data_util as data_util
import auto_patch_vk as auto_patch
import time
import numpy as np
from collections.abc import Iterable 

parser = argparse.ArgumentParser()

parser.add_argument("-datadir", action = "store", default = "data", type = str)
parser.add_argument("-which", action = "store", 
	help = 'simple_cm, simple_fm, lstm', type = str)
parser.add_argument('-which_data', action = "store",
	default = 'cifar10', type = str, help = 'fashion_mnist,cifaf10,GTSRB,us_airline')
parser.add_argument("-target_layer_idx", action = "store", default = -1, type = int)
parser.add_argument("-patch_key", action = "store", default = "key")
parser.add_argument("-path_to_keras_model", action = 'store', default = None)
parser.add_argument("-seed", action = "store", default = 1, type = int)
parser.add_argument("-iter_num", action = "store", default = 100, type = int)
parser.add_argument("-target_indices_file", action = "store", default = None)
parser.add_argument("-dest", default = ".", type = str)
parser.add_argument("-patch_aggr", action = 'store', default = None, type = float)
parser.add_argument("-num_label", type = int, default = 10)
parser.add_argument("-batch_size", type = int, default = None)
parser.add_argument("-on_train", action = "store_true", help = "if given, then evaluate on the training data")
parser.add_argument("-top_n", type = int, default = None)
parser.add_argument("-female_lst_file", action = 'store', 
	default = 'data/lfw_np/female_names_lfw.txt', type = str, help = "for rq6")
parser.add_argument("-loc_method", action = "store", default = 'localiser')


args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)
loc_dest = os.path.join(args.dest, "loc")
os.makedirs(loc_dest, exist_ok=True)

iter_num = args.iter_num
num_label = args.num_label

if args.top_n is None:
	# to target a unique type of misbehaviour per run
	top_n = args.seed 
else: # mainly for rq4 and rq5 (targeting top 3)
	top_n = args.top_n

if args.which_data != 'fm_for_rq5': 
	train_data, test_data = data_util.load_data(
		args.which_data, args.datadir,
		path_to_female_names = args.female_lst_file)

	target_data = test_data if not args.on_train else train_data
	target_X, target_y = target_data
	outs = data_util.get_balanced_dataset(args.target_indices_file, top_n, idx = 0)

	if not isinstance(outs, Iterable):
		print ("There are only {} number of unqiue misclassification types. vs {}".format(outs, top_n))
		sys.exit()
	else:
		(misclf_key, abs_indices, new_test_indices, _) = outs
		target_data = (target_X[new_test_indices], target_y[new_test_indices])
		indices = [new_test_indices.index(idx) for idx in abs_indices]

	misclf_true, misclf_pred = misclf_key
else: # fm_for_rq5
	msg = "currently, for rq5 + fm, we only support running on the half of test data"
	assert not args.on_train, msg

	# retrive the half of the test data, denoted as validation data
	target_data = data_util.load_rq5_fm_test_val(args.datadir, which_type = "val")
	target_X, target_y = target_data	

	top_n_misclf, indices_to_misclf, indices_to_corrclf = data_util.get_dataset_for_rq5(args.target_indices_file, top_n)	
	misclf_true, misclf_pred = top_n_misclf	
	
	new_test_indices = list(np.append(indices_to_misclf, indices_to_corrclf))
	target_data = (target_X[new_test_indices], target_y[new_test_indices])
	
	# to get the local indices of misclassified inputs
	indices = [new_test_indices.index(idx) for idx in indices_to_misclf]

print ("Processing: {}".format("{}-{}".format(misclf_true, misclf_pred)))

t1 = time.time()
patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
	num_label,
	target_data,
	target_layer_idx=args.target_layer_idx,
	max_search_num = iter_num, 
	search_method = 'DE',
	which = args.which,
	loc_method = args.loc_method,
	patch_target_key = "misclf-{}-{}".format(
		args.patch_key,"{}-{}".format(misclf_true,misclf_pred)),
	path_to_keras_model = args.path_to_keras_model,
	predef_indices_to_chgd = indices, 
	seed = args.seed,
	patch_aggr = args.patch_aggr,
	batch_size = args.batch_size,
	loc_dest = loc_dest,
	#loc_file = None,
	target_all = True)
		
t2 = time.time()
print ("Time for patching: {}".format(t2 - t1))
print ("patched_model_name", patched_model_name)	

os.rename(
	patched_model_name, 
	os.path.join(args.dest, os.path.basename(patched_model_name)))

