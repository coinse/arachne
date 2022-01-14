"""
RQ3 script
"""
import argparse
import os, sys
import utils.data_util as data_util
import auto_patch_vk as auto_patch
import time
import numpy as np
import gc
from collections.abc import Iterable 

parser = argparse.ArgumentParser()

parser.add_argument("-datadir", action = "store", default = "data", type = str)
parser.add_argument("-which", action = "store", 
	help = 'simple_cm, simple_fm', type = str)
parser.add_argument('-which_data', action = "store",
	default = 'cifar10', type = str, help = 'fashion_mnist,cifaf10,lfw')
parser.add_argument("-tensor_name_file", action = "store",
	default = "data/tensor_names/tensor.lastLayer.names ", type = str)
parser.add_argument("-patch_key", action = "store", default = "key")
parser.add_argument("-path_to_keras_model", action = 'store', default = None)
parser.add_argument("-seed", action = "store", default = 1, type = int)
parser.add_argument("-iter_num", action = "store", default = 100, type = int)
parser.add_argument("-target_indices_file", action = "store", default = None)
parser.add_argument("-dest", default = ".", type = str)
parser.add_argument("-patch_aggr", action = 'store', default = None, type = float)
parser.add_argument("-w_hist", type = int, default = 0)
parser.add_argument("-num_label", type = int, default = 10)
parser.add_argument("-batch_size", type = int, default = None)
parser.add_argument("-on_train", action = "store_true", help = "if given, then evaluate on the training data")
parser.add_argument("-top_n", type = int, default = None)
parser.add_argument("-female_lst_file", action = 'store', 
	default = 'data/lfw_np/female_names_lfw.txt', type = str)
parser.add_argument("-loc_method", action = "store", default = 'localiser')

args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)

train_data, test_data = data_util.load_data(args.which_data, args.datadir, with_hist = bool(args.w_hist), path_to_female_names = args.female_lst_file)
target_data = test_data if not args.on_train else train_data
target_X, target_y = target_data

iter_num = args.iter_num
num_label = args.num_label 
if args.top_n is None:
	top_n = args.seed # to target a unique type of misbehaviour per run
else: # mainly for RQ4
	top_n = args.top_n

outs = data_util.get_balanced_dataset(args.target_indices_file, top_n, idx = 0)
if not isinstance(outs, Iterable):
	print ("There are only {} number of unqiue misclassification types. vs {}".format(outs, top_n))
	sys.exit()
else:
	(misclf_key, abs_indices, new_test_indices, _) = outs
	target_data = (target_X[new_test_indices], target_y[new_test_indices]) 
	indices = [new_test_indices.index(idx) for idx in abs_indices]

print ("Processing: {}".format("{}-{}".format(misclf_key[0], misclf_key[1])))
#num_of_sampled_correct = num_test - num_entire_misclfs
#print ("The number of correct samples: {}".format(num_of_sampled_correct))
num_wrong_inputs_to_patch = len(indices)
print ('pre_defined', num_wrong_inputs_to_patch)	
print (indices)
t1 = time.time()
print (len(abs_indices))

patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
	num_label,
	target_data,
	args.tensor_name_file,
	max_search_num = iter_num, 
	search_method = 'DE',
	which = args.which,
	loc_method = args.loc_method,
	patch_target_key = "misclf-{}-{}".format(args.patch_key,"{}-{}".format(misclf_key[0],misclf_key[1])),
	path_to_keras_model = args.path_to_keras_model,
	predef_indices_to_chgd = indices, 
	seed = args.seed,
	patch_aggr = args.patch_aggr,
	batch_size = args.batch_size,
	is_multi_label = True, #True if args.which != 'lstm' else False,  
	#loc_dest = "new_loc/rq6/all/use_both/pa2",
	loc_dest = "new_loc/lstm/us_airline/top_{}/pa{}".format(top_n, args.patch_aggr),
	#loc_dest = "new_loc/lstm/us_airline/only_the_last", #"new_loc/rq6/all/use_both/pa3", #,"new_loc/lstm/us_airline", # "rq2_loc_reuter/only_last", #"new_loc/rq6/all/use_both",
	target_all = True)
		
t2 = time.time()
print ("Time for patching: {}".format(t2 - t1))
print ("patched_model_name", patched_model_name)	

os.rename(patched_model_name, os.path.join(args.dest, os.path.basename(patched_model_name)))

gc.collect()


