"""
RQ2 script
"""
import argparse
import os, sys
import utils.data_util as data_util
import auto_patch_vk as auto_patch
import time
import numpy as np
import gc


parser = argparse.ArgumentParser()

parser.add_argument("-datadir", action = 'store', default = None)
parser.add_argument("-which", action = "store", 
	help = 'simple_cm, simple_fm', type = str)
parser.add_argument('-which_data', action = "store",
	default = 'cifar10', type = str, help = 'fashion_mnist,cifaf10,lfw')
parser.add_argument("-tensor_name_file", action = "store",
	default = "data/tensor_names/tensor.lastLayer.names ", type = str)
parser.add_argument("-loc_method", action = "store", default = 'localiser')
parser.add_argument("-patch_target_key", action = "store", default = "best")
parser.add_argument("-path_to_keras_model", action = 'store', default = None)
parser.add_argument("-seed", action = "store", default = 1, type = int)
parser.add_argument("-iter_num", action = "store", default = 100, type = int)
parser.add_argument("-target_indices_file", action = 'store', default = None)
parser.add_argument("-dest", action = "store", default = ".")
parser.add_argument("-patch_aggr", action = 'store', default = None, type = int)
parser.add_argument("-w_hist", type = int, default = 0)
parser.add_argument("-num_label", type = int, default = 10)
parser.add_argument("-batch_size", type = int, default = None)
parser.add_argument("-with_val", action = "store_true", 
	help = "if given, we will divide the test data to the half and half")

args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)

## get the indices to incorrect inputs, which will be considered as changed inputs 
#if args.target_indices_file is not None:
	#import pandas as pd
	#df = pd.read_csv(args.target_indices_file)
	#indices = df['index'].values
	#np.random.seed(args.seed)
#else:
	#predef_indices_to_wrong	= None
#
#num_wrong_inputs_to_patch = int(len(indices) * 0.1) # for RQ2
#if num_wrong_inputs_to_patch > 10: # just random number, actually, I already know that there are more than ten;; but not for GTSRB
	#predef_indices_to_wrong = np.random.choice(indices, num_wrong_inputs_to_patch, replace = False)
#else:# GTSRB... we may generate more simple model for GTSRB (-> we might have to look why the trainig for GTSRB works so well)
	## maybe, we can compare the variations of input images with the same label and check whether the similarity or the difference to
	## the other images are greater in GTSRB. If it is, it explains why the model trained for GTSBR achieve so high accuracy
	#predef_indices_to_wrong = indices

train_data, test_data = data_util.load_data(args.which_data, args.datadir, with_hist = bool(args.w_hist))

if not args.with_val:
	predef_indices_to_wrong = data_util.get_misclf_for_rq2(args.target_indices_file, percent = 0.1, seed = args.seed)
else:
	# here, args.target_indices_file should be the complete prediction file
	abs_misclf_indices, target_indices, _ = data_util.get_test_val_dataset_by_perc(args.target_indices_file, percent = 0.1, seed = args.seed)
	test_data = (test_data[0][target_indices], test_data[1][target_indices])
	predef_indices_to_wrong = [np.where(target_indices == idx)[0][0] for idx in abs_misclf_indices]
	print ("in!")
#train_data, test_data = data_util.load_data(args.which_data, args.datadir, with_hist = bool(args.w_hist))

#num_train = len(train_data[1])
#num_entire_misclfs = len(indices)
#num_entire_corrclfs = num_train - num_entire_misclfs
#train_X,train_y = train_data
#test_X,test_y = test_data
#test_X = test_X[:100]
#test_y = test_y[:100]

num_label = args.num_label
iter_num = args.iter_num
t1 = time.time()
patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
	num_label,
	test_data, #train_data,
	args.tensor_name_file,
	max_search_num = iter_num, 
	search_method = 'DE',
	which = args.which,
	loc_method = args.loc_method, 
	patch_target_key = args.patch_target_key,
	path_to_keras_model = args.path_to_keras_model,
	predef_indices_to_chgd = predef_indices_to_wrong,
	seed = args.seed, 
	patch_aggr = args.patch_aggr, #) #True)
	batch_size = args.batch_size,
	is_multi_label = True, # if (args.which != 'lstm' or 'multi' in args.path_to_keras_model) else False, 
	target_all = False, #True, 
	loc_dest = "rq2_loc_reuter")

os.replace(patched_model_name.replace("None", "model"), 
	os.path.join(args.dest, patched_model_name.replace("None", "model")))

t2 = time.time()
print ("Time for patching: %f" % (t2 - t1))
print ("patched_model_name", patched_model_name)

