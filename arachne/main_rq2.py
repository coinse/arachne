"""
RQ2 script
"""
import argparse
import os, sys
import utils.data_util as data_util
import auto_patch_v3 as auto_patch
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
parser.add_argument("-loc_method", action = "store", default = 'gradient_loss')
parser.add_argument("-patch_target_key", action = "store", default = "best")
parser.add_argument("-path_to_keras_model", action = 'store', default = None)
parser.add_argument("-seed", action = "store", default = 1, type = int)
parser.add_argument("-iter_num", action = "store", default = 100, type = int)
parser.add_argument("-target_indices_file", action = 'store', default = None)
parser.add_argument("-dest", action = "store", default = ".")
parser.add_argument("-patch_aggr", action = 'store', default = None, type = int)

args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)

if args.target_indices_file is not None:
	import pandas as pd
	df = pd.read_csv(args.target_indices_file)
	indices = df['index'].values
	np.random.seed(args.seed)
else:
	predef_indices_to_wrong	= None

num_wrong_inputs_to_patch = int(len(indices) * 0.1) # for RQ2
predef_indices_to_wrong = np.random.choice(indices, num_wrong_inputs_to_patch, replace = False)
print ("predef", predef_indices_to_wrong, df.loc[df['index'] == predef_indices_to_wrong[0]].values[0])

total_train_data, test_data = data_util.load_data(args.which_data, args.datafile)

#num_of_sampled_correct = args.sampled_correct_num
num_train = len(total_train_data[1])
num_entire_misclfs = len(indices)
num_entire_corrclfs = num_train - num_entire_misclfs

train_data = total_train_data
train_X,train_y = train_data
test_X,test_y = test_data

num_label = 10
	
iter_num = args.iter_num

t1 = time.time()
patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
	num_label,
	train_data,
	args.tensor_name_file,
	max_search_num = iter_num, 
	search_method = 'DE',
	which = args.which,
	loc_method = args.loc_method, 
	patch_target_key = args.patch_target_key,
	path_to_keras_model = args.path_to_keras_model,
	predef_indices_to_wrong = predef_indices_to_wrong, 
	seed = args.seed, #)
	patch_aggr = args.patch_aggr) #True)


os.replace(patched_model_name.replace("None", "model") + ".json", 
	os.path.join(args.dest, patched_model_name.replace("None", "model") + ".json"))

t2 = time.time()
print ("So far...", indices_to_patched)
print ("Time for patching: %f" % (t2 - t1))
print ("patched_model_name", patched_model_name)

