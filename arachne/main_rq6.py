"""
RQ6 script
"""
import argparse
import os, sys
import utils.data_util as data_util
import auto_patch_vk as auto_patch
import time
import numpy as np
import gc

TOP_N = 1

parser = argparse.ArgumentParser()

parser.add_argument("-datadir", action = "store", default = "data", type = str)
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
parser.add_argument("-female_lst_file", action = 'store', 
	default = 'data/lfw_np/female_names_lfw.txt', type = str)
args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)

which = 'lfw_vgg'
which_data = 'lfw'
train_data, test_data = data_util.load_data(which_data, args.datadir, 
	path_to_female_names = args.female_lst_file)

train_X,train_y = train_data
num_train = len(train_y)

test_X,test_y = test_data
test_X = np.asarray(test_X[::2])
test_y = np.asarray(test_y[::2])
test_data = [test_X, test_y]

num_test = len(test_y)

iter_num = args.iter_num
num_label = 2

# miclfds: key = (true label, predicted label), values: indices to the misclassified inputs 
misclfds = data_util.get_misclf_indices(args.target_indices_file, 
	target_indices = None, 
	use_all = False) 

num_entire_misclfs = np.sum([len(vs) for vs in misclfds.values()])

sorted_keys = data_util.sort_keys_by_cnt(misclfds)
misclf_key = sorted_keys[TOP_N-1]

indices = misclfds[misclf_key]
indices = [int(i/2) for i in indices]

print ("Processing: {}".format("{}-{}".format(misclf_key[0],misclf_key[1])))
#num_of_sampled_correct = num_test - num_entire_misclfs
#print ("The number of correct samples: {}".format(num_of_sampled_correct))
#num_wrong_inputs_to_patch = len(indices)
#print ('pre_defined', num_wrong_inputs_to_patch)	

t1 = time.time()
patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
	num_label,
	test_data, 
	args.tensor_name_file,
	max_search_num = iter_num, 
	search_method = 'DE',
	which = which,
	loc_method = "localiser",
	patch_target_key = "misclf-{}-{}".format(args.patch_key,"{}-{}".format(misclf_key[0],misclf_key[1])),
	path_to_keras_model = args.path_to_keras_model,
	predef_indices_to_wrong = indices,
	seed = args.seed,
	patch_aggr = args.patch_aggr)
		
t2 = time.time()
print ("Time for patching: {}".format(t2 - t1))
print ("patched_model_name", patched_model_name)	

os.rename(patched_model_name + ".json", os.path.join(args.dest, os.path.basename(patched_model_name) + ".json"))

gc.collect()


