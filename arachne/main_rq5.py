"""
RQ5 script
"""
import argparse
import os, sys
import utils.data_util as data_util
import auto_patch_vk as auto_patch
import time
import numpy as np
import gc
# temp
import run_localise

parser = argparse.ArgumentParser()

parser.add_argument("-datadir", action = "store", default = "data", type = str)
parser.add_argument("-which", action = "store", 
	help = 'cnn1, cnn2, cnn3, simple_cm, simple_fm, lfw_vgg', type = str)
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
parser.add_argument("-org_label", action = 'store', default = 3, type = int)
parser.add_argument("-pred_label", action = 'store', default = 5, type = int)
# temp
parser.add_argument("-new_loc", type = int, default = 0)
parser.add_argument("-loc_file", type = str, default = None)

args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)

train_data, test_data = data_util.load_data(args.which_data, args.datadir)
train_X,train_y = train_data
num_train = len(train_y)

test_X,test_y = test_data
test_X, test_y, indices_to_test = data_util.divide_into_val_and_test(test_X, test_y, 
	num_label = 10, is_val = True, half_n = int(len(test_y)/10/2))

test_data = [test_X, test_y]

index_file = os.path.join(args.dest, "{}.indices_to_val.csv").format(args.which)
with open(index_file, 'w') as f:
	f.write('index\n')
	for idx in indices_to_test:
		f.write(str(idx) + "\n")

print ("Record indices to valid in file {}".format(index_file))
num_test = len(test_y)
iter_num = args.iter_num
num_label = 10

# miclfds: key = (true label, predicted label), values: indices to the misclassified inputs 
misclfds = data_util.get_misclf_indices(args.target_indices_file, 
	target_indices = indices_to_test, 
	use_all = False) 


num_entire_misclfs = np.sum([len(vs) for vs in misclfds.values()])

sorted_keys = data_util.sort_keys_by_cnt(misclfds)
misclf_key = (args.org_label, args.pred_label)
indices = misclfds[misclf_key]

print ("Processing: {}".format("{}-{}".format(misclf_key[0],misclf_key[1])))
#num_of_sampled_correct = num_test - num_entire_misclfs
#print ("The number of correct samples: {}".format(num_of_sampled_correct))

t1 = time.time()

if bool(args.new_loc): # temp
	output, costs_and_keys = run_localise.localise_offline(
		num_label,
		test_data,
		args.tensor_name_file,
		path_to_keras_model = args.path_to_keras_model,
		predef_indices_to_wrong = indices,
		seed = args.seed,
		target_all = True)

	print ("The size of the pareto front: {}".format(len(output)))	
	print (output)
	
	import pandas as pd
	output_df = pd.DataFrame({'layer':[vs[0] for vs in output], 'weight':[vs[1] for vs in output]})
	dest = os.path.join(args.dest, "new_loc/{}_{}".format(args.org_label, args.pred_label))
	os.makedirs(dest, exist_ok= True)
	destfile = os.path.join(dest, "rq5.{}.pkl".format(args.which_data))
	print ("dest file: {}".format(destfile))
	output_df.to_pickle(destfile)
	
	import pickle
	with open(os.path.join(dest, "rq5.all_cost.{}.pkl".format(args.which_data)), 'wb') as f:
		pickle.dump(costs_and_keys, f)

	import sys; sys.exit()
else:
	# patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
	# 	num_label,
	# 	test_data, 
	# 	args.tensor_name_file,
	# 	max_search_num = iter_num, 
	# 	search_method = 'DE',
	# 	which = args.which,
	# 	loc_method = "localiser",
	# 	patch_target_key = "misclf-{}-{}".format(args.patch_key,"{}-{}".format(misclf_key[0],misclf_key[1])),
	# 	path_to_keras_model = args.path_to_keras_model,
	# 	predef_indices_to_wrong = indices,
	# 	seed = args.seed,
	# 	patch_aggr = args.patch_aggr)
	patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
		num_label,
		test_data, 
		args.tensor_name_file,
		max_search_num = iter_num, 
		search_method = "DE",
		which = args.which,
		loc_method = "localiser",
		patch_target_key = "misclf-{}-{}".format(args.patch_key,"{}-{}".format(misclf_key[0],misclf_key[1])),
		path_to_keras_model = args.path_to_keras_model,
		predef_indices_to_wrong = indices,
		seed = args.seed,
		patch_aggr = args.patch_aggr, 
		target_all = False,
		loc_file = args.loc_file)
		
t2 = time.time()
print ("Time for patching: {}".format(t2 - t1))
print ("patched_model_name", patched_model_name)	

os.rename(patched_model_name + ".json", os.path.join(args.dest, os.path.basename(patched_model_name) + ".json"))

gc.collect()


