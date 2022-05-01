"""
RQ2 script
"""
import argparse
import os
import utils.data_util as data_util
import auto_patch_vk as auto_patch
import time

parser = argparse.ArgumentParser()

parser.add_argument("-datadir", action = 'store', default = None)
parser.add_argument("-which", action = "store", 
	help = 'simple_cm, simple_fm', type = str)
parser.add_argument('-which_data', action = "store",
	default = 'cifar10', type = str, help = 'fashion_mnist,cifaf10,GTSRB')
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
parser.add_argument("-num_label", type = int, default = 10)
parser.add_argument("-batch_size", type = int, default = None)
parser.add_argument("-target_layer_idx", action = "store", 
	default = -1, type = int, help = "an index to the layer to localiser nws")

args = parser.parse_args()
os.makedirs(args.dest, exist_ok = True)
loc_dest = os.path.join(args.dest, "loc")
os.makedirs(loc_dest, exist_ok=True)

train_data, test_data = data_util.load_data(args.which_data, args.datadir)
predef_indices_to_wrong = data_util.get_misclf_for_rq2(
	args.target_indices_file, percent = 0.1, seed = args.seed)

num_label = args.num_label
iter_num = args.iter_num
t1 = time.time()

patched_model_name, indices_to_target_inputs, indices_to_patched = auto_patch.patch(
	num_label,
	test_data,
	target_layer_idx = args.target_layer_idx, 
	max_search_num = iter_num, 
	search_method = 'DE',
	which = args.which,
	loc_method = args.loc_method, 
	patch_target_key = args.patch_target_key,
	path_to_keras_model = args.path_to_keras_model,
	predef_indices_to_chgd = predef_indices_to_wrong,
	seed = args.seed, 
	patch_aggr = args.patch_aggr, 
	batch_size = args.batch_size,
	loc_dest = loc_dest,
	#loc_file = None,
	target_all = True)

os.replace(patched_model_name.replace("None", "model"), 
	os.path.join(args.dest, patched_model_name.replace("None", "model")))

t2 = time.time()
print ("Time for patching: %f" % (t2 - t1))
print ("patched_model_name", patched_model_name)

