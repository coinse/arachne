"""
RQ1 script
"""
import os, sys
import pandas as pd
import utils.data_util as data_util
import auto_patch
from main_eval import read_and_add_flag, combine_init_aft_predcs
import json
import numpy as np
##
import run_localise

def get_brokens(combined_df):
	"""
	"""
	brokens = combined_df.loc[\
		(combined_df.pred != combined_df.new_pred) & (combined_df.init_flag == True)]

	return brokens

def get_patcheds(combined_df):
	"""
	"""
	patcheds = combined_df.loc[\
		(combined_df.pred != combined_df.new_pred) & (combined_df.init_flag == False)]

	return patcheds

def set_loc_name(dest, aft_pred_file, key):
	"""
	"""
	basename = os.path.basename(aft_pred_file).replace("indices.csv", "loc.{}.json".format(key))
	basename = basename.replace(".init_pred.", ".")
	loc_name = os.path.join(dest, basename)
	
	return loc_name


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-init_pred_file", type = str, help = "original")
	parser.add_argument("-aft_pred_file", type = str, help = "with noise")
	parser.add_argument("-num_label", type = int, default = 10)
	parser.add_argument("-datadir", action = "store", default = "data", type = str)
	parser.add_argument("-which", action = "store", 
		help = 'simple_cm, simple_fm', type = str)
	parser.add_argument('-which_data', action = "store",
		default = 'cifar10', type = str, help = 'fashion_mnist,cifaf10,lfw')
	parser.add_argument("-tensor_name_file", action = "store",
		default = "data/tensor_names/tensor.lastLayer.names ", type = str)
	parser.add_argument("-loc_method", action = "store", default = None, help = 'random, localiser, gradient_loss')
	parser.add_argument("-path_to_keras_model", action = 'store', default = None)
	parser.add_argument("-seed", action = "store", default = 1, type = int)
	parser.add_argument("-dest", default = ".", type = str)
	# temporary for localising over all
	parser.add_argument("-new_loc", type = int, default = 0)

	args = parser.parse_args()	

	init_pred_df = read_and_add_flag(args.init_pred_file)
	aft_pred_df = read_and_add_flag(args.aft_pred_file)
	combined_df = combine_init_aft_predcs(init_pred_df, aft_pred_df)

	brokens = get_brokens(combined_df)
	patcheds = get_patcheds(combined_df)

	indices_to_wrong = brokens.index.values

	# is_input_2d = True => to match the format with faulty model
	train_data, test_data = data_util.load_data(args.which_data, args.datadir, is_input_2d = True)
	train_X, train_y = train_data
	num_train = len(train_y)
	test_X, test_y = test_data

	dest = args.dest
	os.makedirs(dest, exist_ok = True)
	path_to_loc_file = set_loc_name(dest, args.aft_pred_file, args.seed)

	if bool(args.new_loc): # temp
		output = run_localise.localise_offline(
			args.num_label,
			train_data,
			args.tensor_name_file,
			path_to_keras_model = args.path_to_keras_model,
			predef_indices_to_wrong = indices_to_wrong,
			seed = args.seed,
			target_all = True)
			
		#import time
		#print ("start sleeping")
		#time.sleep(10)
		#import sys; sys.exit()
		print ("The size of the pareto front: {}".format(len(output)))	
		print (output)
	
		import pandas as pd
		output_df = pd.DataFrame({'layer':[vs[0] for vs in output], 'weight':[vs[1] for vs in output]})

		dest = os.path.join(args.dest, "new_loc")
		os.makedirs(dest, exist_ok= True)
		destfile = os.path.join(dest, "rq1.{}.pkl".format(args.which_data))
		output_df.to_pickle(destfile)
		import sys; sys.exit()
	else:
		indices_to_places_to_fix, front_lst = auto_patch.patch(
			args.num_label,
			train_data,
			args.tensor_name_file,
			which = args.which,
			loc_method = args.loc_method,
			path_to_keras_model = args.path_to_keras_model,
			predef_indices_to_wrong = indices_to_wrong,
			seed = args.seed,
			only_loc = True)	
	
	print ("Localised nerual weights({}):".format(len(indices_to_places_to_fix)))
	print ("\t".join([str(index) for index in indices_to_places_to_fix]))
	if args.loc_method == 'localiser':
		indices_to_places_to_fix = np.int32(indices_to_places_to_fix).tolist()
		front_lst = [np.int32(sub_front).tolist() for sub_front in front_lst]

		with open(path_to_loc_file, 'w') as f:
			f.write(json.dumps({"front_0":indices_to_places_to_fix, "fronts":front_lst}))

	elif args.loc_method == 'gradient_loss':
		sorted_indices_and_grads = sorted(front_lst, key = lambda v: v[1], reverse = True)
		final_results = {'weights':[(int(ridx),int(cidx)) for (ridx,cidx),_ in sorted_indices_and_grads],
			'grads':[float(grad) for _,grad in sorted_indices_and_grads]}

		with open(path_to_loc_file, 'w') as f:
			f.write(json.dumps(final_results))
	else: 
		pass
