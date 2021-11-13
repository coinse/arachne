"""
RQ1 script
"""
import os, sys
import pandas as pd
import utils.data_util as data_util
import auto_patch_vk as auto_patch
from main_eval import read_and_add_flag, combine_init_aft_predcs
import json
import numpy as np
##
import run_localise

LAYER = -1

def return_target_fault_id(afile, seed):
	if afile is None:
		return seed
	else:
		df = pd.read_csv(afile)
		if LAYER < 0 or df.iloc[seed].layer == LAYER:
			return df.iloc[seed].id	
		else:
			return None

def return_target_mdl_and_gt_path(afile, seed, which_data):
	"""
	"""
	which_keys = {'cifar10':'cifar', 'fashion_mnist':'fmnist', 'GTSRB':'gtsrb'}
	num_sample = 1
	fault_id = return_target_fault_id(afile, seed)
	if fault_id is None:
		return None, None
	which_key = which_keys[which_data]

	if which_data == 'cifar10':
		target_mdl_path_fm = 'data/models/faulty_models/by_tweak/chg/0_001/mv/{}/{}/{}_simple_90p_seed{}.h5'	
		target_mdl_path = target_mdl_path_fm.format(which_data, num_sample, which_key, fault_id)
	#else: #fmnist_simple_seed0.h5
	elif which_data == 'fashion_mnist':
		target_mdl_path_fm = 'data/models/faulty_models/by_tweak/chg/0_001/mv/{}/{}/{}_simple_seed{}.h5'
		target_mdl_path = target_mdl_path_fm.format(which_data, num_sample, which_key, fault_id)
	elif which_data == 'GTSRB':#data/models/faulty_models/by_tweak/chg/0_001/mv/GTSRB/1/gtsrb.model.0.wh.0_seed20.h5 
		target_mdl_path_fm = 'data/models/faulty_models/by_tweak/chg/0_001/mv/{}/{}/{}.model.0.wh.0_seed{}.h5'
		target_mdl_path = target_mdl_path_fm.format(which_data, num_sample, which_key, fault_id)
	else:
		print ("{} not supported".format(which_data))
		sys.exit()
	target_gt_path_fm = 'data/models/faulty_models/by_tweak/chg/0_001/mv/{}/{}/faulty_nws.{}.pkl'
	target_gt_path = target_gt_path_fm.format(which_data, num_sample, fault_id)

	return target_mdl_path, target_gt_path

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
	parser.add_argument("-loc_method", action = "store", default = None, help = 'random, localiser, gradient_loss, c_localiser')
	#parser.add_argument("-path_to_keras_model", action = 'store', default = None)
	#parser.add_argument("-path_to_faulty_model", action = 'store', default = None, type = str)
	parser.add_argument("-seed", action = "store", default = 1, type = int)
	parser.add_argument("-dest", default = ".", type = str)
	# temporary for localising over all
	parser.add_argument("-new_loc", type = int, default = 0)
	parser.add_argument("-target_all", type = int, default = 1)
	parser.add_argument("-w_hist", type = int, default = 0)
	parser.add_argument("-on_test", action = "store_true", 
		help = 'if given, localise based on the behaviour difference on the test data')
	# 
	#parser.add_argument("-gt_file", type = str, default = None)
	# to retrieve those 
	parser.add_argument("-fid_file", type = str, 
		help = "a file that contains the ids (used seeds) of target faulty model", default = None)

	args = parser.parse_args()	

	path_to_faulty_model, gt_file = return_target_mdl_and_gt_path(args.fid_file, args.seed, args.which_data)
	if path_to_faulty_model is None:
		print ('Seed {} not our target for layer {}'.format(args.seed, LAYER))
		sys.exit()
	print (path_to_faulty_model, gt_file)

	# is_input_2d = True => to match the format with faulty model
	train_data, test_data = data_util.load_data(args.which_data, args.datadir, with_hist = bool(args.w_hist))
	train_X, train_y = train_data
	num_train = len(train_y)
	test_X, test_y = test_data
	# set X and y for the localisation 
	X,y = train_data if not args.on_test else test_data

	init_pred_df = read_and_add_flag(args.init_pred_file)
	init_acc = np.sum(init_pred_df.true == init_pred_df.pred)/len(init_pred_df)

	if args.aft_pred_file is None:
		#assert path_to_faulty_model is not None, "Neither aft_pred_file nor path_to_faulty_model is given"
		from tensorflow.keras.models import load_model

		faulty_mdl = load_model(path_to_faulty_model, compile = False)
		predcs = faulty_mdl.predict(X) if args.which_data != 'fashion_mnist' else faulty_mdl.predict(X).reshape(len(X),-1)
		pred_labels = np.argmax(predcs, axis = 1)
		
		corr_predictions = pred_labels == y	
		acc = np.sum(corr_predictions)/corr_predictions.shape[0]
		print ("Acc: {} -> {}".format(init_acc, acc))

		aft_preds = []
		aft_preds_column = ['index', 'true', 'pred', 'flag']
		for i, (true_label, pred_label) in enumerate(zip(y, pred_labels)):
			aft_preds.append([i, true_label, pred_label, true_label == pred_label])
		aft_pred_df = pd.DataFrame(aft_preds, columns = aft_preds_column)
	else:
		aft_pred_df = read_and_add_flag(args.aft_pred_file)
	
	combined_df = combine_init_aft_predcs(init_pred_df, aft_pred_df)

	## this should be changed... 
	## both brokens and patched denote the changes in a target DNN model
	brokens = get_brokens(combined_df) # correct -> incorrect
	patcheds = get_patcheds(combined_df) # incorrect -> correct'
	chgds = combined_df.loc[combined_df.pred != combined_df.new_pred] # all
	unchgds = combined_df.loc[combined_df.pred == combined_df.new_pred]
	assert len(brokens) + len(patcheds) == len(chgds), "{} vs {}".format(len(brokens) + len(patcheds), len(chgds))
	print ("chgds: {}, unchgds:{}".format(len(chgds), len(unchgds)))
	sys.exit()	
	# currently, we are using only the ones that are broken
	#indices_to_wrong = brokens.index.values
	#print ("Indices to wrong", indices_to_wrong)
	indices_to_chgd = chgds.index.values
	print ("Indices to changed", indices_to_chgd, len(indices_to_chgd))
	indices_to_unchgd = unchgds.index.values
	#dest = args.dest
	#os.makedirs(dest, exist_ok = True)
	#path_to_loc_file = set_loc_name(dest, args.aft_pred_file, args.seed)

	if bool(args.new_loc): # temp
		output = run_localise.localise_offline(
			args.num_label,
			train_data if not args.on_test else test_data,
			args.tensor_name_file,
			path_to_keras_model = path_to_faulty_model, #args.path_to_keras_model,
			predef_indices_to_wrong = indices_to_chgd, #indices_to_wrong,
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
		indices_to_places_to_fix, entire_k_and_cost = auto_patch.patch(
			args.num_label,
			train_data if not args.on_test else test_data,
			args.tensor_name_file,
			which = args.which,
			loc_method = args.loc_method, 
			patch_target_key = "loc.{}".format(args.seed),
			path_to_keras_model = path_to_faulty_model, #args.path_to_keras_model,
			predef_indices_to_chgd = indices_to_chgd, #indices_to_wrong,
			predef_indices_to_unchgd = indices_to_unchgd,
			seed = args.seed,
			target_all = bool(args.target_all),
			only_loc = True)

	print ("Localised nerual weights({}):".format(len(indices_to_places_to_fix)))

	#if args.gt_file is not None:
	if True:
		gt_df = pd.read_pickle(gt_file)
		gts_layer = gt_df.layer.values
		gts_weight = gt_df.w_idx.values
		gts = list(zip(gts_layer, gts_weight)) # a list of [layer, index to a weight (np.ndarray)]

		localised_at = []
		print ("Selected")
	
		layers_from_loc = np.asarray([vs[0] for vs in indices_to_places_to_fix])
		layers_from_gt = gts_layer
		intersected = np.intersect1d(layers_from_loc, layers_from_gt)
		if len(intersected) > 0:
			for i, (l_idx, w_idx) in enumerate(indices_to_places_to_fix):
				if l_idx in intersected:
					_indices_to_l = np.where(gts_layer == l_idx)[0]
					_indices_to_w = np.where(list(map(np.array_equal, gts_weight, len(gts_weight)*[np.asarray(w_idx)])))[0]
					indices_to_common = np.intersect1d(_indices_to_l, _indices_to_w)
					if len(indices_to_common) > 0:
						localised_at.append(i)
						print("Inclu", i, l_idx, w_idx, _indices_to_w)

		if args.loc_method == 'localiser':
			print ("Localised within the pareto front of the length of {}".format(len(indices_to_places_to_fix)))
		
		print ("\tAt", localised_at)
		print ('\t', [idx/len(entire_k_and_cost) if entire_k_and_cost is not None else idx/len(indices_to_places_to_fix) for idx in localised_at])


	#print ("\t".join([str(index) for index in indices_to_places_to_fix]))
	#if args.loc_method == 'localiser':
	#	indices_to_places_to_fix = np.int32(indices_to_places_to_fix).tolist()
	#	front_lst = [np.int32(sub_front).tolist() for sub_front in front_lst]
	#
	#	with open(path_to_loc_file, 'w') as f:
	#		f.write(json.dumps({"front_0":indices_to_places_to_fix, "fronts":front_lst}))

	# if args.loc_method == 'gradient_loss':
	# 	sorted_indices_and_grads = sorted(front_lst, key = lambda v: v[1], reverse = True)
	# 	final_results = {'weights':[(int(ridx),int(cidx)) for (ridx,cidx),_ in sorted_indices_and_grads],
	# 		'grads':[float(grad) for _, grad in sorted_indices_and_grads]}


	# 	with open(path_to_loc_file, 'w') as f:
	# 		f.write(json.dumps(final_results))
	# else: 
	# 	pass
