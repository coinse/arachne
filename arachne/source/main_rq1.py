"""
RQ1 script
"""
import os, sys
import pandas as pd
import utils.data_util as data_util
import auto_patch_vk as auto_patch
from utils.eval_util import read_and_add_flag, combine_init_aft_predcs
import numpy as np

LAYER = -1 # all layers
Faulty_mdl_path = "../data/models/faulty_models/by_tweak/chg/0_001/mv" # final_data/models/rq1_faulty_mdl/

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
		target_mdl_path_fm = os.path.join(Faulty_mdl_path, '{}/{}/{}_simple_90p_seed{}.h5')
		target_mdl_path = target_mdl_path_fm.format(which_data, num_sample, which_key, fault_id)
	elif which_data == 'fashion_mnist':
		target_mdl_path_fm = os.path.join(Faulty_mdl_path, '{}/{}/{}_simple_seed{}.h5')
		target_mdl_path = target_mdl_path_fm.format(which_data, num_sample, which_key, fault_id)
	elif which_data == 'GTSRB':
		target_mdl_path_fm = os.path.join(Faulty_mdl_path, '{}/{}/{}.model.0.wh.0_seed{}.h5')
		target_mdl_path = target_mdl_path_fm.format(which_data, num_sample, which_key, fault_id)
	else:
		print ("{} not supported".format(which_data))
		sys.exit()

	target_gt_path_fm = os.path.join(Faulty_mdl_path, '{}/{}/faulty_nws.{}.pkl')
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


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-init_pred_file", type = str, help = "original")
	parser.add_argument("-aft_pred_file", type = str, help = "with noise. for just running rq1, not needed")
	parser.add_argument("-num_label", type = int, default = 10)
	parser.add_argument("-datadir", action = "store", default = "data", type = str)
	parser.add_argument("-which", action = "store", help = 'simple_cm, simple_fm', type = str)
	parser.add_argument('-which_data', action = "store", default = 'cifar10', type = str, 
		help = 'fashion_mnist,cifaf10,lfw')
	parser.add_argument("-loc_method", action = "store", default = None, 
	help = 'random, localiser, gradient_loss, c_localiser')
	parser.add_argument("-seed", action = "store", default = 1, type = int)
	parser.add_argument("-dest", default = ".", type = str)
	parser.add_argument("-target_all", type = int, default = 1)
	parser.add_argument("-target_layer_idx", action = "store", default = -1, type = int)
	parser.add_argument("-on_test", action = "store_true", 
		help = 'if given, localise based on the behaviour difference on the test data')
	parser.add_argument("-fid_file", type = str, 
		help = "a file that contains the ids (used seeds) of target faulty model", default = None)

	args = parser.parse_args()	

	path_to_faulty_model, gt_file = return_target_mdl_and_gt_path(args.fid_file, args.seed, args.which_data)
	if path_to_faulty_model is None:
		print ('Seed {} not our target for layer {}'.format(args.seed, LAYER))
		sys.exit()

	# is_input_2d = True => to match the format with faulty model
	train_data, test_data = data_util.load_data(args.which_data, args.datadir)
	train_X, train_y = train_data
	num_train = len(train_y)
	test_X, test_y = test_data
	# set X and y for the localisation 
	X,y = train_data if not args.on_test else test_data

	init_pred_df = read_and_add_flag(args.init_pred_file)
	init_acc = np.sum(init_pred_df.true == init_pred_df.pred)/len(init_pred_df)

	if args.aft_pred_file is None:
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

	brokens = get_brokens(combined_df) # correct -> incorrect
	patcheds = get_patcheds(combined_df) # incorrect -> correct

	chgds = combined_df.loc[combined_df.pred != combined_df.new_pred] # all
	unchgds = combined_df.loc[combined_df.pred == combined_df.new_pred]
	
	assert len(brokens) + len(patcheds) == len(chgds), "{} vs {}".format(len(brokens) + len(patcheds), len(chgds))
	#print ("chgds: {}, unchgds:{}".format(len(chgds), len(unchgds)))
	indices_to_chgd = chgds.index.values
	#print ("Indices to changed", indices_to_chgd, len(indices_to_chgd))
	indices_to_unchgd = unchgds.index.values
	indices_to_places_to_fix, entire_k_and_cost = auto_patch.patch(
		args.num_label,
		(X,y),
		target_layer_idx=args.target_layer_idx,
		which = args.which,
		loc_method = args.loc_method, 
		patch_target_key = "loc.{}".format(args.seed),
		path_to_keras_model = path_to_faulty_model, 
		predef_indices_to_chgd = indices_to_chgd, 
		predef_indices_to_unchgd = indices_to_unchgd,
		seed = args.seed,
		target_all = bool(args.target_all),
		only_loc = True,
		loc_dest = "results/rq1")

	# for evaluation 
	gt_df = pd.read_pickle(gt_file)
	gts_layer = gt_df.layer.values
	gts_weight = gt_df.w_idx.values
	gts = list(zip(gts_layer, gts_weight)) # a list of [layer, index to a weight (np.ndarray)]

	localised_at = []
	
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
					print("Include", i, l_idx, w_idx, _indices_to_w)
	
	#if args.loc_method == 'localiser':
	#	print ("Localised within the pareto front of the length of {}".format(len(indices_to_places_to_fix)))
	#print ("\tAt", localised_at)
	#print ('\t', [idx/len(entire_k_and_cost) 
	#	if entire_k_and_cost is not None else idx/len(indices_to_places_to_fix) for idx in localised_at])

