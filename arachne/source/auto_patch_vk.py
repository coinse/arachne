"""
Auto-patch module
	: contains functions related to patching NN model
"""
import utils.data_util as data_util
import time
import os
import tensorflow as tf

def patch(
	num_label,
	data,
	target_layer_idx = -1, # by default, target the last if target_all = False
	max_search_num = 200, 
	search_method = "DE",
	which = 'mnist',
	loc_method = "localiser",
	patch_target_key = None,
	path_to_keras_model = None,
	predef_indices_to_chgd = None,
	predef_indices_to_unchgd = None, 
	seed = 1,
	only_loc = False,
	patch_aggr = None, 
	target_all = False,
	loc_file = None,
	loc_dest = None,
	batch_size = None, 
	is_multi_label = True):
	"""
	only_loc = True:
		Ret(list, list):
			return (indices to localised nerual weights, full indices)
	only_loc = False (patch):
		Ret(str, list, list):
			return (a path to a generated patch, 
				indices to the target negative inputs, 
				indices to patched ones)
	"""
	import search.de_vk as de
	import numpy as np
	import random
	import run_localise
	from tensorflow.keras.models import load_model
	from collections.abc import Iterable

	random.seed(seed)
	np.random.seed(seed)

	if loc_dest is None:
		loc_dest = "."

	#################################################
	################# Prepare #######################
	#################################################

	data_X, data_y = data
	num_data = len(data_X)
	assert num_data == len(data_y), "%d vs %d" % (num_data, len(data_y))
	
	from collections import Iterable
	if not isinstance(data_y[0], Iterable) and (num_label > 2 or is_multi_label):
		from utils.data_util import format_label
		data_y = format_label(data_y, num_label)

	# index to target layer: e.g., 0 = the first hidden layer
	if not target_all:
		indices_to_target_layers = np.int32(target_layer_idx)
		from collections.abc import Iterable
		if not isinstance(indices_to_target_layers, Iterable):
			indices_to_target_layers = [indices_to_target_layers]
	else: # target all, but only those that statisfy the predefined layer conditions
		indices_to_target_layers = None
	
	run_localise.reset_keras([])
	
	model = load_model(path_to_keras_model, compile = False)
	target_weights = run_localise.get_target_weights(model,
		path_to_keras_model, 
		indices_to_target = indices_to_target_layers)

	#print ('Total {} layers are targeted'.format(target_weights.keys()))
	# compute prediction & corr_predictions
	predictions = model.predict(data_X)
	if len(predictions.shape) == 3:
		predictions = predictions.reshape((predictions.shape[0], predictions.shape[-1]))
	
	# first check whether this task is multi-class classification
	if len(predictions.shape) >= 2 and predictions.shape[-1] > 1: 
		correct_predictions = np.argmax(predictions, axis = 1)
		correct_predictions = correct_predictions == np.argmax(data_y, axis = 1)
	else:
		correct_predictions = np.round(predictions).flatten() == data_y

	if not only_loc:
		indices_to_target = data_util.split_into_wrong_and_correct(correct_predictions)
		#check whether given predef_indices_to_chgd to wrong is actually correct
		if predef_indices_to_chgd is not None:  # Since, here, we asssume an ideal model
			diff = set(predef_indices_to_chgd) - set(indices_to_target['wrong'])
			assert len(diff) == 0, diff 
		indices_to_target['wrong'] = predef_indices_to_chgd
	else: # only loc, so do not need to care about correct and wrong classification
		assert predef_indices_to_unchgd is not None, "For this, both the list of changed and unchanged should be given"
		indices_to_target = {'wrong':predef_indices_to_chgd, 'correct':predef_indices_to_unchgd}	

	indices_to_selected_wrong = indices_to_target['wrong'] # target all of them 
	#print ('Total number of wrongly processed input(s): {}'.format(len(indices_to_selected_wrong)))
	indices_to_correct = indices_to_target['correct']
	
	# extract the input vectors that are directly related to our target 
	new_indices_to_target = list(indices_to_correct) + list(indices_to_selected_wrong) 

	### sample inputs for localisation ###
	if only_loc:
		_, sampled_indices_to_correct = run_localise.sample_input_for_loc_by_rd(
			indices_to_selected_wrong, indices_to_correct)
	else:
		_, sampled_indices_to_correct = run_localise.sample_input_for_loc_by_rd(
			indices_to_selected_wrong, indices_to_correct, 
			predictions[new_indices_to_target], data_y[new_indices_to_target])

	# redefine new_indices_to_target (global indices)
	new_indices_to_target = list(indices_to_selected_wrong) + list(indices_to_correct)
	new_indices_for_loc = list(indices_to_selected_wrong) + list(sampled_indices_to_correct)
	####################################################################

	# target predictions
	predictions = predictions[new_indices_to_target] # slice

	# target data
	X = data_X[new_indices_to_target]; X_for_loc = data_X[new_indices_for_loc]
	y = data_y[new_indices_to_target]; y_for_loc = data_y[new_indices_for_loc]

	########### For logging & testing ############# 
	num_of_our_target = len(new_indices_to_target); num_of_our_loc_target = len(new_indices_for_loc)
	num_of_wrong = len(indices_to_selected_wrong)
	
	# set new local indices to correct & wrong for the new predictions
	indices_to_selected_wrong = list(range(0, num_of_wrong))
	indices_to_correct = list(range(num_of_wrong, num_of_our_target))
	indices_to_correct_for_loc = list(range(num_of_wrong, num_of_our_loc_target))

	assert_msg = "%d + %d vs %d" % (len(indices_to_correct), len(indices_to_selected_wrong), num_of_our_target)
	assert len(indices_to_correct) + len(indices_to_selected_wrong) == num_of_our_target, assert_msg
	assert len(X) == num_of_our_target, "%d vs %d" % (len(X), num_of_our_target)
	assert len(predictions) == num_of_our_target, "%d vs %d" % (len(predictions), num_of_our_target)
	########### logging and testing end ###########

	t1 = time.time()
	import pandas as pd

	if loc_method == 'gradient_loss': 
		if not only_loc: # for RQ2 
			if which == 'simple_fm':
				top_n = int(np.round(7.6))
			elif which == 'simple_cm':
				top_n = int(np.round(11.6))
			elif which == 'GTSRB': # GTSRB
				top_n = int(np.round(14.3))
			else: # lstm
				print ("Not yet")
				top_n = 14
		else:
			top_n = -1

		indices_w_costs = run_localise.localise_by_gradient(
				X_for_loc, y_for_loc,
				indices_to_selected_wrong,
				indices_to_correct_for_loc,
				target_weights,
				path_to_keras_model = path_to_keras_model,
				is_multi_label = is_multi_label)
		
		# retrieve only the indices
		indices_to_places_to_fix = [v[0] for v in indices_w_costs[:top_n]]
		loc_dest = os.path.join(loc_dest, "gl")
		os.makedirs(loc_dest, exist_ok=True)

		output_df = pd.DataFrame(
			{'layer':[vs[0] for vs in indices_to_places_to_fix], 
			'weight':[vs[1] for vs in indices_to_places_to_fix]}) 
		destfile = os.path.join(loc_dest, "loc.{}.{}.pkl".format(patch_target_key, int(target_all)))
		output_df.to_pickle(destfile)
		print ("Saved to", destfile)
		if only_loc:
			all_cost_file = os.path.join(loc_dest, 
				"loc.{}.{}.grad.all_cost.pkl".format(patch_target_key, int(target_all)))
			with open(all_cost_file, 'wb') as f:
				import pickle
				pickle.dump(indices_w_costs, f)	
	elif loc_method == 'localiser':
		if loc_file is None or not (os.path.exists(loc_file)):
			indices_to_places_to_fix, front_lst = run_localise.localise_by_chgd_unchgd(
				X_for_loc, y_for_loc,
				indices_to_selected_wrong,
				indices_to_correct_for_loc,
				target_weights,
				path_to_keras_model = path_to_keras_model, 
				is_multi_label = is_multi_label)
			#print ("Places to fix", indices_to_places_to_fix)
			output_df = pd.DataFrame(
				{'layer':[vs[0] for vs in indices_to_places_to_fix], 
				'weight':[vs[1] for vs in indices_to_places_to_fix]})

			loc_dest = os.path.join(loc_dest, "bl")
			os.makedirs(loc_dest, exist_ok= True)
			destfile = os.path.join(loc_dest, "loc.{}.{}.pkl".format(patch_target_key, int(target_all)))
			output_df.to_pickle(destfile)
			print ("Saved to", destfile)
			if only_loc:
				all_cost_file = os.path.join(loc_dest, 
					"loc.{}.{}.all_cost.pkl".format(patch_target_key, int(target_all)))
				with open(all_cost_file, 'wb') as f:
					import pickle
					pickle.dump(front_lst, f)
		else: # the localisation results already exist
			import pandas as pd
			df = pd.read_pickle(loc_file)
			indices_to_places_to_fix = df.values
	else: # randomly select
		if not only_loc:
			if which == 'simple_fm':
				top_n = int(np.round(7.6))
			elif which == 'simple_cm':
				top_n = int(np.round(11.6))
			elif which == 'GTSRB': 
				top_n = int(np.round(14.3))
			else: # for LSTM
				top_n = 14
			num_random_sample = top_n 
		else:
			num_random_sample = -1

		indices_to_places_to_fix = run_localise.localise_by_random_selection(
			num_random_sample, target_weights)	 

		loc_dest = os.path.join(loc_dest, "rd")	
		os.makedirs(loc_dest, exist_ok=True)
		output_df = pd.DataFrame({
			'layer':[vs[0] for vs in indices_to_places_to_fix], 
			'weight':[vs[1] for vs in indices_to_places_to_fix]}) 
		destfile = os.path.join(loc_dest, "loc.{}.{}.pkl".format(patch_target_key, int(target_all)))
		output_df.to_pickle(destfile)

	t2 = time.time()
	print ("Time taken for localisation: %f" % (t2 - t1))
	if not only_loc:
		print ("places to fix", indices_to_places_to_fix)	
		print ("numebr of places to fix: {}".format(len(indices_to_places_to_fix)))
	run_localise.reset_keras([model])
	#if loc_method == 'localiser':
	#	print (indices_to_places_to_fix) # logging
	
	if only_loc: # RQ1
		if loc_method in ['localiser', 'c_localiser']:
			return indices_to_places_to_fix, front_lst
		elif loc_method == 'gradient_loss':
			return indices_to_places_to_fix, indices_w_costs
		else:
			return indices_to_places_to_fix, None

	# reset seed and start searching
	random.seed(seed + 1) 
	np.random.seed(seed + 1) 
	indices_patched = []
	
	t1 = time.time()
	############################################################################
	################################ PATCH #####################################
	############################################################################
	# patch target layers
	indices_to_ptarget_layers = sorted(
		list(set([idx_to_tl if not isinstance(idx_to_tl, Iterable) else idx_to_tl[0] 
		for idx_to_tl,_ in indices_to_places_to_fix])))
	print ("Patch target layers", indices_to_ptarget_layers)

	if search_method == 'DE':
		searcher = de.DE_searcher(
			np.float32(X), np.float32(y),
			indices_to_correct, [],
			num_label,
			indices_to_ptarget_layers,
			mutation = (0.5, 1), 
			recombination = 0.7,
			max_search_num = max_search_num,
			initial_predictions = None, 
			path_to_keras_model = path_to_keras_model,
			patch_aggr = patch_aggr,
			batch_size = batch_size,
			act_func = tf.nn.relu if which == 'GTSRB' else None,
			is_multi_label = is_multi_label,
			is_lstm = 'lstm' in which)

		places_to_fix = indices_to_places_to_fix
		searcher.set_indices_to_wrong(indices_to_selected_wrong)	
		name_key = str(0) if patch_target_key is None else str(patch_target_key)
		_, saved_path = searcher.search(places_to_fix, name_key = name_key)
	else:
		print ("{} not supported yet".format(search_method))
		import sys; sys.exit()
			
	t2 = time.time()
	print ("Time taken for pure patching: %f" % (t2 - t1))

	return saved_path, indices_to_selected_wrong, indices_patched




	
			



