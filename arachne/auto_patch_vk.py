"""
Auto-patch module
	: contains functions related to patching NN model
"""
import utils.data_util as data_util
import time
import os
from gen_frame_graph import generate_empty_graph
import tensorflow as tf

def patch(
	num_label,
	data,
	tensor_name_file,
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
	loss_funcs = None):
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
	from tensorflow.keras.models import load_model, Model
	import subprocess
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
	if not isinstance(data_y[0], Iterable):
		from utils.data_util import format_label
		data_y = format_label(data_y, num_label)

	# index to target layer: e.g., 0 = the first hidden layer
	if not target_all:
		indices_to_target_layers = np.int32(data_util.read_tensor_name(tensor_name_file)['t_layer'])
		from collections.abc import Iterable
		if not isinstance(indices_to_target_layers, Iterable):
			indices_to_target_layers = [indices_to_target_layers]
	else: # target all, but only those that statisfy the predefined layer conditions
		indices_to_target_layers = None
	
	run_localise.reset_keras([])
	
	model = load_model(path_to_keras_model, compile = False)
	result = subprocess.run(['nvidia-smi'], shell = True) #stdout = subprocess.PIPE.stdout, stderr = subprocess.PIPE.stderr)
	target_weights = run_localise.get_target_weights(model,
		path_to_keras_model, 
		indices_to_target = indices_to_target_layers, 
		target_all = target_all) # if target_all == True, then indices_to_target will be ignored

	result = subprocess.run(['nvidia-smi'], shell = True) #stdout = subprocess.PIPE.stdout, stderr = subprocess.PIPE.stderr)
	print (result)
	print ('Total {} layers are targeted'.format(target_weights.keys()))

	# compute prediction & corr_predictions
	predictions = model.predict(data_X)
	if len(predictions.shape) == 3:
		predictions = predictions.reshape((predictions.shape[0], predictions.shape[-1]))

	correct_predictions = np.argmax(predictions, axis = 1)
	correct_predictions = correct_predictions == np.argmax(data_y, axis = 1)
	print ("The predictions", correct_predictions.shape)
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
	print ('Total number of wrongly processed input(s): {}'.format(len(indices_to_selected_wrong)))
	indices_to_correct = indices_to_target['correct']

	# extract the input vectors that are directly related to our target 
	# correct one first, followed by misclassified ones
	# FOR LFW, THIS WILL BE USED TO SLICE THE PRE-COMPUTE ATS
	new_indices_to_target = list(indices_to_correct) + list(indices_to_selected_wrong) 
	print ("New", len(new_indices_to_target))
	# set input for the searcher -> searcher will look only upon this input hereafter

	### sample inputs for localisation (newly added) ###################
	# sampled_indices_to_correct -> will be used for localisation only. For the APR, all correct inputs will be used as default, but, actually I plan to compare both ...(?)
	if only_loc:
		print ("Target to sample", indices_to_correct[0])
		print ("\t", indices_to_selected_wrong[0])
		_, sampled_indices_to_correct = run_localise.sample_input_for_loc_by_rd(indices_to_selected_wrong, indices_to_correct)
	else:
		_, sampled_indices_to_correct = run_localise.sample_input_for_loc_by_rd(
			indices_to_selected_wrong, indices_to_correct, predictions[new_indices_to_target], data_y[new_indices_to_target])

	# redefine new_indices_to_target (this is global indices)
	new_indices_to_target = list(indices_to_selected_wrong) + list(indices_to_correct)
	new_indices_for_loc = list(indices_to_selected_wrong) + list(sampled_indices_to_correct)
	####################################################################

	# extraction for the target predictions
	predictions = predictions[new_indices_to_target] # slice

	# extraction for data
	X = data_X[new_indices_to_target]; X_for_loc = data_X[new_indices_for_loc]
	y = data_y[new_indices_to_target]; y_for_loc = data_y[new_indices_for_loc]

	########### For logging & testing ############# 
	num_of_our_target = len(new_indices_to_target); num_of_our_loc_target = len(new_indices_for_loc)
	num_of_wrong = len(indices_to_selected_wrong)
	num_of_correct = len(indices_to_correct)

	print ("The number of our target:%d, (%d(correct), %d(wrong))" % (num_of_our_target, num_of_correct, num_of_wrong))
	# set new local indices to correct & wrong for the new predictions
	indices_to_selected_wrong = list(range(0, num_of_wrong))
	indices_to_correct = list(range(num_of_wrong, num_of_our_target))
	indices_to_correct_for_loc = list(range(num_of_wrong, num_of_our_loc_target))

	assert_msg = "%d + %d vs %d" % (len(indices_to_correct), len(indices_to_selected_wrong), num_of_our_target)
	assert len(indices_to_correct) + len(indices_to_selected_wrong) == num_of_our_target, assert_msg
	assert len(X) == num_of_our_target, "%d vs %d" % (len(X), num_of_our_target)
	assert len(predictions) == num_of_our_target, "%d vs %d" % (len(predictions), num_of_our_target)
	########### logging and testing end ###########

	#t1 = time.time()
	import pickle
	import pandas as pd

	if loc_method == 'gradient_loss': # here, top n is not the number of inpouts, arather it is the number of neural weights to fix
		if not only_loc: # for RQ2 
			#### should fix this -> should be the average number of pareto-front size
			# the top_n for each dataset is configured empirically -> the mean of localised neural weights by our FL approch
			if which == 'simple_fm':
				top_n = int(np.round(7.6))
			elif which == 'simple_cm':
				top_n = int(np.round(11.6))
			else: # GTSRB
				top_n = int(np.round(14.3))
		else:
			# retrieve all
			top_n = -1

		indices_w_costs = run_localise.localise_by_gradient_v3(
				X_for_loc, y_for_loc,
				indices_to_selected_wrong,
				indices_to_correct_for_loc,
				target_weights,
				path_to_keras_model = path_to_keras_model,
				loss_funcs = loss_funcs)
		
		# retrieve only the indices
		indices_to_places_to_fix = [v[0] for v in indices_w_costs[:top_n]]
		loc_dest = os.path.join(loc_dest, "new_loc/{}/grad/on_test".format(which))
		os.makedirs(loc_dest, exist_ok=True)

		output_df = pd.DataFrame({'layer':[vs[0] for vs in indices_to_places_to_fix], 'weight':[vs[1] for vs in indices_to_places_to_fix]}) 
		destfile = os.path.join(loc_dest, "loc.{}.{}.pkl".format(patch_target_key, int(target_all)))
		output_df.to_pickle(destfile)

		# comment out b/c too big
		#with open(os.path.join(loc_dest, "loc.all_cost.{}.{}.grad.pkl".format(patch_target_key, int(target_all))), 'wb') as f:
		#	pickle.dump(indices_w_costs, f)	
		# comment out end for loc result saving 
	elif loc_method == 'localiser':
		if loc_file is None or not (os.path.exists(loc_file)):
			print (len(indices_to_correct_for_loc), len(indices_to_selected_wrong))
			print ("Now ready to localiser")
			### *** Now this may return (idx_to_tl, idx_to_w (0 for kerenl and 1 for recurr_kernel)) 
			print ("x", X_for_loc.shape)
			indices_to_places_to_fix, front_lst = run_localise.localise_by_chgd_unchgd(
				X_for_loc, y_for_loc,
				indices_to_selected_wrong,
				indices_to_correct_for_loc,
				target_weights,
				path_to_keras_model = path_to_keras_model, 
				loss_funcs = loss_funcs)

			print ("Places to fix", indices_to_places_to_fix)

			output_df = pd.DataFrame({'layer':[vs[0] for vs in indices_to_places_to_fix], 'weight':[vs[1] for vs in indices_to_places_to_fix]})
			loc_dest = os.path.join(loc_dest, "new_loc/{}/c_loc/on_test/".format(which))
			os.makedirs(loc_dest, exist_ok= True)
			destfile = os.path.join(loc_dest, "loc.{}.{}.pkl".format(patch_target_key, int(target_all)))
			output_df.to_pickle(destfile)
			
			print ("Saved to", destfile)
			#with open(os.path.join(loc_dest, "loc.all_cost.{}.{}.pkl".format(patch_target_key, int(target_all))), 'wb') as f:
			#	pickle.dump(front_lst, f)
				
		else: # since I dont' want to localise again
			import pandas as pd
			df = pd.read_pickle(loc_file)
			indices_to_places_to_fix = df.values
			
	else: # randomly select
		if not only_loc:
			if which == 'simple_fm':
				top_n = int(np.round(7.6))
			elif which == 'simple_cm':
				top_n = int(np.round(11.6))
			elif which == 'GTSBR': # GTSRB
				top_n = int(np.round(14.3))
			else: # for LSTM
				print ("not yet")
				import sys; sys.exit()
			num_random_sample = top_n 
		else:
			num_random_sample = -1

		indices_to_places_to_fix = run_localise.localise_by_random_selection(
			num_random_sample, target_weights)	 

		loc_dest = os.path.join(loc_dest, "new_loc/{}/random/on_test".format(which))	
		os.makedirs(loc_dest, exist_ok=True)

		output_df = pd.DataFrame({'layer':[vs[0] for vs in indices_to_places_to_fix], 'weight':[vs[1] for vs in indices_to_places_to_fix]}) 
		destfile = os.path.join(loc_dest, "loc.{}.{}.pkl".format(patch_target_key, int(target_all)))
		output_df.to_pickle(destfile)

		with open(os.path.join(loc_dest, "loc.all_cost.{}.{}.random.pkl".format(patch_target_key, int(target_all))), 'wb') as f:
			pickle.dump(indices_to_places_to_fix, f)

	t2 = time.time()
	#print ("Time taken for localisation: %f" % (t2 - t1))
	run_localise.reset_keras([model])
	if loc_method == 'localiser':
		print (indices_to_places_to_fix) # loggin

	import sys; sys.exit()
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
		list(set(
			[idx_to_tl if not isinstance(idx_to_tl, Iterable) else idx_to_tl[0] for idx_to_tl,_ in indices_to_places_to_fix])))
	print ("Patch target layers", indices_to_ptarget_layers)
	
	if search_method == 'DE':
		searcher = de.DE_searcher(
			X, y,
			indices_to_correct, [],
			num_label,
			indices_to_ptarget_layers,
			mutation = (0.5, 1), 
			recombination = 0.7,
			max_search_num = max_search_num,
			initial_predictions = None, #predictions,
			path_to_keras_model = path_to_keras_model,
			patch_aggr = patch_aggr,
			batch_size = batch_size,
			act_func = tf.nn.relu if which == 'GTSRB' else None,
			at_indices = None if which != 'lfw_vgg' else new_indices_to_target)

		places_to_fix = indices_to_places_to_fix
		searcher.set_indices_to_wrong(indices_to_selected_wrong)
		name_key = str(0) if patch_target_key is None else str(patch_target_key)
		patched_model_name, saved_path = searcher.search(places_to_fix, name_key = name_key)
	else:
		print ("{} not supported yet".format(search_method))
		import sys; sys.exit()
			
	t2 = time.time()
	print ("Time taken for pure patching: %f" % (t2 - t1))

	return saved_path, indices_to_selected_wrong, indices_patched




	
			



