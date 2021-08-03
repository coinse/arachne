"""
Auto-patch module
	: contains functions related to patching NN model
"""
import utils.model_util as model_util
import utils.apricot_rel_util as apricot_rel_util
import utils.torch_rel_util as torch_rel_util 
import utils.data_util as data_util
import time
import os
import search.other_localisers as other_localisers
from gen_frame_graph import generate_empty_graph

# def where_to_fix_from_bl(
# 	indices_to_wrong,
# 	input_data, output_data, num_label,
# 	predictions, ## newly added
# 	which,
# 	empty_graph, 
# 	tensor_name_file, 
# 	init_plchldr_feed_dict = None,
# 	path_to_keras_model = None,
# 	name_of_target_weight = 'fw3',
# 	pareto_ret_all = False):
# 	"""
# 	"""
# 	from utils.data_util import read_tensor_name
# 	from search.det_localiser import Localiser
# 	import numpy as np
#
# 	nodes_to_lookat, indices_to_cands_and_grads = other_localisers.where_to_fix_from_gl(
# 		indices_to_wrong, 
# 		read_tensor_name(tensor_name_file)['t_lab_loss'],
# 		read_tensor_name(tensor_name_file)['t_weight'],
# 		num_label,
# 		input_data, output_data, 
# 		empty_graph = empty_graph,
# 		plchldr_feed_dict = init_plchldr_feed_dict,
# 		path_to_keras_model = path_to_keras_model,
# 		top_n = -1) # return all
#
# 	print ("Input empty graph", empty_graph)
# 	alocaliser = Localiser(
# 		input_data[indices_to_wrong], 
# 		output_data[indices_to_wrong],
# 		num_label,
# 		predictions, ## newly added
# 		tensor_name_file,
# 		empty_graph = None, #empty_graph, # will generate new empty graph for this
# 		which = which, 
# 		init_weight_value = None,
# 		nodes_to_lookat = None,
# 		path_to_keras_model = path_to_keras_model,
# 		base_indices_to_cifar10 = None)
#
# 	if which == 'lfw_vgg':
# 		pareto_ret_all = True # retrurn all
#
# 	places_to_fix, front_lst = alocaliser.run(
# 		indices_to_wrong,
# 		from_where_to_fix_nw_down = (nodes_to_lookat, indices_to_cands_and_grads),
# 		pareto_ret_all = pareto_ret_all)
#		
# 	if which == 'lfw_vgg':
# 		places_to_fix = [p for ps in front_lst[:10] for p in ps]
#
# 	places_to_fix = list(set(places_to_fix))
# 	return places_to_fix, front_lst


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
	predef_indices_to_wrong = None,
	#top_n = -1,
	seed = 1,
	only_loc = False,
	patch_aggr = None, 
	target_all = False,
	loc_file = None):
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

	random.seed(seed)
	np.random.seed(seed)

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
	
	import subprocess	
	model = load_model(path_to_keras_model, compile = False)
	result = subprocess.run(['nvidia-smi'], shell = True) #stdout = subprocess.PIPE.stdout, stderr = subprocess.PIPE.stderr)
	print (result)
	target_weights = run_localise.get_target_weights(model,
		path_to_keras_model, 
		indices_to_target = indices_to_target_layers, 
		target_all = target_all) # if target_all == True, then indices_to_target will be ignored

	result = subprocess.run(['nvidia-smi'], shell = True) #stdout = subprocess.PIPE.stdout, stderr = subprocess.PIPE.stderr)
	print (result)
	print ('Total {} layers are targeted'.format(target_weights.keys()))
	#### HOW CAN WE KNOW WHICH LAYER IS PREDICTION LAYER and WEIGHT LAYER? => assumes they are given;;;
	# if not, then ... well everything becomes complicated
	# identify using print (l['name'], l['class_name']) ..? d['layers'] -> mdl.get_config()
	## -> at least for predc & corr_predc
	# compute prediction & corr_predictions
	predictions = model.predict(data_X)
	correct_predictions = np.argmax(predictions, axis = 1)
	correct_predictions = correct_predictions == np.argmax(data_y, axis = 1)
	
	indices_to_target = data_util.split_into_wrong_and_correct(correct_predictions)

	#check whether gien predef_indices_to_wrong to wrong is actually correct
	if predef_indices_to_wrong is not None:
		diff = set(predef_indices_to_wrong) - set(indices_to_target['wrong'])
		assert len(diff) == 0, diff 
		indices_to_target['wrong'] = predef_indices_to_wrong

	indices_to_selected_wrong = indices_to_target['wrong'] # target all of them 
	print ('Total number of wrongly processed input(s): {}'.format(len(indices_to_selected_wrong)))

	indices_to_correct = indices_to_target['correct']
	#if which == 'GTSRB':
	#	num = int(len(indices_to_correct)/2)
	#	indices_to_correct = np.random.choice(indices_to_correct, num, replace = False)
	#	print ("Due to memory allocation error, we use only half of it: {} -> {}".format(len(indices_to_correct), num))
	# logging
	print ('Number of wrong: %d' % (len(indices_to_selected_wrong)))

	# extract the input vectors that are directly related to our target 
	# correct one first, followed by misclassified ones
	# FOR LFW, THIS WILL BE USED TO SLICE THE PRE-COMPUTE ATS
	new_indices_to_target = list(indices_to_correct) + list(indices_to_selected_wrong) 
	# set input for the searcher -> searcher will look only upon this input hereafter

	# extraction for the target predictions
	predictions = predictions[new_indices_to_target] # slice
	# extraction for data
	X = data_X[new_indices_to_target]
	y = data_y[new_indices_to_target]

	########### For logging & testing ############# 
	num_of_our_target = len(new_indices_to_target)
	num_of_wrong = len(indices_to_selected_wrong)
	num_of_correct = len(indices_to_correct)

	print ("The number of our target:%d, (%d(correct), %d(wrong))" % (num_of_our_target, num_of_correct, num_of_wrong))
	# set new local indices to correct & wrong for the new predictions
	indices_to_correct = list(range(0, num_of_correct))
	indices_to_selected_wrong = list(range(num_of_correct, num_of_our_target))

	assert_msg = "%d + %d vs %d" % (len(indices_to_correct), len(indices_to_selected_wrong), num_of_our_target)
	assert len(indices_to_correct) + len(indices_to_selected_wrong) == num_of_our_target, assert_msg
	assert len(X) == num_of_our_target, "%d vs %d" % (len(X), num_of_our_target)
	assert len(predictions) == num_of_our_target, "%d vs %d" % (len(predictions), num_of_our_target)
	########### logging and testing end ###########

	#t1 = time.time()
	if loc_method == 'gradient_loss': # here, top n is not the number of inpouts, arather it is the number of neural weights to fix
		if not only_loc: # for RQ2 
			#### should fix this 
			top_n = int(np.round(13.3)) if which == 'simple_cm' else int(np.round(7.8))
		else:
			# retrieve all
			top_n = -1

		indices_w_costs = run_localise.localise_by_gradient(
				X, y,
				indices_to_selected_wrong,
				target_weights,
				path_to_keras_model = path_to_keras_model)
		
		# retrieve only the indices
		indices_to_places_to_fix = [v[0] for v in indices_w_costs[:top_n]]
	elif loc_method == 'localiser':
		# indices_to_places_to_fix, front_lst = where_to_fix_from_bl(
		# 	indices_to_selected_wrong,
		# 	X, y, num_label,
		# 	predictions, ## newly added to compute the forward impact
		# 	which,
		# 	empty_graph_for_fl, 
		# 	tensor_name_file, 
		# 	init_plchldr_feed_dict = init_plchldr_feed_dict,
		# 	path_to_keras_model = path_to_keras_model,
		# 	pareto_ret_all = only_loc)
		print (os.path.exists(loc_file))
		if loc_file is None or not (os.path.exists(loc_file)):
			indices_to_places_to_fix, front_lst = run_localise.localise_offline_v2(
				X, y,
				indices_to_selected_wrong,
				target_weights,
				path_to_keras_model = path_to_keras_model)
			print ("Places to fix", indices_to_places_to_fix)
			#import sys; sys.exit()
			import pickle
			import pandas as pd
			output_df = pd.DataFrame({'layer':[vs[0] for vs in indices_to_places_to_fix], 'weight':[vs[1] for vs in indices_to_places_to_fix]})
			loc_dest = os.path.join("new_loc/{}".format(which))
			os.makedirs(loc_dest, exist_ok= True)
			destfile = os.path.join(loc_dest, "rq5.{}.{}.pkl".format(patch_target_key, int(target_all)))
			output_df.to_pickle(destfile)

			with open(os.path.join(loc_dest, "rq5.all_cost.{}.{}.pkl".format(patch_target_key, int(target_all))), 'wb') as f:
				pickle.dump(front_lst, f)
		else: # since I dont' want to localise again
			#indices_to_places_to_fix = [(2, (0, 2, 0, 9)), (2, (1, 1, 0, 9)), (2, (1, 2, 1, 9)), (2, (2, 1, 0, 9)), (2, (2, 1, 2, 9)), (2, (2, 2, 0, 9)), (25, (553, 3)), (25, (553, 9)), (25, (970, 4)), (25, (1977, 5))]
			import pandas as pd
			df = pd.read_pickle(loc_file)
			print (df)
			indices_to_places_to_fix = df.values
	else: # randomly select
		if not only_loc:
			top_n = int(np.round(13.3)) if which == 'simple_cm' else int(np.round(7.8))
			num_random_sample = top_n 
		else:
			num_random_sample = -1

		indices_to_places_to_fix = run_localise.localise_by_random_selection(
			num_random_sample, target_weights)	 

	#t2 = time.time()
	#print ("Time taken for localisation: %f" % (t2 - t1))
	run_localise.reset_keras([model])
	print (indices_to_places_to_fix)
	result = subprocess.run(['nvidia-smi'], shell = True)
	print (result)
	if only_loc:
		if loc_method == 'localiser':
			return indices_to_places_to_fix, front_lst
		elif loc_method == 'gradient_loss':
			#return indices_to_places_to_fix, indices_and_grads
			return indices_to_places_to_fix, indices_w_costs
		else:
			#return indices_to_places_to_fix, None
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
	indices_to_ptarget_layers = sorted(list(set([idx_to_tl for idx_to_tl,_ in indices_to_places_to_fix])))
	print ("Patch target layers", indices_to_ptarget_layers)
	#import sys; sys.exit()

	if search_method == 'DE':
		# searcher = de.DE_searcher(
		# 	X, y,
		# 	indices_to_correct, [],
		# 	num_label,
		# 	tensor_name_file,
		# 	mutation = (0.5, 1), 
		# 	recombination = 0.7,
		# 	max_search_num = max_search_num,
		# 	initial_predictions = predictions, 
		# 	path_to_keras_model = path_to_keras_model,
		# 	empty_graph = None, 
		# 	which = which,
		# 	w_gather = False,
		# 	patch_aggr = patch_aggr, 
		# 	at_indices = None if which != 'lfw_vgg' else new_indices_to_target)
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
			at_indices = None if which != 'lfw_vgg' else new_indices_to_target)

		places_to_fix = indices_to_places_to_fix
		searcher.set_indices_to_wrong(indices_to_selected_wrong)
		
		name_key = str(0) if patch_target_key is None else str(patch_target_key)
		
		#patched_model_name = searcher.search(
		#	places_to_fix,
		#	sess = None,
		#	name_key = name_key)
		patched_model_name, saved_path = searcher.search(places_to_fix, name_key = name_key)

	else:
		print ("{} not supported yet".format(search_method))
		import sys; sys.exit()
			
	t2 = time.time()
	print ("Time taken for pure patching: %f" % (t2 - t1))
	
	#return patched_model_name, indices_to_selected_wrong, indices_patched
	return saved_path, indices_to_selected_wrong, indices_patched




	
			



