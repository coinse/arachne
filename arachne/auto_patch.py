"""
Auto-patch module
	: contains functions related to patching NN model
"""
import utils.model_util as model_util
import utils.apricot_rel_util as apricot_rel_util
import utils.torch_rel_util as torch_rel_util 
import time
import search.other_localisers as other_localisers
from gen_frame_graph import generate_empty_graph


def where_to_fix_from_bl(
	indices_to_wrong,
	input_data, output_data, num_label,
	predictions, ## newly added
	which,
	empty_graph, 
	tensor_name_file, 
	init_plchldr_feed_dict = None,
	path_to_keras_model = None,
	name_of_target_weight = 'fw3',
	pareto_ret_all = False):
	"""
	"""
	from utils.data_util import read_tensor_name
	from search.det_localiser import Localiser
	import numpy as np

	nodes_to_lookat, indices_to_cands_and_grads = other_localisers.where_to_fix_from_gl(
		indices_to_wrong, 
		read_tensor_name(tensor_name_file)['t_lab_loss'],
		read_tensor_name(tensor_name_file)['t_weight'],
		num_label,
		input_data, output_data, 
		empty_graph = empty_graph,
		plchldr_feed_dict = init_plchldr_feed_dict,
		path_to_keras_model = path_to_keras_model,
		top_n = -1) # return all

	print ("Input empty graph", empty_graph)
	alocaliser = Localiser(
		input_data[indices_to_wrong], 
		output_data[indices_to_wrong],
		num_label,
		predictions, ## newly added
		tensor_name_file,
		empty_graph = None, #empty_graph, # will generate new empty graph for this
		which = which, 
		init_weight_value = None,
		nodes_to_lookat = None,
		path_to_keras_model = path_to_keras_model,
		base_indices_to_cifar10 = None)

	if which == 'lfw_vgg':
		pareto_ret_all = True # retrurn all

	places_to_fix, front_lst = alocaliser.run(
		indices_to_wrong,
		from_where_to_fix_nw_down = (nodes_to_lookat, indices_to_cands_and_grads),
		pareto_ret_all = pareto_ret_all)
		
	if which == 'lfw_vgg':
		places_to_fix = [p for ps in front_lst[:10] for p in ps]

	places_to_fix = list(set(places_to_fix))
	return places_to_fix, front_lst


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
	top_n = -1,
	seed = 1,
	only_loc = False,
	patch_aggr = None):
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
	import search.de as de
	import numpy as np
	import random

	USE_CORRECT_DATA_SAMPLING = True

	random.seed(seed)
	np.random.seed(seed)

	data_X, data_y = data
	num_data = len(data_X)
	
	assert num_data == len(data_y), "%d vs %d" % (num_data, len(data_y))
	from collections import Iterable
	if not isinstance(data_y[0], Iterable):
		from utils.data_util import format_label
		data_y = format_label(data_y, num_label)

	predict_tensor_name = "predc"
	corr_predict_tensor_name = 'correct_predc'

	empty_graph = generate_empty_graph(which, 
		data_X, 
		num_label, 
		path_to_keras_model = path_to_keras_model, 
		w_gather = False)
	
	if which != 'lfw_vgg':	
		kernel_and_bias_pairs = apricot_rel_util.get_weights(path_to_keras_model)
	else:
		kernel_and_bias_pairs = torch_rel_util.get_weights(path_to_keras_model)

	init_plchldr_feed_dict = {'fw3:0':np.float32(kernel_and_bias_pairs[-1][0]), 'fb3:0':kernel_and_bias_pairs[-1][1]}

	from utils.data_util import split_into_wrong_and_correct
	t1 = time.time()
	sess, (predictions, correct_predictions) = model_util.predict(
		data_X, data_y, num_label,
		predict_tensor_name = predict_tensor_name, 
		corr_predict_tensor_name = corr_predict_tensor_name,
		indices_to_slice_tensor_name = None, #'indices_to_slice' if w_gather else None,
		sess = None, 
		empty_graph = empty_graph,
		plchldr_feed_dict = init_plchldr_feed_dict,
		use_pretr_front = path_to_keras_model is not None)
	#
	sess.close()
	t2 = time.time()
	indices_to_target = split_into_wrong_and_correct(correct_predictions)
	entire_indices_to_wrong = indices_to_target['wrong']

	# check whether gien predef_indices_to_wrong to wrong is actually correct
	if predef_indices_to_wrong is not None:
		diff = set(predef_indices_to_wrong) - set(indices_to_target['wrong'])
		assert len(diff) == 0, diff 
		indices_to_target['wrong'] = predef_indices_to_wrong

	indices_to_selected_wrong = indices_to_target['wrong'] # target all of them 
	print ('Total number of wrongly processed input(s): {}'.format(len(indices_to_selected_wrong)))

	# check for correctly classified ones
	t1 = time.time()	
	indices_to_correct = indices_to_target['correct']

	t2 = time.time()
	# logging
	print ('Number of wrong: %d' % (len(indices_to_selected_wrong)))

	# extract the input vectors that are directly related to our target 
	# correct one first, followed by misclassified ones
	# FOR LFW, THIS WILL BE USED TO SLICE THE PRE-COMPUTE ATS
	new_indices_to_target = list(indices_to_correct) + list(indices_to_selected_wrong) 
	# set input for the searcher -> searcher will look only upon this input hereafter

	# extraction for predictions
	predictions = predictions[new_indices_to_target] # slice
	# extraction for data
	X = data_X[new_indices_to_target]
	y = data_y[new_indices_to_target]

	########### For logging & testing 
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
	########### logging and testing end

	# find the neuron in the last weight layer which is considered to be resposible for
	# wrong classification for each wrongly classified inputs
	from utils.data_util import read_tensor_name

	empty_graph_for_fl = generate_empty_graph(which, 
		X, 
		num_label, 
		path_to_keras_model = path_to_keras_model, 
		w_gather = True,
		indices = None if which != 'lfw_vgg' else new_indices_to_target)
	
	#t1 = time.time()
	if loc_method == 'gradient_loss': # here, top n is not the number of inpouts, arather it is the number of neural weights to fix
		if not only_loc: # for RQ2 
			top_n = int(np.round(13.3)) if which == 'simple_cm' else int(np.round(7.8))
		else:
			top_n = -1

		weight_tensor_name = read_tensor_name(tensor_name_file)['t_weight']
		loss_tensor_name = read_tensor_name(tensor_name_file)['t_lab_loss']

		indices_to_places_to_fix, indices_and_grads = other_localisers.where_to_fix_from_gl(
			indices_to_selected_wrong, 
			loss_tensor_name,
			weight_tensor_name,
			num_label,
			X, y,
			empty_graph = empty_graph_for_fl,
			path_to_keras_model = path_to_keras_model,
			plchldr_feed_dict = init_plchldr_feed_dict,
			top_n = top_n)

	elif loc_method == 'localiser':
		indices_to_places_to_fix, front_lst = where_to_fix_from_bl(
			indices_to_selected_wrong,
			X, y, num_label,
			predictions, ## newly added to compute the forward impact
			which,
			empty_graph_for_fl, 
			tensor_name_file, 
			init_plchldr_feed_dict = init_plchldr_feed_dict,
			path_to_keras_model = path_to_keras_model,
			pareto_ret_all = only_loc)

	else: # randomly select
		if not only_loc:
			num_random_sample = top_n 
		else:
			num_random_sample = -1

		indices_to_places_to_fix = other_localisers.where_to_fix_from_random(
			tensor_name_file,
			num_random_sample,
			empty_graph = empty_graph_for_fl)

	#t2 = time.time()
	#print ("Time taken for localisation: %f" % (t2 - t1))
	#print (indices_to_places_to_fix)

	if only_loc:
		if loc_method == 'localiser':
			return indices_to_places_to_fix, front_lst
		elif loc_method == 'gradient_loss':
			return indices_to_places_to_fix, indices_and_grads
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
	if search_method == 'DE':
		searcher = de.DE_searcher(
			X, y,
			indices_to_correct, [],
			num_label,
			tensor_name_file,
			mutation = (0.5, 1), 
			recombination = 0.7,
			max_search_num = max_search_num,
			initial_predictions = predictions, 
			path_to_keras_model = path_to_keras_model,
			empty_graph = None, 
			which = which,
			w_gather = False,
			patch_aggr = patch_aggr, 
			at_indices = None if which != 'lfw_vgg' else new_indices_to_target)

		places_to_fix = indices_to_places_to_fix
		searcher.set_indices_to_wrong(indices_to_selected_wrong)
		
		name_key = str(0) if patch_target_key is None else str(patch_target_key)
			
		patched_model_name = searcher.search(
			places_to_fix,
			sess = None,
			name_key = name_key)
	else:
		print ("{} not supported yet".format(search_method))
		import sys; sys.exit()
			
	t2 = time.time()
	print ("Time taken for pure patching: %f" % (t2 - t1))
	
	return patched_model_name, indices_to_selected_wrong, indices_patched




	
			



