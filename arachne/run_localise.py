"""
Localise faults in offline for any faults
"""
#from main_eval import read_and_add_flag, combine_init_aft_predcs
import numpy as np
import random
import utils.data_util as data_util
import time
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

def get_target_weights(model, path_to_keras_model, indices_to_target = None, target_all = True):
	"""
	return indices to weight layers denoted by indices_to_target, or return all trainable layers
	"""
	import re

	# target only the layer with its class type in this list, but if target_all, then return all trainables
	targeting_clname_pattns = ['Dense*', 'Conv*'] if not target_all else None
	is_target = lambda clname,targets: any([bool(re.match(t,clname)) for t in targets]) or (targets is None)
	if target_all:
		indices_to_target = None

	if model is None:
		assert path_to_keras_model is not None
		model = load_model(path_to_keras_model, compile=False)

	target_weights = {} # key = layer index, value: [weight value, layer name]
	if indices_to_target:
		for i, layer in enumerate(model.layers):
			if i in indices_to_target:
				ws = layer.get_weights()
				assert len(ws) > 0, "the target layer doesn't have weight"
				#target_weights[i] = ws[0] # we will target only the weight, and not the bias
				target_weights[i] = [ws[0], type(layer).__name__]
	else:
		for i, layer in enumerate(model.layers):
			class_name = type(layer).__name__
			if is_target(class_name, targeting_clname_pattns): 
				ws = layer.get_weights()
				if len(ws): # has weight
					#target_weights[i] = ws[0]
					target_weights[i] = [ws[0], type(layer).__name__]
					
	return target_weights

def is_FC(lname):
	"""
	"""
	import re
	pattns = ['Dense*'] # now, only Dense
	return any([bool(re.match(t,lname)) for t in pattns])

def is_C2D(lname):
	"""
	"""
	import re
	pattns = ['Conv2D']
	return any([bool(re.match(t,lname)) for t in pattns])

def compute_gradient_to_output(model, target, X):
	"""
	compute gradients normalisesd and averaged for a given input X
	"""
	import tensorflow.keras.backend as K
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")

	sess = K.get_session()
	tensor_grad = tf.gradients(
		model.output, 
		target,
		name = 'output_grad')

	gradient = sess.run(tensor_grad, feed_dict={model.input: X})[0]
	sess.close()

	print ("tensor grad", tensor_grad)
	print ("\t", gradient.shape)

	gradient = np.abs(gradient)
	reshaped_gradient = gradient.reshape(gradient.shape[0],-1) # flatten
	norm_gradient = norm_scaler.fit_transform(reshaped_gradient) # normalised
	mean_gradient = np.mean(norm_gradient, axis = 0) # compute mean for a given input
	ret_gradient = mean_gradient.reshape(gradient.shape[1:]) # reshape to the orignal shape
	
	return ret_gradient 
			
def localise_offline(
	num_label,
	data,
	tensor_name_file,
	path_to_keras_model = None,
	predef_indices_to_wrong = None,
	seed = 1,
	target_all = False):
	"""
	localise offline
	"""
	random.seed(seed)
	np.random.seed(seed)

	data_X, data_y = data
	num_data = len(data_X)
	assert num_data == len(data_y), "%d vs %d" % (num_data, len(data_y))
	
	from collections import Iterable
	if not isinstance(data_y[0], Iterable):
		from utils.data_util import format_label
		data_y = format_label(data_y, num_label)

	# this will be fixed (..at least for classification tasks)
	#predict_tensor_name = "predc"
	#corr_predict_tensor_name = 'correct_predc'
	#if which != 'lfw_vgg':
	#	kernel_and_bias_pairs = apricot_rel_util.get_weights(path_to_keras_model)
	#else:
	#	kernel_and_bias_pairs = torch_rel_util.get_weights(path_to_keras_model)
	#init_plchldr_feed_dict = {'fw3:0':np.float32(kernel_and_bias_pairs[-1][0]), 'fb3:0':kernel_and_bias_pairs[-1][1]}

	#init_plchldr_feed_dict = {}
	# index to target layer: e.g., 0 = the first hidden layer
	if not target_all:
		indices_to_target_layers = np.int32(data_util.read_tensor_name(tensor_name_file)['t_layer'])
	else:
		indices_to_target_layers = None
	# get ranges
	# min_idx_to_tl = np.min(indices_to_target_layers); max_idx_to_tl = np.max(indices_to_target_layers)
	#target_weights = {}
	##for idx_to_tl in np.arange(min_idx_to_tl, max_idx_to_tl + 1):
	#for idx_to_tl in indices_to_target_layers:
	#	target_weights[idx_to_tl] = [kernel_and_bias_pairs[idx_to_tl]]

	model = load_model(path_to_keras_model)
	target_weights = get_target_weights(model,
		path_to_keras_model, 
		indices_to_target = indices_to_target_layers, 
		target_all = target_all) # if target_all == True, then indices_to_target will be ignored

	#### HOW CAN WE KNOW WHICH LAYER IS PREDICTION LAYER and WEIGHT LAYER? => assumes they are given;;;
	# if not, then ... well everything becomes complicated
	# identify using print (l['name'], l['class_name']) ..? d['layers'] -> mdl.get_config()
	## -> at least for predc & corr_predc
	# compute prediction & corr_predictions
	predictions = model.predict(data_X)
	correct_predictions = np.argmax(predictions, axis = 1)

	# ##########
	# empty_graph = generate_empty_graph(which, 
	# 	data_X, 
	# 	num_label, 
	# 	weight_shapes,
	# 	min_idx_to_tl,
	# 	path_to_keras_model = path_to_keras_model, 
	# 	w_gather = False)
	#from utils.data_util import split_into_wrong_and_correct
	#t1 = time.time()
	# sess, (predictions, correct_predictions) = model_util.predict(
	# 	data_X, data_y, num_label,
	# 	predict_tensor_name = predict_tensor_name, 
	# 	corr_predict_tensor_name = corr_predict_tensor_name,
	# 	indices_to_slice_tensor_name = None, #'indices_to_slice' if w_gather else None,
	# 	sess = None, 
	# 	empty_graph = empty_graph,
	# 	plchldr_feed_dict = init_plchldr_feed_dict,
	# 	use_pretr_front = path_to_keras_model is not None)
	# #
	# sess.close()
	#t2 = time.time()

	indices_to_target = data_util.split_into_wrong_and_correct(correct_predictions)
	#entire_indices_to_wrong = indices_to_target['wrong']

	#check whether gien predef_indices_to_wrong to wrong is actually correct
	if predef_indices_to_wrong is not None:
		diff = set(predef_indices_to_wrong) - set(indices_to_target['wrong'])
		assert len(diff) == 0, diff 
		indices_to_target['wrong'] = predef_indices_to_wrong

	indices_to_selected_wrong = indices_to_target['wrong'] # target all of them 
	print ('Total number of wrongly processed input(s): {}'.format(len(indices_to_selected_wrong)))

	indices_to_correct = indices_to_target['correct']
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

	## Now, start localisation !!! ##
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")

	total_cands = {}
	FIs = None; grad_scndcr = None

	print ('Total {} layers are targeted'.format(len(target_weights)))
	t0 = time.time()
	for idx_to_tl, vs in target_weights.items():
		t1 = time.time()
		total_cands.append([])
		t_w, lname = vs
		############ FI ############
		t_model = Model(inputs = model.input, output = model.layers[idx_to_tl - 1].output)
		prev_output = t_model.predict(X)
		layer_config = model.layers[idx_to_tl].get_config() 

		# if this takes too long, then change to tensor and compute them using K (backend)
		if is_FC(lname):
			from_front = []
			for idx in range(t_w.shape[-1]):
				assert prev_output.shape[-1] == t_w.shape[0], "{} vs {}".format(
					prev_output.shape[-1], t_w.shape[0])

				output = np.multiply(prev_output, t_w[:,idx]) # -> shape = prev_output.shape
				output = np.abs(output)
				output = norm_scaler.fit_transform(output) # -> shape = prev_output.shape (normalisation on )
				#output_tensor = tf.math.reduce_mean(output_tensor, axis = 0) # compute an average for given inputs
				output = np.mean(output, axis = 0) # -> shape = (reshaped_t_w.shape[-1],)
				#temp_tensors.append(output_tensor)
				from_front.append(output) # 

			# work for Dense. but, for the others?
			from_front = np.asarray(from_front)
			from_front = from_front.T
			print ('From front'. from_front.shape)
			# behind
			# sess = K.get_session()
			# tensor_grad = tf.gradients(
			# 	model.output, 
			# 	model.layers[idx_to_tl],
			# 	name = 'output_grad')
			# gradient = sess.run(tensor_grad, feed_dict={model.input: X})[0]
			# sess.close()
			# print ("tensor grad", tensor_grad)
			# print ("\t", gradient.shape)
			# gradient = np.abs(gradient)
			# reshaped_gradient = gradient.reshape(gradient.shape[0],-1)
			# norm_gradient = norm_scaler.fit_transform(reshaped_gradient)
			# mean_gradient = np.mean(norm_gradient, axis = 0)
			# gradient_value_from_behind = mean_gradient.reshape(gradient.shape[1:])
			# from_behind = gradient_value_from_behind # pos... what if pos is 3-d 
			from_behind = compute_gradient_to_output(model, model.layers[idx_to_tl], X)
			print ("From behind", from_behind.shape) 
			FIs = from_front * from_behind
			############ FI end #########

			# Gradient
			grad_scndcr = compute_gradient_to_output(model, model.layers[idx_to_tl].weights[0], X)	
			# G end
		elif is_C2D(lname):
			kernel_shape = t_w.shape[:2]  # kernel == filter
			#true_ws_shape = t_w.shape[2:] # this is (Channel_in (=prev_output.shape[1]), Channel_out (=output.shape[1]))
			strides = layer_config['strides']
			padding_type =  layer_config['padding']

			if padding_type == 'valid':
				paddings = [0,0]
			elif padding_type == 'same':
				#P = ((S-1)*W-S+F)/2
				#paddings = [int(((strides[i]-1)*true_ws_shape[i]-strides[i]+kernel_shape[i])/2) for i in range(2)]
				paddings = [0,0] # since, we are getting the padded input
			else:
				print ("padding type: {} not supported".format(padding_type))
				paddings = [0,0]

			num_kernels = prev_output.shape[1] # Channel_in
			assert num_kernels == t_w.shape[2], "{} vs {}".format(num_kernels, t_w.shape[2])

			# H x W				
			input_shape = prev_output.shape[2:] # the last two (front two are # of inputs and # of kernels (Channel_in))

			# (W1−F+2P)/S+1, W1 = input volumne , F = kernel, P = padding
			n_mv_0 = (input_shape[0] - kernel_shape[0] + 2 * paddings[0])/strides[0] + 1 # H_out
			n_mv_1 = (input_shape[1] - kernel_shape[1] + 2 * paddings[1])/strides[1] + 1 # W_out
			# indices_to_each_w = {}
			# for i in range(kernel_shape[0]): # t_w.shape[0]
			# 	for j in range(kernel_shape[1]): # t_w.shape[1]
			# 		#indices_to_each_w[p_to_kernel] = {'org':(i,j)}
			# 		#indices_to_each_w[p_to_kernel]['to_lookat'] = np.asarray([[idx_1+p_to_kernel, idx_2+p_to_kernel] \
			# 		#	for idx_1 in range(n_mv_0) for idx_2 in range(n_mv_1)])
			# 		to_lookat = [[idx_1 + i, idx_2 + j] for idx_1 in range(n_mv_0) for idx_2 in range(n_mv_1)]
			# 		indices_to_each_w[(i,j)] = np.asarray(to_lookat)
			##
			k = 0 # for logging
			n_output_channel = t_w.shape[-1] # Channel_out
			#from_front = []
			# for idx_ol in range(n_output_channel): # t_w.shape[-1]
			# 	# for idx_2 in range(num_kernels): # num_kernels = t_w.shape[2]
			# 	# 	for j,vs in indices_to_each_w.items():
			# 	# 		from_w = t_w[idx_1, idx_2,vs['orig'][0],vs['org'][1]]
			# 	# 		from_prev = prev_output[:,idx_2,vs['to_lookat'][:,0]][:,vs['to_lookat'][:,1]]
			# 	# 	pass
			# 	###
			# 	from_front.append([]) # length would be kernel_shape[0] * kernel_shape[1]
			# 	for idx_to_w, to_lookat in indices_to_each_w.items():
			# 		from_w = t_w[idx_to_w[0], idx_to_w[1], :, idx_ol] 
			# 		# # input, # prev channel, height, width 
			# 		from_prev = prev_output[:, :, to_lookat[:,0], to_lookat[:,1]]
			# 		if k == 0: # loging
			# 			print ("Sample", from_w.shape, from_prev.shape)
			# 		output = from_prev * from_w
			# 		if k == 0:
			# 			print ("output", output.shape)
			# 		output = np.abs(output)
			# 		output = norm_scaler.fit_transform(output)
			# 		output = np.mean(output, axis = 0) 
			# 		if k == 0:
			# 			print ('mean', output.shape)
			# 			k += 1
			# 		from_front[-1].append(output)
			# 	###
			#from_front = np.zeros((n_output_channel, n_mv_0, n_mv_1))
			from_front = []
			# move axis for easier computation
			tr_prev_output = np.moveaxis(prev_output, [0,1,2,3], [0,3,1,2])
			for idx_ol in range(n_output_channel): # t_w.shape[-1]
				for i in range(n_mv_0): # H
					indices_to_k1 = np.arange(i*strides[0], i*strides[0]+kernel_shape[0], 1)
					for j in range(n_mv_1): # W
						indices_to_k2 = np.arange(j*strides[1], j*strides[1]+kernel_shape[1], 1)	
						curr_prev_output = tr_prev_output[:,indices_to_k1,:,:][:,:,indices_to_k2,:]
						output = curr_prev_output * t_w[:,:,:,idx_ol]
						output = np.abs(output)
						if k == 0:
							print ("output", output.shape)
						
						sum_output = np.sum(output) # since they are all added to compute a single output tensor
						output = output/sum_output # normalise -> [# input, F1, F2, Channel_in]
						output = np.mean(output, axis = 0) # sum over a given input set # [F1, F2, Channel_in]
						if k == 0:
							print ('mean', output.shpe, "should be", (kernel_shape[0],kernel_shape[1],prev_output.shape[1]))
						#from_front[(idx_ol, i, j)] = output # output -> []
						from_front.append(output)

			from_front = np.asarray(from_front)
			print ("From front", from_front.shape) # [Channel_out * n_mv_0 * n_mv_1, F1, F2, Channel_in]
			from_front = from_front.reshape(
				(n_output_channel,n_mv_0,n_mv_1,kernel_shape[0],kernel_shape[1],prev_output[1]))
			print ("reshaped", from_front.shape)
			from_front = np.moveaxis(from_front, [0,1,2], [3,4,5])
			print ("axis moved", from_front.shape) # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]

			# from behind
			# sess = K.get_session()
			# tensor_grad = tf.gradients(
			# 	model.output, 
			# 	model.layers[idx_to_tl],
			# 	name = 'output_grad')
			# gradient = sess.run(tensor_grad, feed_dict={model.input: X})[0]
			# sess.close()
			# print ("tensor grad", tensor_grad)
			# print ("\t", gradient.shape)
			# gradient = np.abs(gradient)
			# reshaped_gradient = gradient.reshape(gradient.shape[0],-1)
			# norm_gradient = norm_scaler.fit_transform(reshaped_gradient)
			# mean_gradient = np.mean(norm_gradient, axis = 0) # compute mean for a given input
			# gradient_value_from_behind = mean_gradient.reshape(gradient.shape[1:])
			# from_behind = gradient_value_from_behind # pos... what if pos is 3-d 
			# the same shape with output -> [Channel_output, H_out (n_mv_0), W_out (n_mv_1)]
			from_behind = compute_gradient_to_output(model, model.layers[idx_to_tl], X) # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
			print ("From behind", from_behind.shape)
			from_behind = from_behind.reshape(-1,) # [Channel_out * n_mv_0 * n_mv_1,]
			FIs = from_front * from_behind # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]
			FIs = np.mean(np.mean(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]

			## Gradient
			# will be [F1, F2, Channel_in, Channel_out]
			grad_scndcr = compute_gradient_to_output(model, model.layers[idx_to_tl].weights[0], X)
			# ##		
		else:
			print ("Currenlty not supported: {}. (shoulde be filtered before)".format(lname))		
			import sys; sys.exit()

		t2 = time.time()
		print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
		####
		pairs = np.asarray([FIs.flatten(), grad_scndcr.flatten()]).T
		total_cands[idx_to_tl] = {'shape':FIs.shape, 'costs':pairs}
	
	t3 = time.time()
	print ("Time for computing total costs: {}".format(t3 - t0))

	# compute pareto front
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = [([idx_to_tl, local_i], c) for idx_to_tl in indices_to_tl for local_i,c in enumerate(total_cands[total_cands]['costs'])]
	costs = [vs[1] for vs in costs_and_keys]

	def get_org_index(flatten_idx, cands):
		"""
		"""
		org_index = []
		for local_s in cands['shape']:
			org_index.append(int(flatten_idx / local_s))
			flatten_idx = flatten_idx % local_s
		return ",".join(org_index)

	# a list of [index to the target layer, index to a neural weight]
	indices_to_nodes = [[vs[0][0], get_org_index(vs[0][1], total_cands[vs[0][0]])] for vs in costs_and_keys]

	t4 = time.time()
	#while len(curr_nodes_to_lookat) > 0:
	_costs = costs.copy()
	is_efficient = np.arange(costs.shape[0])
	next_point_index = 0 # Next index in the is_efficient array to search for
	while next_point_index < len(_costs):
		nondominated_point_mask = np.any(_costs > _costs[next_point_index], axis=1)
		nondominated_point_mask[next_point_index] = True
		is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
		_costs = _costs[nondominated_point_mask]
		next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1	

	pareto_front = [tuple(v) for v in np.asarray(indices_to_nodes)[is_efficient]]
	t5 = time.time()
	print ("Time for computing the pareto front: {}".format(t5 - t4))
	return pareto_front

# if __name__ == "__main__":
# 	import argparse
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("-init_pred_file", type = str, help = "original")
# 	parser.add_argument("-aft_pred_file", type = str, help = "with noise")
# 	parser.add_argument("-num_label", type = int, default = 10)
# 	parser.add_argument("-datadir", action = "store", default = "data", type = str)
# 	parser.add_argument("-which", action = "store", 
# 		help = 'simple_cm, simple_fm', type = str)
# 	parser.add_argument('-which_data', action = "store",
# 		default = 'cifar10', type = str, help = 'fashion_mnist,cifaf10,lfw')
# 	parser.add_argument("-tensor_name_file", action = "store",
# 		default = "data/tensor_names/tensor.lastLayer.names ", type = str)
# 	parser.add_argument("-loc_method", action = "store", default = None, help = 'random, localiser, gradient_loss')
# 	parser.add_argument("-path_to_keras_model", action = 'store', default = None)
# 	parser.add_argument("-seed", action = "store", default = 1, type = int)
# 	parser.add_argument("-dest", default = ".", type = str)

# 	args = parser.parse_args()	
# 	###
# 	init_pred_df = read_and_add_flag(args.init_pred_file)
# 	aft_pred_df = read_and_add_flag(args.aft_pred_file)
# 	combined_df = combine_init_aft_predcs(init_pred_df, aft_pred_df)

# 	brokens = get_brokens(combined_df)
# 	patcheds = get_patcheds(combined_df)

# 	indices_to_wrong = brokens.index.values

# 	# is_input_2d = True => to match the format with faulty model
# 	train_data, test_data = data_util.load_data(args.which_data, args.datadir, is_input_2d = True)
# 	train_X, train_y = train_data
# 	num_train = len(train_y)
# 	test_X, test_y = test_data

# 	dest = args.dest
# 	os.makedirs(dest, exist_ok = True)
# 	path_to_loc_file = set_loc_name(dest, args.aft_pred_file, args.seed)
# 	###
# 	### from here algorithm independent
# 	USE_CORRECT_DATA_SAMPLING = True

# 	random.seed(args.seed)
# 	np.random.seed(args.seed)

# 	data_X, data_y = train_data
# 	num_data = len(data_X)
# 	num_label = args.num_label
# 	path_to_keras_model = args.path_to_keras_model
# 	tensor_name_file = args.tensor_name_file

# 	assert num_data == len(data_y), "%d vs %d" % (num_data, len(data_y))
# 	from collections import Iterable
# 	if not isinstance(data_y[0], Iterable):
# 		from utils.data_util import format_label
# 		data_y = format_label(data_y, num_label)

# 	# this will be fixed (..at least for classification tasks)
# 	predict_tensor_name = "predc"
# 	corr_predict_tensor_name = 'correct_predc'
	
# 	if args.which != 'lfw_vgg':	
# 		kernel_and_bias_pairs = apricot_rel_util.get_weights(path_to_keras_model)
# 	else:
# 		kernel_and_bias_pairs = torch_rel_util.get_weights(path_to_keras_model)

# 	#init_plchldr_feed_dict = {'fw3:0':np.float32(kernel_and_bias_pairs[-1][0]), 'fb3:0':kernel_and_bias_pairs[-1][1]}
# 	init_plchldr_feed_dict = {}
# 	indices_to_target_layers = np.int32(data_util.read_tensor_name(args.tensor_name_file)['t_layer']) # index to target layer: e.g., 0 = the first hidden layer
# 	min_idx_to_tl = np.min(indices_to_target_layers); max_idx_to_tl = np.max(indices_to_target_layers)
# 	target_weights = {}
# 	for idx_to_tl in np.arange(min_idx_to_tl, max_idx_to_tl + 1):
# 		#init_plchldr_feed_dict['fw{}:0'.format(idx_to_tl)] = kernel_and_bias_pairs[idx_to_tl][0]
# 		#init_plchldr_feed_dict['fb{}:0'.format(idx_to_tl)] = kernel_and_bias_pairs[idx_to_tl][1]
# 		target_weights[idx_to_tl] = [kernel_and_bias_pairs[idx_to_tl]

# 	##
# 	#for idx_to_tl in np.arange(min_idx_to_tl, max_idx_to_tl + 1):
# 	#	model, _ = gen_frame_graph.build_keras_model_front_v2(path_to_keras_model, idx_to_target_w = -1)
# 	#	outputs = model.predict(inputs)
# 	##

# 	empty_graph = gen_frame_graph.generate_empty_graph(which, 
# 		data_X, 
# 		num_label, 
# 		weight_shapes,
# 		min_idx_to_tl,
# 		path_to_keras_model = path_to_keras_model, 
# 		w_gather = False)

# 	t1 = time.time()
# 	sess, (predictions, correct_predictions) = model_util.predict(
# 		data_X, data_y, num_label,
# 		predict_tensor_name = predict_tensor_name, 
# 		corr_predict_tensor_name = corr_predict_tensor_name,
# 		indices_to_slice_tensor_name = None, #'indices_to_slice' if w_gather else None,
# 		sess = None, 
# 		empty_graph = empty_graph,
# 		plchldr_feed_dict = init_plchldr_feed_dict,
# 		use_pretr_front = path_to_keras_model is not None)
# 	#
# 	sess.close()
# 	t2 = time.time()
# 	indices_to_target = data_util.split_into_wrong_and_correct(correct_predictions)
# 	entire_indices_to_wrong = indices_to_target['wrong']

# 	# check whether gien predef_indices_to_wrong to wrong is actually correct
# 	if indices_to_wrong is not None:
# 		diff = set(indices_to_wrong) - set(indices_to_target['wrong'])
# 		assert len(diff) == 0, diff 
# 		indices_to_target['wrong'] = indices_to_wrong

# 	indices_to_selected_wrong = indices_to_target['wrong'] # target all of them 
# 	print ('Total number of wrongly processed input(s): {}'.format(len(indices_to_selected_wrong)))

# 	# check for correctly classified ones
# 	t1 = time.time()	
# 	indices_to_correct = indices_to_target['correct']

# 	t2 = time.time()
# 	# logging
# 	print ('Number of wrong: %d' % (len(indices_to_selected_wrong)))

# 	# extract the input vectors that are directly related to our target 
# 	# correct one first, followed by misclassified ones
# 	# FOR LFW, THIS WILL BE USED TO SLICE THE PRE-COMPUTE ATS
# 	new_indices_to_target = list(indices_to_correct) + list(indices_to_selected_wrong) 
# 	# set input for the searcher -> searcher will look only upon this input hereafter

# 	# extraction for predictions
# 	predictions = predictions[new_indices_to_target] # slice
# 	# extraction for data
# 	X = data_X[new_indices_to_target]
# 	y = data_y[new_indices_to_target]

# 	########### For logging & testing 
# 	num_of_our_target = len(new_indices_to_target)
# 	num_of_wrong = len(indices_to_selected_wrong)
# 	num_of_correct = len(indices_to_correct)

# 	print ("The number of our target:%d, (%d(correct), %d(wrong))" % (num_of_our_target, num_of_correct, num_of_wrong))
# 	# set new local indices to correct & wrong for the new predictions
# 	indices_to_correct = list(range(0, num_of_correct))
# 	indices_to_selected_wrong = list(range(num_of_correct, num_of_our_target))

# 	assert_msg = "%d + %d vs %d" % (len(indices_to_correct), len(indices_to_selected_wrong), num_of_our_target)
# 	assert len(indices_to_correct) + len(indices_to_selected_wrong) == num_of_our_target, assert_msg
# 	assert len(X) == num_of_our_target, "%d vs %d" % (len(X), num_of_our_target)
# 	assert len(predictions) == num_of_our_target, "%d vs %d" % (len(predictions), num_of_our_target)

# 	##
# 	from sklearn.preprocessing import Normalizer
# 	norm_scaler = Normalization(norm = "l1")

# 	model, _ = gen_frame_graph.build_keras_model_front_v2(path_to_keras_model, idx_to_target_w = -1)
# 	all_ws = model.trainable_weights
# 	ws = []
# 	for w in all_ws:
# 		if 'kernel' in w.name:
# 			ws.append(w)

# 	for idx_to_tl in np.arange(min_idx_to_tl, max_idx_to_tl + 1):
# 		# FI
# 		t_model, _ = gen_frame_graph.build_keras_model_front_v2(path_to_keras_model, idx_to_target_w = idx_to_tl)
# 		prev_output = t_model.predict(X)
		
# 		from_front = []
# 		weight_value_to_use = target_weights[idx_to_tl][0]
# 		for idx in range(weight_value_to_use.shape[-1]):
# 			#output_tensor = tf.math.multiply(prev_tensor, weight_tensor[:,idx])
# 			output = np.multiply(prev_output, weight_value_to_use[:,idx])
# 			#output_tensor = tf.math.abs(output_tensor)
# 			output = np.abs(output)
# 			#sum_tensor = tf.math.reduce_sum(output_tensor, axis = -1) #
# 			#sum_output = np.reduce_sum(output, axis = -1)
# 			# norm
# 			#output_tensor = tf.transpose(tf.div_no_nan(tf.transpose(output_tensor), sum_tensor))
# 			output = norm_scaler.fit_transform(output)
# 			#output_tensor = tf.math.reduce_mean(output_tensor, axis = 0) # compute an average for given inputs
# 			output = np.reduce_mean(output, axis = 0)
# 			#temp_tensors.append(output_tensor)
# 			from_front.append(output)
		
# 		from_front = np.asarray(from_front)
# 		from_front = from_front.T

# 		# behind
# 		tensor_grad = tf.gradients(
# 			model.output, 
# 			model.output,
# 			name = 'output_grad')

# 		# FI end

# 		# Gradient

# 		# G endd
