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
import tensorflow.keras.backend as K
from tqdm import tqdm

def get_target_weights(model, path_to_keras_model, indices_to_target = None, target_all = True):
	"""
	return indices to weight layers denoted by indices_to_target, or return all trainable layers
	"""
	import re

	# target only the layer with its class type in this list, but if target_all, then return all trainables
	targeting_clname_pattns = ['Dense*', 'Conv*'] #if not target_all else None
	is_target = lambda clname,targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])
	#if target_all: # some, like BatchNormalization has trainable weights. Not sure how to filter that out. so, comment out for the temporary use
	#	indices_to_target = None
		
	if model is None:
		assert path_to_keras_model is not None
		model = load_model(path_to_keras_model, compile=False)

	target_weights = {} # key = layer index, value: [weight value, layer name]
	if indices_to_target is not None:
		num_layers = len(model.layers)
		indices_to_target = [idx if idx >= 0 else num_layers + idx for idx in indices_to_target]

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

def compute_gradient_to_output(path_to_keras_model, idx_to_target_layer, X):
	"""
	compute gradients normalisesd and averaged for a given input X
	"""
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")
	
	model = load_model(path_to_keras_model, compile = False)
	target = model.layers[idx_to_target_layer].output

	tensor_grad = tf.gradients(
		model.output, 
		target,
		name = 'output_grad')

	gradient = K.get_session().run(tensor_grad, feed_dict={model.input: X})[0]
	print ("tensor grad", tensor_grad)
	print ("\t", gradient.shape)

	gradient = np.abs(gradient)
	#print ("Gradient", gradient)
	reshaped_gradient = gradient.reshape(gradient.shape[0],-1) # flatten
	norm_gradient = norm_scaler.fit_transform(reshaped_gradient) # normalised
	#print ("Norm", norm_gradient)
	#print (type(norm_gradient))
	print (norm_gradient.shape)
	mean_gradient = np.mean(norm_gradient, axis = 0) # compute mean for a given input
	#print ("Mean", mean_gradient, type(norm_gradient), norm_gradient.dtype)
	ret_gradient = mean_gradient.reshape(gradient.shape[1:]) # reshape to the orignal shape

	#K.clear_session()
	#s = tf.InteractiveSession()
	#K.set_session(s)
	reset_keras([tensor_grad])
	return ret_gradient 
			
def compute_gradient_to_loss(path_to_keras_model, idx_to_target_layer, X, y):
	"""
	compute gradients for the loss
	"""
	model = load_model(path_to_keras_model, compile = False)
	target = model.layers[idx_to_target_layer].weights[0]
	num_label = int(model.output.shape[-1])
	y_tensor = tf.placeholder(tf.float32, shape = [None, num_label], name = 'labels')

	loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
		logits = model.output, 
		labels = y_tensor, 
		name = "per_label_loss") 
	
	tensor_grad = tf.gradients(
		loss_tensor,
		target, 
		name = 'loss_grad')
	
	gradient = K.get_session().run(tensor_grad, feed_dict={model.input: X, y_tensor: y})[0]
	print ("tensor grad to loss", tensor_grad)
	print ("\t", gradient.shape)
	
	gradient = np.abs(gradient)
	#print ("Gradient", gradient)
	#K.clear_session()
	#s = tf.InteractiveSession()
	#K.set_session(s)
	reset_keras([gradient, loss_tensor, y_tensor])
	return gradient
	
def reset_keras(delete_list = None):
	if delete_list is None:
		K.clear_session()
		s = tf.InteractiveSession()
		K.set_session(s)
	else:
		import gc
		#sess = K.get_session()
		K.clear_session()
		#sess.close()
		sess = K.get_session()
		try:
			for d in delete_list:
				del d
		except:
			pass

		print(gc.collect()) # if it's done something you should see a number being outputted

		# use the same config as you used to create the session
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 1
		config.gpu_options.visible_device_list = "0"
		K.set_session(tf.Session(config=config))

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
		from collections.abc import Iterable
		if not isinstance(indices_to_target_layers, Iterable):
			indices_to_target_layers = [indices_to_target_layers]
	else: # target all, but only those that statisfy the predefined layer conditions
		indices_to_target_layers = None
	# get ranges
	# min_idx_to_tl = np.min(indices_to_target_layers); max_idx_to_tl = np.max(indices_to_target_layers)
	#target_weights = {}
	##for idx_to_tl in np.arange(min_idx_to_tl, max_idx_to_tl + 1):
	#for idx_to_tl in indices_to_target_layers:
	#	target_weights[idx_to_tl] = [kernel_and_bias_pairs[idx_to_tl]]

	model = load_model(path_to_keras_model, compile = False)
	target_weights = get_target_weights(model,
		path_to_keras_model, 
		indices_to_target = indices_to_target_layers, 
		target_all = target_all) # if target_all == True, then indices_to_target will be ignored

	print ('Total {} layers are targeted'.format(target_weights.keys()))
	#### HOW CAN WE KNOW WHICH LAYER IS PREDICTION LAYER and WEIGHT LAYER? => assumes they are given;;;
	# if not, then ... well everything becomes complicated
	# identify using print (l['name'], l['class_name']) ..? d['layers'] -> mdl.get_config()
	## -> at least for predc & corr_predc
	# compute prediction & corr_predictions
	predictions = model.predict(data_X)
	correct_predictions = np.argmax(predictions, axis = 1)
	correct_predictions = correct_predictions == np.argmax(data_y, axis = 1)
	#print ("Corr", correct_predictions[:10], correct_predictions.shape)
	#print (data_y[:10])
	#import sys; sys.exit()
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
	## slice inputs
	target_X = X[indices_to_selected_wrong]
	target_y = y[indices_to_selected_wrong]

	loc_start_time = time.time()
	##
	for idx_to_tl, vs in target_weights.items():
		t1 = time.time()
		t_w, lname = vs
		print ("targeting layer {} ({})".format(idx_to_tl, lname))
		#t_w_tensor = model.layers[idx_to_tl].weights[0]
		############ FI ############
		model = load_model(path_to_keras_model, compile = False)
		prev_output = model.layers[idx_to_tl - 1].output
		layer_config = model.layers[idx_to_tl].get_config() 

		# if this takes too long, then change to tensor and compute them using K (backend)
		if is_FC(lname):
			from_front = []
			#model = load_model(path_to_keras_model, compile = False)
			#t_w_tensor = model.layers[idx_to_tl].weights[0]
			t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
			prev_output = t_model.predict(target_X)
			#reset_keras([t_model])
			#prev_output = model.layers[idx_to_tl - 1].output	
			##
			#from_front_tensors = []
			for idx in tqdm(range(t_w.shape[-1])):
				assert int(prev_output.shape[-1]) == t_w.shape[0], "{} vs {}".format(
					int(prev_output.shape[-1]), t_w.shape[0])
					
				output = np.multiply(prev_output, t_w[:,idx]) # -> shape = prev_output.shape
				#output = tf.math.multiply(prev_output, t_w_tensor[:,idx]) # shape = prev_output.shape
				output = np.abs(output)
				#output = tf.math.abs(output)
				output = norm_scaler.fit_transform(output) # -> shape = prev_output.shape (normalisation on )
				#t_sum = tf.math.reduce_sum(output, axis = -1) 
				#output = tf.transpose(tf.div_no_nan(tf.transpose(output), t_sum))
				output = np.mean(output, axis = 0) # -> shape = (reshaped_t_w.shape[-1],)
				#output = tf.math.reduce_mean(output, axis = 0) #  
				#temp_tensors.append(output_tensor)
				#from_front_tensors.append(output) #
				from_front.append(output) 
			
			#outputs = K.get_session().run(from_front_tensors, feed_dict = {model.input: target_X})
			#print ("This?")
			##K.clear_session()
			##s = tf.InteractiveSession()
			##K.set_session(s)
			#reset_keras(from_front_tensors)
			#from_front = np.asarray(outputs).T
			# work for Dense. but, for the others?
			from_front = np.asarray(from_front)
			from_front = from_front.T
			print ('From front', from_front.shape)
			#print (np.sum(from_front, axis = 0))
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
			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X)
			print ("From behind", from_behind.shape)
			#print ("\t FR", from_front[:10])
			#print ("\t BH" , from_behind[:10])
			FIs = from_front * from_behind
			print ("Max: {}, min:{}".format(np.max(FIs), np.min(FIs)))
			############ FI end #########

			# Gradient
			grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y)
			print ("Vals", FIs.shape, grad_scndcr.shape)	
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

			#num_kernels = prev_output.shape[1] # Channel_in
			num_kernels = int(prev_output.shape[1]) # Channel_in
			assert num_kernels == t_w.shape[2], "{} vs {}".format(num_kernels, t_w.shape[2])

			# H x W				
			#input_shape = prev_output.shape[2:] # the last two (front two are # of inputs and # of kernels (Channel_in))
			input_shape = [int(v) for v in prev_output.shape[2:]] # the last two (front two are # of inputs and # of kernels (Channel_in))

			# (W1−F+2P)/S+1, W1 = input volumne , F = kernel, P = padding
			n_mv_0 = int((input_shape[0] - kernel_shape[0] + 2 * paddings[0])/strides[0] + 1) # H_out
			n_mv_1 = int((input_shape[1] - kernel_shape[1] + 2 * paddings[1])/strides[1] + 1) # W_out
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
			from_front = []
			#from_front_tensors = []
			# move axis for easier computation
			print ("range", n_output_channel, n_mv_0, n_mv_1) # currenlty taking too long -> change to compute on gpu
			#tr_prev_output = np.moveaxis(prev_output, [0,1,2,3], [0,3,1,2])
			#tr_prev_output = tf.transpose(prev_output, [0,2,3,1])
		

			## THINK ABOUT CHANGE THIS TO USE A PLACEHOLDER, AND THEREBY, GENERATING THE GRAPH ONLY THE ONCE 
			## WHICH REMOVE THE OVERHAEAD AND COMPLEXITY FROM GENERATING OPERATIONS AGAIN AND AGAIN AND 
			## CLAER THEM TO REDUCE THE TIME 
			## --> THEN THEN QUESTION WOULD BE WHICH TENSOR WOULD BE THE STARTING 
			for idx_ol in tqdm(range(n_output_channel)): # t_w.shape[-1]
				#print ("All nodes", [n.name for n in tf.get_default_graph().as_graph_def().node])
				l_from_front_tensors = []
				t0 = time.time()
				t1 = time.time()
				# due to clear_session()
				model = load_model(path_to_keras_model, compile = False)
				t2 = time.time()
				print ("Time for loading a model: {}".format(t2 - t1))
				#t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
				prev_output = model.layers[idx_to_tl - 1].output
				tr_prev_output = tf.transpose(prev_output, [0,2,3,1])
				#print ("All nodes", [n.name for n in tf.get_default_graph().as_graph_def().node])
				#t_w_tensor = model.layers[idx_to_tl].weights[0]
				
				t1 = time.time()	
				for i in range(n_mv_0): # H
					indices_to_k1 = np.arange(i*strides[0], i*strides[0]+kernel_shape[0], 1)
					for j in range(n_mv_1): # W
						indices_to_k2 = np.arange(j*strides[1], j*strides[1]+kernel_shape[1], 1)
						#curr_prev_output = tr_prev_output[:,indices_to_k1,:,:][:,:,indices_to_k2,:]
						curr_prev_output = tr_prev_output[:,i*strides[0]:i*strides[0]+kernel_shape[0],:,:][:,:,j*strides[1]:j*strides[1]+kernel_shape[1],:]	
						#print (curr_prev_output)
						output = curr_prev_output * t_w[:,:,:,idx_ol]
						#output = np.abs(output)
						output = tf.math.abs(output)
						#print (output)
						#print (output.shape)
						if k == 0:
							#print ("output", output.shape)
							print ("output", [int(v) for v in output.shape[1:]])
						
						#sum_output = np.sum(output) # since they are all added to compute a single output tensor
						sum_output = tf.math.reduce_sum(output) # since they are all added to compute a single output tensor
						#output = output/sum_output # normalise -> [# input, F1, F2, Channel_in]
						output = tf.div_no_nan(output, sum_output) # normalise -> [# input, F1, F2, Channel_in]
						#output = np.mean(output, axis = 0) # sum over a given input set # [F1, F2, Channel_in]
						output = tf.math.reduce_mean(output, axis = 0) # sum over a given input set # [F1, F2, Channel_in]
						if k == 0:
							print ('mean', [int(v) for v in output.shape[1:]], "should be", (kernel_shape[0],kernel_shape[1],prev_output.shape[1]))
							k += 1
						#from_front[(idx_ol, i, j)] = output # output -> []
						#output_v = K.get_session().run(output, feed_dict = {model.input: target_X})[0]
						#from_#front.append(output)#_v)
						l_from_front_tensors.append(output)
				
				t2 = time.time()
				print ("Time for generating tensors: {}".format(t2 - t1))
				t1 = time.time()
				outputs = K.get_session().run(l_from_front_tensors, feed_dict = {model.input: target_X})
				reset_keras([l_from_front_tensors] + [model])
				#outputs = sess.run(l_from_front_tensors, feed_dict = {model.input: target_X})
				t2 = time.time()
				print ("Time for computing: {}".format(t2 - t1))
				print ("Total time: {}".format(t2 - t0))
				from_front.extend(outputs)
			###############################################################################################################
			###############################################################################################################
			
			#outputs = K.get_session().run(from_front_tensors, feed_dict = {model.input: target_X})
			#from_front = np.asarray(outputs)	
			from_front = np.asarray(from_front)
			print ("From front", from_front.shape) # [Channel_out * n_mv_0 * n_mv_1, F1, F2, Channel_in]
			from_front = from_front.reshape(
				(n_output_channel,n_mv_0,n_mv_1,kernel_shape[0],kernel_shape[1],int(prev_output.shape[1])))
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
			#
			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X) # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
			print ("From behind", from_behind.shape)
			#from_behind = from_behind.reshape(-1,) # [Channel_out * n_mv_0 * n_mv_1,]
			
			t1 = time.time()
			FIs = from_front * from_behind # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]
			t2 = time.time()
			print ('Time for multiplying front and behind results: {}'.format(t2 - t1))
			#FIs = np.mean(np.mean(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
			FIs = np.sum(np.sum(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
			t3 = time.time()
			print ('Time for computing mean for FIs: {}'.format(t3 - t2))
			## Gradient
			# will be [F1, F2, Channel_in, Channel_out]
			grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y)
			# ##		
		else:
			print ("Currenlty not supported: {}. (shoulde be filtered before)".format(lname))		
			import sys; sys.exit()

		t2 = time.time()
		print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
		####
		pairs = np.asarray([grad_scndcr.flatten(), FIs.flatten()]).T
		#print ("Pairs", pairs.shape)
		total_cands[idx_to_tl] = {'shape':FIs.shape, 'costs':pairs}
		#sess = K.get_session()
		#sess.close()
	
	t3 = time.time()
	print ("Time for computing total costs: {}".format(t3 - t0))

	# compute pareto front
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = [([idx_to_tl, local_i], c) for idx_to_tl in indices_to_tl for local_i,c in enumerate(total_cands[idx_to_tl]['costs'])]
	costs = np.asarray([vs[1] for vs in costs_and_keys])
	#print (costs_and_keys[0])
	#print (costs[0])
	#print (costs[:10], costs[-10:])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))
	#print (total_cands)

	# a list of [index to the target layer, index to a neural weight]
	indices_to_nodes = [[vs[0][0], np.unravel_index(vs[0][1], total_cands[vs[0][0]]['shape'])] for vs in costs_and_keys]
	#print (indices_to_nodes[0], indices_to_nodes[-1])
	#print (len(costs), len(indices_to_nodes), len(costs_and_keys))
	#print ("Cost", costs[:20], indices_to_nodes[:20]) 

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
	#pareto_front = [[int(idx_to_tl), [int(v) for v in inner_indices.split(",")]] for idx_to_tl,inner_indices in pareto_front]
	##
	t5 = time.time()
	print ("Time for computing the pareto front: {}".format(t5 - t4))
	loc_end_time = time.time()
	print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))
	return pareto_front, costs_and_keys

def generate_FI_tensor_cnn(t_w_v, prev_output_v):
	"""
	kernel == filter
	"""
	#t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
	#prev_output_v = t_model.predict(X)

	# prev_output shape = [# of inputs, Channel_in, H, W]
	#tr_prev_output = tf.transpose(prev_output, [0,2,3,1])  # [# of inputs. H, W, Channel_in]
	# shape of curr_prev_output: [:,kernel_shape[0],kerne_shape[1],Channel_in]
	#curr_prev_output = tr_prev_output[:,i*strides[0]:i*strides[0]+kernel_shape[0],:,:][:,:,j*strides[1]:j*strides +kernel_shape[1],:]
	#curr_prev_output_as_input_t = tf.placeholder(tf.float32, [None, t_w_v.shape[-2], kernel_shape[0], kernel_shape[1]]) 

	t_w_t = tf.constant(t_w_v, dtype = tf.float32)
	prev_output = tf.constant(prev_output_v, dtype = tf.float32) # 
	tr_prev_output = tf.transpose(prev_output, [0,2,3,1]) # to here, take 0.015 ...? meaning the problem exist on front ..?
	
	# this would be for gathering
	indices_to_k1 = tf.placeholder(tf.int32, shape = (None,)) 
	indices_to_k2 = tf.placeholder(tf.int32, shape = (None,))
	curr_prev_output_slice = tf.gather(tr_prev_output, indices_to_k1, axis = 1)
	# curr_prev_output_slice = prev_output[:,indices_to_k1]
	curr_prev_output_slice = tf.gather(curr_prev_output_slice, indices_to_k2, axis = 2) # take almost 0.018 to this

	idx_to_out_channel = tf.placeholder(tf.int32, shape =())
	curr_t_w_as_input_t = t_w_t[:,:,:,idx_to_out_channel]

	output = curr_prev_output_slice * curr_t_w_as_input_t # output = curr_prev_output * t_w[:,:,:,idx_ol]
	output = tf.math.abs(output)
	sum_output = tf.math.reduce_sum(output)
	output = tf.div_no_nan(output, sum_output)
	output = tf.math.reduce_mean(output, axis = 0)

	return {'output':output, 'temp':prev_output, 'indices_to_k1':indices_to_k1, 'indices_to_k2':indices_to_k2, 'idx_to_out_channel':idx_to_out_channel}

def generate_FI_tensor_cnn_v2(t_w_v):
	"""
	kernel == filter
	"""
	#t_w_t = tf.constant(t_w_v, dtype = tf.float32)
	t_w_t = tf.placeholder(tf.float32, shape = (t_w_v.shape[0], t_w_v.shape[1], t_w_v.shape[2]))
	#prev_output = tf.constant(prev_output_v, dtype = tf.float32) # 
	#tr_prev_output = tf.transpose(prev_output, [0,2,3,1]) # to here, take 0.015 ...? meaning the problem exist on front ..?
	
	# this would be for gathering
	#indices_to_k1 = tf.placeholder(tf.int32, shape = (None,)
	#indices_to_k2 = tf.placeholder(tf.int32, shape = (None,))
	#curr_prev_output_slice = tf.gather(tr_prev_output, indices_to_k1, axis = 1)
	# curr_prev_output_slice = prev_output[:,indices_to_k1]
	#curr_prev_output_slice = tf.gather(curr_prev_output_slice, indices_to_k2, axis = 2) # take almost 0.018 to this
	curr_prev_output_slice = tf.placeholder(tf.float32, shape = (None, t_w_v.shape[0], t_w_v.shape[1], t_w_v.shape[2]))
	
	idx_to_out_channel = tf.placeholder(tf.int32, shape =())
	#curr_t_w_as_input_t = t_w_t[:,:,:,idx_to_out_channel]

	output = curr_prev_output_slice * t_w_t #curr_t_w_as_input_t # output = curr_prev_output * t_w[:,:,:,idx_ol]
	output = tf.math.abs(output)
	sum_output = tf.math.reduce_sum(output)
	output = tf.div_no_nan(output, sum_output)
	output = tf.math.reduce_mean(output, axis = 0)

	return {'output':output, 'curr_prev_output_slice':curr_prev_output_slice, 't_w_t':t_w_t, 'idx_to_out_channel':idx_to_out_channel}


def localise_offline_v2(
	X, y,
	indices_to_selected_wrong,
	target_weights,
	path_to_keras_model = None):
	"""
	localise offline
	"""

	## Now, start localisation !!! ##
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")

	total_cands = {}
	FIs = None; grad_scndcr = None

	print ('Total {} layers are targeted'.format(len(target_weights)))
	t0 = time.time()
	## slice inputs
	target_X = X[indices_to_selected_wrong]
	target_y = y[indices_to_selected_wrong]

	loc_start_time = time.time()
	##
	for idx_to_tl, vs in target_weights.items():
		t1 = time.time()
		t_w, lname = vs
		print ("targeting layer {} ({})".format(idx_to_tl, lname))
		#t_w_tensor = model.layers[idx_to_tl].weights[0]
		############ FI ############
		model = load_model(path_to_keras_model, compile = False)
		if idx_to_tl == 0: # meaning the model doesn't specify the input layer explicitly
			prev_output = target_X
		else:
			prev_output = model.layers[idx_to_tl - 1].output
		layer_config = model.layers[idx_to_tl].get_config() 

		# if this takes too long, then change to tensor and compute them using K (backend)
		if is_FC(lname):
			from_front = []
			#model = load_model(path_to_keras_model, compile = False)
			#t_w_tensor = model.layers[idx_to_tl].weights[0]
			if idx_to_tl == 0 or idx_to_tl - 1 == 0:
				prev_output = target_X
			else:
				t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
				prev_output = t_model.predict(target_X)
			#prev_output = model.layers[idx_to_tl - 1].output	
			##
			#from_front_tensors = []
			for idx in tqdm(range(t_w.shape[-1])):
				assert int(prev_output.shape[-1]) == t_w.shape[0], "{} vs {}".format(
					int(prev_output.shape[-1]), t_w.shape[0])
					
				output = np.multiply(prev_output, t_w[:,idx]) # -> shape = prev_output.shape
				#output = tf.math.multiply(prev_output, t_w_tensor[:,idx]) # shape = prev_output.shape
				output = np.abs(output)
				#output = tf.math.abs(output)
				output = norm_scaler.fit_transform(output) # -> shape = prev_output.shape (normalisation on )
				#t_sum = tf.math.reduce_sum(output, axis = -1) 
				#output = tf.transpose(tf.div_no_nan(tf.transpose(output), t_sum))
				output = np.mean(output, axis = 0) # -> shape = (reshaped_t_w.shape[-1],)
				#output = tf.math.reduce_mean(output, axis = 0) #  
				#temp_tensors.append(output_tensor)
				#from_front_tensors.append(output) #
				from_front.append(output) 
			
			from_front = np.asarray(from_front)
			from_front = from_front.T
			print ('From front', from_front.shape)
			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X)
			print ("From behind", from_behind.shape)
			#print ("\t FR", from_front[:10])
			#print ("\t BH" , from_behind[:10])
			FIs = from_front * from_behind
			print ("Max: {}, min:{}".format(np.max(FIs), np.min(FIs)))
			############ FI end #########

			# Gradient
			grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y)
			print ("Vals", FIs.shape, grad_scndcr.shape)	
			# G end
		elif is_C2D(lname):
			kernel_shape = t_w.shape[:2] # kernel == filter
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

			#num_kernels = prev_output.shape[1] # Channel_in
			num_kernels = int(prev_output.shape[1]) # Channel_in
			assert num_kernels == t_w.shape[2], "{} vs {}".format(num_kernels, t_w.shape[2])

			# H x W				
			#input_shape = prev_output.shape[2:] # the last two (front two are # of inputs and # of kernels (Channel_in))
			input_shape = [int(v) for v in prev_output.shape[2:]] # the last two (front two are # of inputs and # of kernels (Channel_in))

			# (W1−F+2P)/S+1, W1 = input volumne , F = kernel, P = padding
			n_mv_0 = int((input_shape[0] - kernel_shape[0] + 2 * paddings[0])/strides[0] + 1) # H_out
			n_mv_1 = int((input_shape[1] - kernel_shape[1] + 2 * paddings[1])/strides[1] + 1) # W_out

			k = 0 # for logging
			n_output_channel = t_w.shape[-1] # Channel_out
			from_front = []
			#from_front_tensors = []
			# move axis for easier computation
			print ("range", n_output_channel, n_mv_0, n_mv_1) # currenlty taking too long -> change to compute on gpu
		
			# for idx_ol in tqdm(range(n_output_channel)): # t_w.shape[-1]
			# 	#print ("All nodes", [n.name for n in tf.get_default_graph().as_graph_def().node])
			# 	l_from_front_tensors = []
			# 	t0 = time.time()
			# 	t1 = time.time()
			# 	# due to clear_session()
			# 	model = load_model(path_to_keras_model, compile = False)
			# 	t2 = time.time()
			# 	print ("Time for loading a model: {}".format(t2 - t1))
			# 	#t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
			# 	prev_output = model.layers[idx_to_tl - 1].output
			# 	tr_prev_output = tf.transpose(prev_output, [0,2,3,1])
			# 	#print ("All nodes", [n.name for n in tf.get_default_graph().as_graph_def().node])
			# 	#t_w_tensor = model.layers[idx_to_tl].weights[0]

			# 	t1 = time.time()	
			# 	for i in range(n_mv_0): # H
			# 		indices_to_k1 = np.arange(i*strides[0], i*strides[0]+kernel_shape[0], 1)
			# 		for j in range(n_mv_1): # W
			# 			indices_to_k2 = np.arange(j*strides[1], j*strides[1]+kernel_shape[1], 1)
			# 			#curr_prev_output = tr_prev_output[:,indices_to_k1,:,:][:,:,indices_to_k2,:]
			# 			curr_prev_output = tr_prev_output[:,i*strides[0]:i*strides[0]+kernel_shape[0],:,:][:,:,j*strides[1]:j*strides[1]+kernel_shape[1],:]	
			# 			#print (curr_prev_output)
			# 			output = curr_prev_output * t_w[:,:,:,idx_ol]
			# 			#output = np.abs(output)
			# 			output = tf.math.abs(output)
			# 			#print (output)
			# 			#print (output.shape)
			# 			if k == 0:
			# 				#print ("output", output.shape)
			# 				print ("output", [int(v) for v in output.shape[1:]])
						
			# 			#sum_output = np.sum(output) # since they are all added to compute a single output tensor
			# 			sum_output = tf.math.reduce_sum(output) # since they are all added to compute a single output tensor
			# 			#output = output/sum_output # normalise -> [# input, F1, F2, Channel_in]
			# 			output = tf.div_no_nan(output, sum_output) # normalise -> [# input, F1, F2, Channel_in]
			# 			#output = np.mean(output, axis = 0) # sum over a given input set # [F1, F2, Channel_in]
			# 			output = tf.math.reduce_mean(output, axis = 0) # sum over a given input set # [F1, F2, Channel_in]
			# 			if k == 0:
			# 				print ('mean', [int(v) for v in output.shape[1:]], "should be", (kernel_shape[0],kernel_shape[1],prev_output.shape[1]))
			# 				k += 1
			# 			#from_front[(idx_ol, i, j)] = output # output -> []
			# 			#output_v = K.get_session().run(output, feed_dict = {model.input: target_X})[0]
			# 			#from_#front.append(output)#_v)
			# 			l_from_front_tensors.append(output)
				
			# 	t2 = time.time()
			# 	print ("Time for generating tensors: {}".format(t2 - t1))
			# 	t1 = time.time()
			# 	outputs = K.get_session().run(l_from_front_tensors, feed_dict = {model.input: target_X})
			# 	reset_keras([l_from_front_tensors] + [model])
			# 	#outputs = sess.run(l_from_front_tensors, feed_dict = {model.input: target_X})
			# 	t2 = time.time()
			# 	print ("Time for computing: {}".format(t2 - t1))
			# 	print ("Total time: {}".format(t2 - t0))
			# 	from_front.extend(outputs)
			
			###############
			if idx_to_tl == 0 or idx_to_tl - 1 == 0:
				prev_output_v = target_X
			else:
				t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
				prev_output_v = t_model.predict(target_X)
			tr_prev_output_v = np.moveaxis(prev_output_v, [1,2,3],[3,1,2])
			#tensors_for_FI = generate_FI_tensor_cnn(t_w, prev_output_v)
			tensors_for_FI = generate_FI_tensor_cnn_v2(t_w)
			
			for idx_ol in tqdm(range(n_output_channel)): # t_w.shape[-1]
				for i in range(n_mv_0): # H
					for j in range(n_mv_1): # W
						curr_prev_output_slice = tr_prev_output_v[:,i*strides[0]:i*strides[0]+kernel_shape[0],:,:]
						curr_prev_output_slice = curr_prev_output_slice[:,:,j*strides[1]:j*strides[1]+kernel_shape[1],:]
						
						output = curr_prev_output_slice * t_w[:,:,:,idx_ol] 
						output = np.abs(output)
						sum_output = np.nan_to_num(np.sum(output), posinf = 0.)
						output = output/sum_output
						output = np.mean(output, axis = 0) 
						
						from_front.append(output)
			##############
			
			#outputs = K.get_session().run(from_front_tensors, feed_dict = {model.input: target_X})
			#from_front = np.asarray(outputs)	
			from_front = np.asarray(from_front)
			print ("From front", from_front.shape) # [Channel_out * n_mv_0 * n_mv_1, F1, F2, Channel_in]
			from_front = from_front.reshape(
				(n_output_channel,n_mv_0,n_mv_1,kernel_shape[0],kernel_shape[1],int(prev_output.shape[1])))
			print ("reshaped", from_front.shape)
			from_front = np.moveaxis(from_front, [0,1,2], [3,4,5])
			print ("axis moved", from_front.shape) # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]

			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X) # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
			print ("From behind", from_behind.shape)
			#from_behind = from_behind.reshape(-1,) # [Channel_out * n_mv_0 * n_mv_1,]
			
			t1 = time.time()
			FIs = from_front * from_behind # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]
			t2 = time.time()
			print ('Time for multiplying front and behind results: {}'.format(t2 - t1))
			#FIs = np.mean(np.mean(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
			FIs = np.sum(np.sum(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
			t3 = time.time()
			print ('Time for computing mean for FIs: {}'.format(t3 - t2))
			## Gradient
			# will be [F1, F2, Channel_in, Channel_out]
			grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y)
			# ##		
		else:
			print ("Currenlty not supported: {}. (shoulde be filtered before)".format(lname))		
			import sys; sys.exit()

		t2 = time.time()
		print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
		####
		pairs = np.asarray([grad_scndcr.flatten(), FIs.flatten()]).T
		#print ("Pairs", pairs.shape)
		total_cands[idx_to_tl] = {'shape':FIs.shape, 'costs':pairs}
		#sess = K.get_session()
		#sess.close()
	
	t3 = time.time()
	print ("Time for computing total costs: {}".format(t3 - t0))

	# compute pareto front
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = [([idx_to_tl, local_i], c) for idx_to_tl in indices_to_tl for local_i,c in enumerate(total_cands[idx_to_tl]['costs'])]
	costs = np.asarray([vs[1] for vs in costs_and_keys])
	#print (costs_and_keys[0])
	#print (costs[0])
	#print (costs[:10], costs[-10:])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))
	#print (total_cands)

	# a list of [index to the target layer, index to a neural weight]
	indices_to_nodes = [[vs[0][0], np.unravel_index(vs[0][1], total_cands[vs[0][0]]['shape'])] for vs in costs_and_keys]
	#print (indices_to_nodes[0], indices_to_nodes[-1])
	#print (len(costs), len(indices_to_nodes), len(costs_and_keys))
	#print ("Cost", costs[:20], indices_to_nodes[:20]) 

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
	#pareto_front = [[int(idx_to_tl), [int(v) for v in inner_indices.split(",")]] for idx_to_tl,inner_indices in pareto_front]
	##
	t5 = time.time()
	print ("Time for computing the pareto front: {}".format(t5 - t4))
	loc_end_time = time.time()
	print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))
	return pareto_front, costs_and_keys


def localise_by_gradient(
	X, y,
	indices_to_selected_wrong,
	target_weights,
	path_to_keras_model = None):
	"""
	localise offline
	"""
	total_cands = {}

	print ('Total {} layers are targeted'.format(len(target_weights)))
	t0 = time.time()
	## slice inputs
	target_X = X[indices_to_selected_wrong]
	target_y = y[indices_to_selected_wrong]

	loc_start_time = time.time()
	##
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print ("targeting layer {} ({})".format(idx_to_tl, lname))
		
		t1 = time.time()
		grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y)
		t2 = time.time()

		print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
		assert t_w.shape == grad_scndcr.shape, "{} vs {}".format(t_w.shape, grad_scndcr.shape)

		total_cands[idx_to_tl] = {'shape':grad_scndcr.shape, 'costs':grad_scndcr.flatten()}
	
	t3 = time.time()
	print ("Time for computing total costs: {}".format(t3 - t0))

	# compute pareto front
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = [([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
		for idx_to_tl in indices_to_tl 
		for local_i,c in enumerate(total_cands[idx_to_tl]['costs'])]
	
	costs = np.asarray([vs[1] for vs in costs_and_keys])
	#print (costs_and_keys[0])
	#print (costs[0])
	#print (costs[:10], costs[-10:])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))
	#print (total_cands)

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	loc_end_time = time.time()
	print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))

	return sorted_costs_and_keys


def localise_by_random_selection(number_of_place_to_fix, target_weights):
	"""
	randomly select places to fix
	"""
	
	total_indices = []
	for idx_to_tl, vs in target_weights.items():
		t_w, _ = vs
		l_indices = list(np.ndindex(t_w.shape))
		total_indices.extend(list(zip([idx_to_tl] * len(l_indices), l_indices)))
	
	if number_of_place_to_fix > 0 and number_of_place_to_fix < len(total_indices):
		selected_indices = np.random.choice(np.arange(len(total_indices)), number_of_place_to_fix, replace = False)
		indices_to_places_to_fix = [total_indices[idx] for idx in selected_indices]
	else:
		indices_to_places_to_fix = total_indices
	
	
	return indices_to_places_to_fix
