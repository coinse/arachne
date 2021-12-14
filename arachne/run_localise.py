"""
Localise faults in offline for any faults
"""
from re import A
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
from tqdm import tqdm
import lstm_layer

def get_target_weights(model, path_to_keras_model, indices_to_target = None, target_all = True):
	"""
	return indices to weight layers denoted by indices_to_target, or return all trainable layers
	"""
	import re

	# target only the layer with its class type in this list, but if target_all, then return all trainables
	targeting_clname_pattns = ['Dense*', 'Conv*', 'LSTM*'] #if not target_all else None
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
					if is_FC(class_name) or is_C2D(class_name):
						target_weights[i] = [ws[0], type(layer).__name__]
					elif is_LSTM(class_name): # for LSTM, even without bias, a fault can be in the weights of the kernel or the recurrent kernel (hidden state handling)
						assert len(ws) == 3, ws
						target_weights[i] = [ws[:-1], type(layer).__name__] # index 0: for the kernel, index 1: for the recurrent kernel
					else:
						print ("{} not supported yet".format(class_name))
						assert False

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

def is_LSTM(lname):
	"""
	"""
	import re
	pattns = ['LSTM']
	return any([bool(re.match(t,lname)) for t in pattns])

def is_Attention(lname):
	"""
	"""
	import re
	pattns = ['LSTM']
	return any([bool(re.match(t,lname)) for t in pattns])


def compute_gradient_to_output(path_to_keras_model, idx_to_target_layer, X, by_batch = False, on_weight = False, wo_reset = False):
	"""
	compute gradients normalisesd and averaged for a given input X
	on_weight = False -> on output of idx_to_target_layer'th layer
	"""
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")
		
	model = load_model(path_to_keras_model, compile = False)
	target = model.layers[idx_to_target_layer].output if not on_weight else model.layers[idx_to_target_layer].weights[0]
	print ("Target", target)	
	tensor_grad = tf.gradients(
		model.output, 
		target,
		name = 'output_grad')

	print (tensor_grad)
	# since this might cause OOM error, divide them 
	num = X.shape[0]
	if by_batch:
		batch_size = 64 
		num_split = int(np.round(num/batch_size))
		if num_split == 0:
			num_split = 1
		chunks = np.array_split(np.arange(num), num_split)
	else:
		chunks = [np.arange(num)]


	if not on_weight:	
		grad_shape = tuple([num] + [int(v) for v in tensor_grad[0].shape[1:]])
		print ("Grad shape", grad_shape)
		print (tensor_grad)	
		gradient = np.zeros(grad_shape)
		#fn = K.function([model.input], tensor_grad)
		for chunk in chunks:
			_gradient = K.get_session().run(tensor_grad, feed_dict={model.input: X[chunk]})[0]
			#_gradient = fn([X[chunk]])[0]	
			#print ("tensor grad", tensor_grad, X.shape)
			#print ("\t", _gradient.shape)
			gradient[chunk] = _gradient
	
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
	else: # on a weight variable
		#gradients = []
		gradient = None
		for chunk in chunks:
			_gradient = K.get_session().run(tensor_grad, feed_dict={model.input: X[chunk]})[0]
			if gradient is None:
				gradient = _gradient
			else:
				gradient += _gradient
			#gradients.append(_gradient)

		## due to memory error
		#gradient = np.sum(np.asarray(gradients), axis = 0)
		ret_gradient = np.abs(gradient)
		
	#K.clear_session()
	#s = tf.InteractiveSession()
	#K.set_session(s)
	if not wo_reset:
		reset_keras([tensor_grad])
	return ret_gradient 
			
			
def compute_gradient_to_loss(path_to_keras_model, idx_to_target_layer, X, y, 
	target = None, by_batch = False, wo_reset = False, loss_func = 'softmax', **kwargs):
	"""
	compute gradients for the loss. 
	kwargs contains the key-word argumenets required for the loss funation
	"""
	model = load_model(path_to_keras_model, compile = False)
	if target is None:
		target = model.layers[idx_to_target_layer].weights[0] 
	num_label = int(model.output.shape[-1])
	y_tensor = tf.placeholder(tf.float32, shape = [None, num_label], name = 'labels')

	## should be fixed!!! -> to use the activation funciton of the model!!! -> sepcifiy or accept as the argument
	if loss_func == 'softmax':
		# might be changed as the following two
		loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
			logits = model.output, 
			labels = y_tensor, 
			name = "per_label_loss") 
	elif loss_func == 'binary_crossentropy':
		if 'name' in kwargs.keys():
			kwargs.pop("name")
		loss_tensor = tf.keras.losses.BinaryCrossentropy(name = "per_label_loss")
		loss_tensor.__dict__.update(kwargs)
	elif loss_func in ['mean_squared_error', 'mse']:
		if 'name' in kwargs.keys():
			kwargs.pop("name")
		loss_tensor = tf.keras.losses.MeanSquaredError(name = "per_label_loss")
		loss_tensor.__dict__.update(kwargs)
	else:
		print ("{} not supported yet".format())

	tensor_grad = tf.gradients(
		loss_tensor,
		target, 
		name = 'loss_grad')

	# since this might cause OOM error, divide them 
	num = X.shape[0]
	if by_batch:
		batch_size = 64
		num_split = int(np.round(num/batch_size))
		if num_split == 0:
			num_split += 1
		chunks = np.array_split(np.arange(num), num_split)
	else:
		chunks = [np.arange(num)]
	
	gradients = []
	for chunk in chunks:
		_gradient = K.get_session().run(tensor_grad, feed_dict={model.input: X[chunk], y_tensor: y[chunk]})[0]
		gradients.append(_gradient)
		#print ("tensor grad", tensor_grad)
		#print ("\t", _gradient.shape)

	#print (np.asarray(gradients).shape)
	gradient = np.sum(np.asarray(gradients), axis = 0)
	#print (gradient[:10])
	#import sys; sys.exit()	
	gradient = np.abs(gradient)
	if not wo_reset:
		reset_keras([gradient, loss_tensor, y_tensor])

	return gradient
	

def reset_keras(delete_list = None, frac = 1):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = frac)
	config = tf.ConfigProto(gpu_options=gpu_options)
	#config.gpu_options.visible_device_list = "0"

	if delete_list is None:
		K.clear_session()
		s = tf.InteractiveSession(config = config)
		K.set_session(s)
	else:
		import gc
		K.clear_session()
		try:
			for d in delete_list:
				del d
		except:
			pass

		gc.collect() # if it's done something you should see a number being outputted

		# use the same config as you used to create the session
		K.set_session(tf.Session(config = config))


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


def sample_input_for_loc_by_rd(
	indices_to_chgd, 
	indices_to_unchgd,
	predictions = None, ys = None):
	"""
	By default, we assume the random sampling is enough to reflect the initial prediction distributiuon of indices_to_unchgd
	"""
	num_chgd = len(indices_to_chgd)
	if num_chgd >= len(indices_to_unchgd): # no need to do any sampling
		return indices_to_chgd, indices_to_unchgd
	
	if predictions is None and ys is None:
		sampled_indices_to_unchgd = np.random.choice(indices_to_unchgd, num_chgd, replace = False)	
		return indices_to_chgd, sampled_indices_to_unchgd
	else:
		_, sampled_indices_to_unchgd = sample_input_for_loc_sophis(
			indices_to_chgd, 
			indices_to_unchgd, 
			predictions, ys)

		return indices_to_chgd, sampled_indices_to_unchgd


def sample_input_for_loc_sophis(
	indices_to_chgd, 
	indices_to_unchgd, 
	predictions, ys):
	"""
	prediction -> model ouput. Right before outputing as the final classification result 
		from 0~len(indices_to_unchgd)-1, the results of unchagned
		from len(indices_to_unchgd)~end, the results of changed 
	## in auto_patch.patch: new_indices_to_target = list(indices_to_correct) + list(indices_to_selected_wrong) 
	sample the indices to changed and unchanged behaviour later used for localisation 
	"""
	pred_labels = np.argmax(predictions, axis = 1)
	y_labels = np.argmax(ys, axis = 1)
	_indices = np.zeros(len(indices_to_unchgd) + len(indices_to_chgd))
	_indices[:len(indices_to_unchgd)] = indices_to_unchgd
	_indices[len(indices_to_unchgd):] = indices_to_chgd
	
	###  checking  ###
	_indices_to_unchgd = np.where(pred_labels == y_labels)[0]; _indices_to_unchgd.sort()
	indices_to_unchgd = np.asarray(indices_to_unchgd); indices_to_unchgd.sort()
	_indices_to_chgd = np.where(pred_labels != y_labels)[0]; _indices_to_chgd.sort()
	indices_to_chgd = np.asarray(indices_to_chgd); indices_to_chgd.sort()
	assert all(indices_to_unchgd == _indices[_indices_to_unchgd])
	assert all(indices_to_chgd == np.sort(_indices[_indices_to_chgd]))
	###  checking end  ###

	# here, only the labels of the ys (original labels) are considered
	uniq_labels = np.unique(y_labels[_indices_to_unchgd]); uniq_labels.sort()
	grouped_by_label = {uniq_label:[] for uniq_label in uniq_labels}
	for idx in _indices_to_unchgd:	
		pred_label = pred_labels[idx]
		grouped_by_label[pred_label].append(idx)

	#print ("Sampled", [(k, len(vs)) for k,vs in grouped_by_label.items()])
	num_unchgd = len(indices_to_unchgd)
	num_chgd = len(indices_to_chgd)
	sampled_indices_to_unchgd = []
	num_total_sampled = 0
	for uniq_label,vs in grouped_by_label.items():
		num_sample = int(np.round(num_chgd * len(vs)/num_unchgd))
		print ("++", num_sample, len(vs)/num_unchgd)
		if num_sample <= 0:
			num_sample = 1
		
		if num_sample > len(vs):
			num_sample = len(vs)

		sampled_indices_to_unchgd.extend(list(np.random.choice(vs, num_sample, replace = False)))
		num_total_sampled += num_sample

	print ("Total number of sampled: {}".format(num_total_sampled))
	return indices_to_chgd, sampled_indices_to_unchgd


def compute_FI_and_GL(
	X, y,
	indices_to_target,
	target_weights,
	loss_funcs = None, 
	path_to_keras_model = None):
	"""
	compute FL and GL for the given inputs
	"""

	## Now, start localisation !!! ##
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")

	total_cands = {}
	FIs = None; grad_scndcr = None

	print ('Total {} layers are targeted'.format(len(target_weights)))
	t0 = time.time()
	## slice inputs
	print (X.shape, indices_to_target)
	print (X.shape, np.max(indices_to_target), len(indices_to_target))
	target_X = X[indices_to_target]
	target_y = y[indices_to_target]
	
	model = None
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
			#
			if len(prev_output.shape) == 3:
				prev_output = prev_output.reshape(prev_output.shape[0], prev_output.shape[-1])
			#
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
			loss_func = loss_funcs[idx_to_tl] if loss_funcs[idx_to_tl] is not None else 'softmax'
			grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y, loss_func=loss_func)
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

			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X, by_batch = True) # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
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
			loss_func = loss_funcs[idx_to_tl] if loss_funcs[idx_to_tl] is not None else 'softmax'
			grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y, by_batch = True, loss_func=loss_func)
			# ##	
		elif is_LSTM(lname): #
			from scipy.special import expit as sigmoid
			# ********** the hidden states !!! -> must be computed here -> make a temporary LSTM model, compute the sequences of 
			# hidden states and cell states, so, we can compute the impact at once (otherwise, we have to iterate over, which will be costly) 
			# after embedding layer ... so, here, we expect the input to be already formated to (batch_size, time_steps, input_feature_size)
			##############################################
			# should compute from_front and from_behind ##
			##############################################
			num_weights = 2 
			assert len(t_w) == num_weights, t_w
			t_w_kernel, t_w_recurr_kernel = t_w # t_w_kernel: (input_feature_size, 4 * num_units). t_w_recurr_kernel: (num_units, 4 * num_units)
			
			# get the previous output, which will be the input of the lstm
			t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output) # shape = (batch_size, time_steps, input_feature_size)
			if idx_to_tl == 0 or idx_to_tl - 1 == 0: # input is already formated at the preprocessing stage
				prev_output = target_X
			else:
				prev_output = t_model.predict(target_X)
			
			# the input's shape should be (batch_size, time_step, num_features)
			assert len(prev_output.shape) == 3, prev_output.shape
			num_features = prev_output.shape[-1] # the dimension of features that will be processed by the model

			num_units = t_w_kernel.shape[1] # since the length is 2 -> -1 = 1 
			assert t_w_kernel.shape[0] == num_features, "{} (kernel) vs {} (input)".format(t_w_kernel.shape[0], num_features)

			# hidden state and cell state sequences computation
			# generate a temporary model that only contains the target lstm layer 
			# but with the modification to return sequences of hidden and cell states
			temp_lstm_layer_inst = lstm_layer.LSTM_Layer(model.layers[idx_to_tl])
			temp_lstm_mdl = temp_lstm_layer_inst.gen_lstm_layer_from_another()
			# shapes of hstates_sequence and cell_states_sequence: 
			# 	(batch_size, num_units, time_steps), (batch_size, num_units, time_steps) 
			hstates_sequence, cell_states_sequence = temp_lstm_mdl(target_X) 
			init_hstates, init_cell_states = lstm_layer.LSTM_Layer.get_initial_state(model.layers[idx_to_tl])
			if init_hstates is None: 
				init_hstates = np.zeros((len(target_X), num_units)) # based on the how states was handled in the github source
			if init_cell_states is None:
				init_cell_states = np.zeros((len(target_X), num_units)) # shape = (batch_size, num_units)
			
			hstates_sequence = np.insert(hstates_sequence, 0, init_hstates, axis = 1) # shape = (batch_size, time_steps + 1, num_units)
			cell_states_sequence = np.insert(cell_states_sequence, 0, init_cell_states, axis = 1) # shape = (batch_size, time_steps + 1, num_units)

			########################################################### -> checked up to here
			#_gradient = K.get_session().run(temp_lstm, feed_dict={model.input: X[chunk]})[0]
			###############################################################

			bias = model.layer[idx_to_tl].get_weights()[-1] # shape = (4 * num_units,)
			indices_to_each_gates = np.array_split(np.arange(num_units * 4), 4)

			## prepare all the intermediate outputs and the variables that will be used later
			idx_to_input_gate = 0; idx_to_forget_gate = 1; idx_to_cand_gate = 2; idx_to_output_gate = 3
			
			# for kenerl, weight shape = (input_feature_size, num_units) and for recurrent, (num_units, num_units), bias (num_units)
			# and the shape of all the intermedidate outpu is "(batch_size, time_step, num_units)"
			
			# input 
			t_w_kernel_I = t_w_kernel[:, indices_to_each_gates[idx_to_input_gate]] 
			t_w_recurr_kernel_I = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_input_gate]] 
			bias_I = bias[indices_to_each_gates[idx_to_input_gate]]
			I = sigmoid(np.dot(prev_output, t_w_kernel_I) + np.dot(hstates_sequence, t_w_recurr_kernel_I) + bias_I) # then sigmoid

			# forget
			t_w_kernel_F = t_w_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
			t_w_recurr_kernel_F = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
			bias_F = bias[indices_to_each_gates[idx_to_forget_gate]]
			F = sigmoid(np.dot(prev_output, t_w_kernel_F) + np.dot(hstates_sequence, t_w_recurr_kernel_F) + bias_F) # then sigmoid

			# cand
			t_w_kernel_C = t_w_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
			t_w_recurr_kernel_C = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
			bias_C = bias[indices_to_each_gates[idx_to_cand_gate]]
			C = np.tanh(np.dot(prev_output, t_w_kernel_C) + np.dot(hstates_sequence, t_w_recurr_kernel_C) + bias_C) # then tanh

			# output
			t_w_kernel_O = t_w_kernel[:, indices_to_each_gates[idx_to_output_gate]]
			t_w_recurr_kernel_O = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_output_gate]]
			bias_O = bias[indices_to_each_gates[idx_to_output_gate]]
			# shape = (batch_size, time_steps, num_units)
			O = sigmoid(np.dot(prev_output, t_w_kernel_O) + np.dot(hstates_sequence, t_w_recurr_kernel_O) + bias_O) # then sigmoid

			# set arguments to compute forward impact for the neural weights from these four gates
			t_w_kernels = {'input':t_w_kernel_I, 'forget':t_w_kernel_F, 'cand':t_w_kernel_C, 'output':t_w_kernel_O}
			t_w_recurr_kernels = {'input':t_w_recurr_kernel_I, 'forget':t_w_recurr_kernel_F, 'cand':t_w_recurr_kernel_C, 'output':t_w_recurr_kernel_O}

			consts = {}
			consts['input'] = get_constants('input', F, I, C, O, cell_states_sequence)
			consts['forget'] = get_constants('forget', F, I, C, O, cell_states_sequence)
			consts['cand'] = get_constants('cand', F, I, C, O, cell_states_sequence)
			consts['output'] = get_constants('output', F, I, C, O, cell_states_sequence)

			# from_front's shape = (num_units, (num_features + num_units) * 4)
			# gate_orders = ['input', 'forget', 'cand', 'output']
			from_front, gate_orders  = lstm_local_front_FI_for_target_all(
				prev_output, hstates_sequence, num_units, 
				t_w_kernels, t_w_recurr_kernels, consts)

			from_front = from_front.T # ((num_features + num_units) * 4, num_units)
			# N_k_rk_w = num_features + num_units
			N_k_rk_w = int(from_front.shape[1]/4)
			assert N_k_rk_w == num_features + num_units, "{} vs {}".format(N_k_rk_w, num_features + num_units)
			
			## from behind
			from_behind = compute_gradient_to_output(
				path_to_keras_model, idx_to_tl, target_X, by_batch = True) # shape = (num_units,)
			print ("From behind", from_behind.shape)

			t1 = time.time()
			FIs_combined = np.multiply(from_front, from_behind) # shape = (N_k_rk_w, num_units)
			t2 = time.time()
			print ('Time for multiplying front and behind results: {}'.format(t2 - t1))
			
			# reshaping
			FIs_kernel = np.zeros(t_w_kernel.shape) # t_w_kernel's shape (num_features, num_units * 4)
			FIs_recurr_kernel = np.zeros(t_w_recurr_kernel.shape) # t_w_recurr_kernel's shape (num_units, num_units * 4)
			for i, FI_p_gate in enumerate(np.array_split(FIs_combined, 4, axis = 0)): # from (4 * N_k_rk_w, num_units) to (N_k_rk_w, num_units)
				# local indices that will split FI_p_gate (shape = (N_k_rk_w, num_units))
				indices_to_features = np.arange(num_features)
				indices_to_units = np.arange(num_units) + num_features

				FIs_kernel[indices_to_features + (i * N_k_rk_w)] = FI_p_gate[indices_to_features] # shape = (num_features, num_units)
				FIs_recurr_kernel[indices_to_units + (i * N_k_rk_w)] = FI_p_gate[indices_to_units] # shape = (num_units, num_units)
			t3 =time.time()
			FIs = [FIs_kernel, FIs_recurr_kernel]
			print ('Time for formatting: {}'.format(t3 - t2))
			
			## Gradient
			# will be 
			# this should be fixed to process a list of weights (or we can call it twice), and accept other loss function
			tensor_w_kernel, tensor_w_recurr_kernel, _ = model.layers[idx_to_tl].weights[:2]
			loss_func = loss_funcs[idx_to_tl] if loss_funcs[idx_to_tl] is not None else 'mse'
			grad_scndcr_kernel = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y, 
				target = tensor_w_kernel, by_batch = True, loss_func = loss_func)
			grad_scndcr_recurr_kernel = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y, 
				target = tensor_w_recurr_kernel, by_batch = True, loss_func = loss_func)
			
			grad_scndcr = [grad_scndcr_kernel, grad_scndcr_recurr_kernel]
		#elif is_Attention(lname):
		#	pass
		else:
			print ("Currenlty not supported: {}. (shoulde be filtered before)".format(lname))		
			import sys; sys.exit()

		t2 = time.time()
		print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
		####
		from collections.abc import Iterable
		if not isinstance(FIs, Iterable):
			pairs = np.asarray([grad_scndcr.flatten(), FIs.flatten()]).T
			#print ("Pairs", pairs.shape)
			total_cands[idx_to_tl] = {'shape':FIs.shape, 'costs':pairs}
			#sess = K.get_session()
			#sess.close()
		else:
			total_cands[idx_to_tl] = {'shape':[], 'costs':[]}
			pairs = []
			for _FIs, _grad_scndcr in zip(FIs, grad_scndcr):
				# after flatten -> (num_features * num_units or num_units * num_units, )
				pairs = np.asarray([_grad_scndcr.flatten(), _FIs.flatten()]).T
				total_cands[idx_to_tl]['shape'].append(_FIs.shape)
				total_cands[idx_to_tl]['costs'].append(pairs)
	
	t3 = time.time()
	print ("Time for computing total costs: {}".format(t3 - t0))

	return total_cands


def compute_avg_of_output_per_w(x, h, t_w_kernel, t_w_recurr_kernel, const, with_norm = False):
	"""
	A slice for a single neuron (unit or lstm cell)
	x = (batch_size, time_steps, num_features)
	h = (batch_size, time_steps, num_units)
	t_w_kernel = (num_features,)
	t_w_recurr_kernel = (num_units,)
	consts = (batch_size, time_steps) -> the value that is multiplied in the final state computation
	Return the product of the multiplication of weights and input for each unit (i.e., each LSTM cell)
	-> meaning the direct front impact computed per neural weights 
	"""
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")

	out_kernel = np.multiply(x, t_w_kernel) # shape = (batch_size, time_steps, num_features)
	if h is not None: # None -> we will skip the weights for hiddens states
		out_recurr_kernel = np.multiply(h, t_w_recurr_kernel) # shape = (batch_size, time_steps, num_units)
		out = np.append(out_kernel, out_recurr_kernel, axis = -1) # out's shape = (batch_size, time_steps, (num_features + num_units))
	else:
		out = out_kernel # shape = (batch_size, time_steps, num_features)

	# normalise -> make it between 0~1 (this is also to substitute the activation part ... 
	# => all the values would be scale between 0 to its constant, and then they will be normalised between 0~1 with 
	# all other neural weights from different gates before being returned as the front part of the FI
	if with_norm:
		out = np.abs(out)
		out = norm_scaler.fit_transform(out)

	# N = num_features or num_features + num_units
	#for i in range(len(out)):
	#	np.multiply(const, out[:,:,i])
	out = np.einsum('ijk,ij->ijk', out, const) # shape = (batch_size, time_steps, N)
	return out


def get_constants(gate, F, I, C, O, cell_states):
	"""
	... division -> there can be divided by zero here....
	"""
	if gate == 'input':
		#return np.multiply(O, np.divide(C, cell_states[:,1:,:]))
		return np.multipy(O, np.divide(
			C, cell_states[:,1:,:], 
			out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
	elif gate == 'forget':
		#return np.multiply(O, np.divide(cell_states[:,:-1,:], cell_states[:,1:,:]))
		return np.multipy(O, np.divide(
			cell_states[:,:-1,:], cell_states[:,1:,:], 
			out = np.zeros_like(cell_states[:,:-1,:]), where = cell_states[:,1:,:] != 0))
	elif gate == 'cand':
		#return np.multiply(O, np.divide(I, cell_states[:,1:,:]))
		return np.multipy(O, np.divide(
			I, cell_states[:,1:,:], 
			out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
	else: # output
		return np.tanh(cell_states[:,1:,1])
	

def lstm_local_front_FI_for_target_all(
	x, h, num_units, 
	t_w_kernels, t_w_recurr_kernels, consts, 
	gate_orders = ['input', 'forget', 'cand', 'output']):
	"""
	x = previous output
	h = hidden state (-> should be computed using the layer)
	t_w_kernels / t_w_recurr_kernels / consts: 
		a group of neural weights that should be taken into account when measuring the impact.
		arg consts is the corresponding group of constants that will be multiplied to each nueral weight's output, respectively
	"""
	from sklearn.preprocessing import Normalizer
	norm_scaler = Normalizer(norm = "l1")
	from_front = []
	for idx_to_unit in tqdm(num_units):
		out_combined = None
		for gate in gate_orders:
			out = compute_avg_of_output_per_w(
				x, 
				h if h is not None else None, 
				t_w_kernels[gate][:,idx_to_unit],
				t_w_recurr_kernels[gate][:,idx_to_unit],
				consts[gate][...,idx_to_unit],
				with_norm = True)

			if out_combined is None:
				out_combined = out 
			else:
				out_combined = np.append(out_combined, out, axis = -1)

		# the shape of out_combined => (batch_size, time_steps, 4 * (num_features + num_units)) (since this is per unit)
		# here, keep in mind that we have to use a scaler on the current out_combined (for instance, divide by the final output (the last 
		# hidden state won't work here anymore, since the summation of the current value differs from the original due to 
		# the absence of act and the scaling in the middle, etc.))
		scaled_out_combined = norm_scaler.fit_transform(out_combined) # normalised 
		# mean out_combined's shape: ((num_features + num_units) * 4,) -> for each neural weights, take the average across both the time step and the batch
		avg_scaled_out_combined = np.mean(scaled_out_combined.reshape(-1, scaled_out_combined.shape[-1]), axis = 0) # the average over both time step and the batch 
		from_front.append(avg_scaled_out_combined)

	# from_front's shape = (num_units, (num_features + num_units) * 4)
	from_front = np.asarray(from_front)
	print ("For lstm's front part of FI: {}".format(from_front.shape))
	return from_front, gate_orders


def localise_offline_only_changed(
	X, y,
	indices_to_target,
	target_weights,
	path_to_keras_model = None):
	"""
	localise based only on the given target, which is a set of misclassified inputs 
	old version -> not be used
	"""
	loc_start_time = time.time()
	# compute FI and GL
	total_cands = compute_FI_and_GL(
		X, y,
		indices_to_target,
		target_weights,
		path_to_keras_model = path_to_keras_model)

	# compute pareto front
	indices_to_tl = list(total_cands.keys())
	costs_and_keys = [([idx_to_tl, local_i], c) for idx_to_tl in indices_to_tl for local_i,c in enumerate(total_cands[idx_to_tl]['costs'])]
	costs = np.asarray([vs[1] for vs in costs_and_keys])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))

	# a list of [index to the target layer, index to a neural weight]
	indices_to_nodes = [[vs[0][0], np.unravel_index(vs[0][1], total_cands[vs[0][0]]['shape'])] for vs in costs_and_keys]
	
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
	loc_end_time = time.time()
	print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))
	return pareto_front, costs_and_keys


def localise_by_chgd_unchgd(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None,
	loss_funcs = None):
	"""
	Find those likely to be highly influential to the changed behaviour while less influential to the unchanged behaviour
	"""
	from collections.abc import Iterable
	#from scipy.stats import ks_2samp
	loc_start_time = time.time()

	print ("indices to chgd, unchgd", indices_to_chgd, indices_to_unchgd)
	# compute FI and GL with changed inputs
	total_cands_chgd = compute_FI_and_GL(
		X, y,
		indices_to_chgd,
		target_weights,
		loss_funcs = loss_funcs, 
		path_to_keras_model = path_to_keras_model)

	# compute FI and GL with unchanged inputs
	total_cands_unchgd = compute_FI_and_GL(
		X, y,
		indices_to_unchgd,
		target_weights,
		loss_funcs = loss_funcs, 
		path_to_keras_model = path_to_keras_model)

	indices_to_tl = list(total_cands_chgd.keys()) 
	costs_and_keys = []
	shapes = {}
	for idx_to_tl in indices_to_tl:
		if not isinstance(total_cands_chgd[idx_to_tl]['shape'], Iterable):
			assert not isinstance(total_cands_unchgd[idx_to_tl]['shape'], Iterable), type(total_cands_unchgd[idx_to_tl]['shape'])
			cost_from_chgd = total_cands_chgd[idx_to_tl]['costs']
			cost_from_unchgd = total_cands_unchgd[idx_to_tl]['costs']
			## key: more influential to changed behaviour and less influential to unchanged behaviour
			costs_combined = cost_from_chgd/(1. + cost_from_unchgd) # shape = (N,2)
			shapes[idx_to_tl] = total_cands_chgd[idx_to_tl]['shape']

			for i,c in enumerate(costs_combined):
				costs_and_keys.append(([idx_to_tl, i], c))
		else:
			assert isinstance(total_cands_unchgd[idx_to_tl]['shape'], Iterable), type(total_cands_unchgd[idx_to_tl]['shape'])
			num = len(total_cands_unchgd[idx_to_tl]['shape'])
			shapes[idx_to_tl] = []
			for idx_to_pair in range(num):
				cost_from_chgd = total_cands_chgd[idx_to_tl]['costs'][idx_to_pair]
				cost_from_unchgd = total_cands_unchgd[idx_to_tl]['costs'][idx_to_pair]
				## key: more influential to changed behaviour and less influential to unchanged behaviour
				costs_combined = cost_from_chgd/(1. + cost_from_unchgd) # shape = (N,2)
				shapes[idx_to_tl].append(total_cands_chgd[idx_to_tl]['shape'][idx_to_pair])

				for i,c in enumerate(costs_combined):
					costs_and_keys.append(([(idx_to_tl, idx_to_pair), i], c))

	costs = np.asarray([vs[1] for vs in costs_and_keys])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))
	# a list of [index to the target layer, index to a neural weight]
	#indices_to_nodes = [[vs[0][0], np.unravel_index(vs[0][1], shapes[vs[0][0]])] for vs in costs_and_keys]
	indices_to_nodes = []
	for vs in costs_and_keys:
		if not isinstance(vs[0][0], Iterable):
			_idx_to_tl = vs[0][0]
			indices_to_nodes.append([_idx_to_tl, np.unravel_index(vs[0][1], shapes[_idx_to_tl])])
		else:
			_idx_to_tl, idx_to_w = vs[0][0]
			indices_to_nodes.append([(_idx_to_tl, idx_to_w), np.unravel_index(vs[0][1], shapes[_idx_to_tl][idx_to_w])])


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

	pareto_front = [tuple(v) for v in np.asarray(indices_to_nodes, dtype = object)[is_efficient]]
	##
	t5 = time.time()
	print ("Time for computing the pareto front: {}".format(t5 - t4))
	loc_end_time = time.time()
	print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))
	## ** now, the pareto front may contains (idx_to_tl, idx_to_shape (0 (kernel) or 1 (recurr_kernel)))
	return pareto_front, costs_and_keys



def localise_by_sbfl(
	X, y,
	indices_to_selected_wrong,
	indices_to_correct, 
	target_weights,
	path_to_keras_model = None, 
	pass_input = True):
	"""
	Not exactly the same with DeepFL -> adjust it to apply on neural weights instead of neurons 
	"""
	total_cands = {}

	print ('Total {} layers are targeted'.format(len(target_weights)))
	t0 = time.time()
	## slice inputs
	target_X = X[indices_to_selected_wrong + indices_to_correct]
	target_y = y[indices_to_selected_wrong + indices_to_correct]
	# local indices
	new_indices_to_selected_wrong = np.arange(len(target_X))[:len(indices_to_selected_wrong)]
	new_indices_to_correct = np.arange(len(target_X))[len(indices_to_selected_wrong):]

	ochiai = lambda a_s, n_s, a_f, n_f: a_f/np.sqrt((a_f + n_f)*(a_f + a_s)) if (a_f + a_s > 0) else 0.
	loc_start_time = time.time()
	##
	for idx_to_tl, vs in target_weights.items():
		t1 = time.time()
		t_w, lname = vs
		print ("targeting layer {} ({})".format(idx_to_tl, lname))
		#t_w_tensor = model.layers[idx_to_tl].weights[0]

		model = load_model(path_to_keras_model, compile = False)
		if idx_to_tl - 1 == 0 and pass_input: # meaning the model doesn't specify the input layer explicitly
			#prev_output = target_X
			continue # we will pass the first 
		else:
			#prev_output = model.layers[idx_to_tl - 1].output
			t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
			prev_output = t_model.predict(target_X)

		##
		indices_to_active = list(zip(*np.where(prev_output != 0)))
		indices_to_act_input = indices_to_active[:,0] # 
		indices_to_non_active = list(zip(*np.where(prev_output == 0)))
		indices_to_non_act_input = indices_to_non_active[:,0] # 

		### indices to active/non-active and pass/fail inputs -> input-dimension
		indices_to_a_s_input = list(set(new_indices_to_correct).intersection(set(indices_to_act_input))) # active and 
		local_indices_to_a_s_input = [np.where(indices_to_act_input == idx)[0][0] for idx in indices_to_a_s_input]
		indices_to_a_f_input = list(set(new_indices_to_selected_wrong).intersection(set(indices_to_act_input)))
		local_indices_to_a_f_input = [np.where(indices_to_act_input == idx)[0][0] for idx in indices_to_a_f_input]
		
		indices_to_n_s_input = list(set(new_indices_to_correct).intersection(set(indices_to_non_act_input)))
		local_indices_to_n_s_input = [np.where(indices_to_non_act_input == idx)[0][0] for idx in indices_to_n_s_input]
		indices_to_n_f_input = list(set(new_indices_to_selected_wrong).intersection(set(indices_to_non_act_input)))
		local_indices_to_n_f_input = [np.where(indices_to_non_act_input == idx)[0][0] for idx in indices_to_n_f_input]
		
		############### for computing hit spectrum #########################
		## below is for identifying suspicious neurons 
		def set_hit_spectrum(hit_T_indices, prev_output_shape):
			"""
			"""
			hit_cnt_arr = np.zeros(prev_output_shape[1:])
			for idx in hit_T_indices[:,1:]:
				hit_cnt_arr[idx] += 1

			return hit_cnt_arr

		# to construct a hit-spectrum 
		indices_to_as = indices_to_active[local_indices_to_a_s_input] # active in passing inputs 
		indices_to_af = indices_to_active[local_indices_to_a_f_input] # active in failing inputs
		indices_to_ns = indices_to_non_active[local_indices_to_n_s_input] # non-active in passing inputs
		indices_to_nf = indices_to_non_active[local_indices_to_n_f_input] # non-active in failing inputs

		hit_as = set_hit_spectrum(indices_to_as, prev_output.shape)
		hit_af = set_hit_spectrum(indices_to_af, prev_output.shape)
		hit_ns = set_hit_spectrum(indices_to_ns, prev_output.shape)
		hit_nf = set_hit_spectrum(indices_to_nf, prev_output.shape)

		# prev_output without input dimension -> for each, a vector of length 4
		hit_spectrum_of_prev_output = np.moveaxis(np.asarray([hit_as, hit_af, hit_ns, hit_nf]), [0], [-1])
		##########################################################################
		##
		#for i in range(indices_to_as.shape[0]):
		##
		#layer_config = model.layers[idx_to_tl].get_config() 

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
			#
			if len(prev_output.shape) == 3:
				prev_output = prev_output.reshape(prev_output.shape[0], prev_output.shape[-1])
			#
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

			from_behind = compute_gradient_to_output(path_to_keras_model, idx_to_tl, target_X, by_batch = True) # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
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
			grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y, by_batch = True)
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
		if is_C2D(lname):
			by_batch = True
		else:
			by_batch = False
		grad_scndcr = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, target_X, target_y, by_batch = True)
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


def localise_by_gradient_v2(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None):
	"""
	localise using chgd & unchgd
	"""
	total_cands = {}

	print ('Total {} layers are targeted'.format(len(target_weights)))
	t0 = time.time()
	## slice inputs

	loc_start_time = time.time()
	##
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print ("targeting layer {} ({})".format(idx_to_tl, lname))
		
		t1 = time.time()
		#if is_C2D(lname):
			#by_batch = True
		#else:
			#by_batch = False
		grad_scndcr_for_chgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_chgd], y[indices_to_chgd], by_batch = True)
		grad_scndcr_for_unchgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_unchgd], y[indices_to_unchgd], by_batch = True)
		t2 = time.time()

		print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
		assert t_w.shape == grad_scndcr_for_chgd.shape, "{} vs {}".format(t_w.shape, grad_scndcr_for_chgd.shape)

		total_cands[idx_to_tl] = {'shape':grad_scndcr_for_chgd.shape, 'costs':grad_scndcr_for_chgd.flatten()/(1.+grad_scndcr_for_unchgd.flatten())}
	
	t3 = time.time()
	print ("Time for computing total costs: {}".format(t3 - t0))

	indices_to_tl = list(total_cands.keys())
	costs_and_keys = [([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
		for idx_to_tl in indices_to_tl 
		for local_i,c in enumerate(total_cands[idx_to_tl]['costs'])]
	
	costs = np.asarray([vs[1] for vs in costs_and_keys])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	loc_end_time = time.time()
	print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))

	return sorted_costs_and_keys


def localise_by_gradient_v3(
	X, y,
	indices_to_chgd,
	indices_to_unchgd,
	target_weights,
	path_to_keras_model = None, 
	loss_funcs = None):
	"""
	localise using chgd & unchgd
	"""
	from collections.abc import Iterable
	
	total_cands = {}
	print ('Total {} layers are targeted'.format(len(target_weights)))
	t0 = time.time()
	## slice inputs
	loc_start_time = time.time()
	##
	for idx_to_tl, vs in target_weights.items():
		t_w, lname = vs
		print ("targeting layer {} ({})".format(idx_to_tl, lname))
		t1 = time.time()
		if is_C2D(lname) or is_FC(lname):
			loss_func = loss_funcs[idx_to_tl] if loss_funcs[idx_to_tl] is not None else 'softmax'
			grad_scndcr_for_chgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_chgd], y[indices_to_chgd], 
				loss_func = loss_func, by_batch = True)
			grad_scndcr_for_unchgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_unchgd], y[indices_to_unchgd], 
				loss_func = loss_func, by_batch = True)

			assert t_w.shape == grad_scndcr_for_chgd.shape, "{} vs {}".format(t_w.shape, grad_scndcr_for_chgd.shape)
			total_cands[idx_to_tl] = {'shape':grad_scndcr_for_chgd.shape, 
									'costs':grad_scndcr_for_chgd.flatten()/(1.+grad_scndcr_for_unchgd.flatten())}
		elif is_LSTM(lname):
			model = load_model(path_to_keras_model, compile = False)
			tensor_w_kernel, tensor_w_recurr_kernel, _ = model.layers[idx_to_tl].weights[:2]
			loss_func = loss_funcs[idx_to_tl] if loss_funcs[idx_to_tl] is not None else 'mse'
			# for changed inputs
			# kernel
			grad_scndcr_kernel_chgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_chgd], y[indices_to_chgd],
				target = tensor_w_kernel, loss_func = loss_func, by_batch = True)
			# recurrent kernel
			grad_scndcr_recurr_kernel_chgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_chgd], y[indices_to_chgd], 
				target = tensor_w_recurr_kernel, loss_func = loss_func, by_batch = True)
			grad_scndcr_for_chgd = [grad_scndcr_kernel_chgd, grad_scndcr_recurr_kernel_chgd]

			# for unchanged inptus
			# kernel
			grad_scndcr_kernel_unchgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_unchgd], y[indices_to_unchgd],
				target = tensor_w_kernel, loss_func = loss_func, by_batch = True)
			# recurrent kernel
			grad_scndcr_recurr_kernel_unchgd = compute_gradient_to_loss(path_to_keras_model, idx_to_tl, X[indices_to_unchgd], y[indices_to_unchgd], 
				target = tensor_w_recurr_kernel, loss_func = loss_func, by_batch = True)
			grad_scndcr_for_unchgd = [grad_scndcr_kernel_unchgd, grad_scndcr_recurr_kernel_unchgd]
			
			# check
			assert t_w[0].shape == grad_scndcr_for_chgd[0].shape, "{} vs {}".format(t_w[0].shape, grad_scndcr_for_chgd[0].shape)
			assert t_w[1].shape == grad_scndcr_for_chgd[1].shape, "{} vs {}".format(t_w[1].shape, grad_scndcr_for_chgd[1].shape)

			# generate total candidates
			total_cands[idx_to_tl] = {'shape':[], 'costs':[]}
			for _grad_scndr_chgd, _grad_scndr_unchgd in zip(grad_scndcr_for_chgd, grad_scndcr_for_unchgd):
				#_grad_scndr_chgd & _grad_scndr_unchgd -> can be for either kernel or recurrent kernel
				_costs = _grad_scndr_chgd.flatten()/(1. + _grad_scndr_unchgd.flatten())
				total_cands[idx_to_tl]['shape'].append(_grad_scndr_chgd.shape)
				total_cands[idx_to_tl]['costs'].append(_costs)
		else:
			print ("{} not supported yet".format(lname))
			assert False
		###
		
		t2 = time.time()
		print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
	
	t3 = time.time()
	print ("Time for computing total costs: {}".format(t3 - t0))

	indices_to_tl = list(total_cands.keys())
	costs_and_keys = []
	for idx_to_tl in indices_to_tl:
		if not isinstance(total_cands[idx_to_tl]['shape'], Iterable):
			for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
				cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c) 
				costs_and_keys.append(cost_and_key)
		else:
			num = len(total_cands[idx_to_tl]['shape'])
			for idx_to_w in range(num):
				for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
					cost_and_key = ([(idx_to_tl, idx_to_w), np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c) 
					costs_and_keys.append(cost_and_key)

	costs = np.asarray([vs[1] for vs in costs_and_keys])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))

	sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
	loc_end_time = time.time()
	print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))

	return sorted_costs_and_keys


def localise_by_random_selection(number_of_place_to_fix, target_weights):
	"""
	randomly select places to fix
	"""
	from collections.abc import Iterable

	total_indices = []
	for idx_to_tl, vs in target_weights.items():
		t_w, _ = vs
		if not isinstance(t_w, Iterable):
			l_indices = list(np.ndindex(t_w.shape))
			total_indices.extend(list(zip([idx_to_tl] * len(l_indices), l_indices)))
		else: # to handle the layers with more than one weights (e.g., LSTM)
			for idx_to_w, a_t_w in enumerate(t_w):
				l_indices = list(np.ndindex(a_t_w.shape))
				total_indices.extend(list(zip([idx_to_tl, idx_to_w] * len(l_indices), l_indices)))

	if number_of_place_to_fix > 0 and number_of_place_to_fix < len(total_indices):
		selected_indices = np.random.choice(np.arange(len(total_indices)), number_of_place_to_fix, replace = False)
		indices_to_places_to_fix = [total_indices[idx] for idx in selected_indices]
	else:
		indices_to_places_to_fix = total_indices
	
	return indices_to_places_to_fix
