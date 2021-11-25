"""
Localise faults in offline for any faults
"""
import numpy as np
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
			
def compute_gradient_to_loss(path_to_keras_model, idx_to_target_layer, X, y, by_batch = False, wo_reset = False):
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

	###
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

	return total_cands


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
	path_to_keras_model = None):
	"""
	Find those likely to be highly influential to the changed behaviour while less influential to the unchanged behaviour
	"""
	from scipy.stats import ks_2samp
	loc_start_time = time.time()

	print ("indices to chgd, unchgd", indices_to_chgd, indices_to_unchgd)
	# compute FI and GL with changed inputs
	total_cands_chgd = compute_FI_and_GL(
		X, y,
		indices_to_chgd,
		target_weights,
		path_to_keras_model = path_to_keras_model)

	# compute FI and GL with unchanged inputs
	total_cands_unchgd = compute_FI_and_GL(
		X, y,
		indices_to_unchgd,
		target_weights,
		path_to_keras_model = path_to_keras_model)


	indices_to_tl = list(total_cands_chgd.keys()) 
	costs_and_keys = []
	shapes = {}
	for idx_to_tl in indices_to_tl:
		cost_from_chgd = total_cands_chgd[idx_to_tl]['costs']
		cost_from_unchgd = total_cands_unchgd[idx_to_tl]['costs']
		## key: more influential to changed behaviour and less influential to unchanged behaviour
		costs_combined = cost_from_chgd/(1. + cost_from_unchgd) # shape = (N,2)
		shapes[idx_to_tl] = total_cands_chgd[idx_to_tl]['shape']

		for i,c in enumerate(costs_combined):
			costs_and_keys.append(([idx_to_tl, i], c))

	costs = np.asarray([vs[1] for vs in costs_and_keys])
	print ("Indices", indices_to_tl)
	print ("the number of total cands: {}".format(len(costs)))
	# a list of [index to the target layer, index to a neural weight]
	indices_to_nodes = [[vs[0][0], np.unravel_index(vs[0][1], shapes[vs[0][0]])] for vs in costs_and_keys]

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
		if is_C2D(lname):
			by_batch = True
		else:
			by_batch = False

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
