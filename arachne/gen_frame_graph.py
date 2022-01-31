"""
empty graph (tf.Graph) generation script
"""
import os
from re import I

from utils import model_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def build_partially_fronzen_model(mdl, X, indices_to_tls, indices_to_tneurons):
	"""
	indices_to_tls: a list of indices to target layers
	indices_to_tneurons (dict):
		key: indices_to_tl
		value: inner indices to target neurons of a given layer
	
	"""
	import tensorflow as tf
	from tensorflow.keras.models import Model
	import tensorflow.keras.backend as K
	import re
	import numpy as np
	import run_localise

	targeting_clname_pattns = ['Dense*', 'Conv*'] #if not target_all else None
	is_target = lambda clname, targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])

	# dictionary to describe the network graph
	network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

	# Set the input layers of each layer
	for layer in mdl.layers:
		for node in layer._outbound_nodes: # the output node of the current layer (layer)
			layer_name = node.outbound_layer.name # the layer that take node as input 
			# layer_name takes layer.name as input
			if layer_name not in network_dict['input_layers_of']:
				network_dict['input_layers_of'].update({layer_name: [layer.name]})    
			else:
				network_dict['input_layers_of'][layer_name].append(layer.name)

	min_idx_to_tl = np.min(indices_to_tls)
	t_mdl = Model(inputs = mdl.input, outputs = mdl.layers[min_idx_to_tl-1].output)
	prev_output = t_mdl.predict(X)
	num_layers = len(mdl.layers)
	
	# Iterate over all layers after the input
	model_outputs = []
	dtype = mdl.layers[min_idx_to_tl-1].output.dtype
	x = tf.constant(prev_output, dtype = dtype) # can be changed to a placeholder and take the prev_output freely
	
	# Set the output tensor of the input layer (or more exactly, our starting point)
	network_dict['new_output_tensor_of'].update({mdl.layers[min_idx_to_tl-1].name: x})
	
	t_ws = []
	#for layer in model.layers[1:]:
	for idx_to_l in range(min_idx_to_tl, num_layers):
		layer = mdl.layers[idx_to_l]
		layer_name = layer.name
		
		# Determine input tensors
		layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
					   for layer_aux in network_dict['input_layers_of'][layer.name]]
		#layer_input = network_dict[layer_name]

		if len(layer_input) == 1:
			layer_input = layer_input[0]
			
		# Insert layer if name matches the regular expression
		if idx_to_l in indices_to_tls:
			l_class_name = type(layer).__name__
			#new_layer = insert_layer_factory(layer_name) ## => currenlty, support only conv2d and dense layers
			###
			if is_target(l_class_name, targeting_clname_pattns): # is our target (dense and conv2d)
				if run_localise.is_FC(l_class_name):
					# w,b = layer.get_weights()
					# t_w = tf.placeholder(dtype = w.dtype, shape = w.shape)    
					# t_b = tf.constant(b)
					#
					# x = tf.add(tf.matmul(layer_input, t_w), t_b, name = layer_name)
					# t_ws.append(t_w)

					w, b = layer.get_weights()
					t_w_c = tf.constant(w)
					t_w_b = tf.constant(b)

					indices_to_tneurons[idx_to_l]
					#tf.Variable(initial_value=)
					pass
				else: 
					# should be conv2d, if not, then something is wrong
					assert run_localise.is_C2D(l_class_name), "{}".format(l_class_name)

					w,b = layer.get_weights()
					t_w = tf.placeholder(dtype = w.dtype, shape = w.shape)
					t_b = tf.constant(b)
					
					x = tf.nn.conv2d(layer_input, 
							t_w, 
							strides = list(layer.get_config()['strides'])*2, 
							padding = layer.get_config()['padding'].upper(), 
							data_format = 'NCHW',  
							#data_format = 'NHWC',
							name = layer_name)
					
					x = tf.nn.bias_add(x, t_b, data_format = 'NCHW')	
					t_ws.append(t_w)
			else:
				msg = "{}th layer {}({}) is not our target".format(idx_to_l, layer_name, l_class_name)
				assert False, msg
		else:
			layer.trainable = False # freeze non-target layers
			x = layer(layer_input)

		# Set new output tensor (the original one, or the one of the replaced layer)
		network_dict['new_output_tensor_of'].update({layer_name: x}) # x is the output of the layer "layer_name" (current one)
		# Save tensor in output list if it is output in initial model
		#if layer_name in model.output_names:
		#    model_outputs.append(x)

	#return Model(inputs=model.inputs, outputs=model_outputs)
	last_layer_name = mdl.layers[-1].name

	ys = tf.placeholder(dtype = tf.float32, shape = (None,10))
	pred_probs = tf.math.softmax(network_dict['new_output_tensor_of'][last_layer_name]) # softmax
	loss_op = tf.keras.metrics.categorical_crossentropy(ys, pred_probs)

	fn = K.function(t_ws + [ys], [network_dict['new_output_tensor_of'][last_layer_name], loss_op])
	
	return fn, t_ws, ys


def build_mdl_lst(org_mdl, prev_out_shape, indices_to_tls):
	"""
	New 
	"""
	import tensorflow as tf
	from tensorflow.keras.models import Model
	import numpy as np
	import tensorflow.keras.backend as K
	from collections.abc import Iterable

	mdl = tf.keras.models.clone_model(org_mdl)
	
	# dictionary to describe the network graph
	network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

	# Set the input layers of each layer
	for layer in mdl.layers:
		for node in layer._outbound_nodes: # the output node of the current layer (layer)
			layer_name = node.outbound_layer.name # the layer that take node as input 
			# layer_name takes layer.name as input
			if layer_name not in network_dict['input_layers_of']:
				network_dict['input_layers_of'].update({layer_name: [layer.name]})    
			else:
				network_dict['input_layers_of'][layer_name].append(layer.name)

	min_idx_to_tl = np.min([idx if not isinstance(idx, Iterable) else idx[0] for idx in indices_to_tls])
	num_layers = len(mdl.layers)
	model_input = tf.keras.Input(shape = prev_out_shape)

	# Iterate over all layers after the input
	if min_idx_to_tl == 0:# or min_idx_to_tl - 1 == 0:
		layer_name = mdl.layers[0].name
		if model_util.is_Input(type(mdl.layers[0]).__name__): # if the previous layer (layer_name) is an input layer
			network_dict['new_output_tensor_of'].update({layer_name: model_input}) # set model_input as the output of this input layer
		else: # if it is not (happen when using Sequential())
			_input_layer_name = 'input_layer' # -> insert one addiitonal input layer
			network_dict['new_output_tensor_of'].update({_input_layer_name: model_input}) # x is the output of _input_layer_name
			network_dict['input_layers_of'].update({layer_name: [_input_layer_name]}) # the input's of layer_name is _input_layer_name
	else:
		network_dict['new_output_tensor_of'].update({mdl.layers[min_idx_to_tl-1].name: model_input})

	for idx_to_l in range(min_idx_to_tl, num_layers):
		layer = mdl.layers[idx_to_l]
		layer_name = layer.name
		# Determine input tensors
		layer_input = []
		for layer_aux in network_dict['input_layers_of'][layer.name]:
			layer_input.append(network_dict['new_output_tensor_of'][layer_aux])

		if len(layer_input) == 1:
			layer_input = layer_input[0]
		x = layer(layer_input)
		# Set new output tensor (the original one, or the one of the replaced layer)
		network_dict['new_output_tensor_of'].update({layer_name: x}) # x is the output of the layer "layer_name" (current one)

	last_layer_name = mdl.layers[-1].name
	mdl = Model(inputs = model_input, # this is a constant
		outputs = network_dict['new_output_tensor_of'][last_layer_name])

	#fn = K.function([pred_tensor, y_tensor], [loss_op])
	return mdl


def build_k_frame_model_v2(mdl, X, indices_to_tls, act_func = None):
	"""
	** The one that actually used in Arachne. All the others (maybe except build_partially_fronzen_model) will be removed
	"""
	import tensorflow as tf
	from tensorflow.keras.models import Model
	import tensorflow.keras.backend as K
	import re
	import numpy as np
	import run_localise
	from collections.abc import Iterable

	targeting_clname_pattns = ['Dense*', 'Conv*']#, 'LSTM*'] #if not target_all else None
	is_target = lambda clname, targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])

	# dictionary to describe the network graph
	network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

	# Set the input layers of each layer
	for layer in mdl.layers:
		for node in layer._outbound_nodes: # the output node of the current layer (layer)
			layer_name = node.outbound_layer.name # the layer that take node as input 
			# layer_name takes layer.name as input
			if layer_name not in network_dict['input_layers_of']:
				network_dict['input_layers_of'].update({layer_name: [layer.name]})    
			else:
				network_dict['input_layers_of'][layer_name].append(layer.name)

	#min_idx_to_tl = np.min(indices_to_tls) # !!!!!
	min_idx_to_tl = np.min([idx if not isinstance(idx, Iterable) else idx[0] for idx in indices_to_tls])
	#t_mdl = Model(inputs = mdl.input, outputs = mdl.layers[min_idx_to_tl-1].output)
	#prev_output = t_mdl.predict(X)
	num_layers = len(mdl.layers)
	
	# Iterate over all layers after the input
	model_outputs = []	
	# Set the output tensor of the input layer (or more exactly, our starting point)
	if min_idx_to_tl == 0 or min_idx_to_tl - 1 == 0:
		x = tf.constant(X, dtype = tf.float32) #X.dtype)
		layer_name = mdl.layers[0].name
		network_dict['new_output_tensor_of'].update({layer_name: x})
	else:
		t_mdl = Model(inputs = mdl.input, outputs = mdl.layers[min_idx_to_tl-1].output)
		#if conv_in_to_3d:
		#	prev_output = t_mdl.predict(X)
		#else:
		prev_output = t_mdl.predict(X)	
		dtype = mdl.layers[min_idx_to_tl-1].output.dtype
		x = tf.constant(prev_output, dtype = dtype) # can be changed to a placeholder and take the prev_output freely

		network_dict['new_output_tensor_of'].update({mdl.layers[min_idx_to_tl-1].name: x})

	t_ws = []
	for idx_to_l in range(min_idx_to_tl, num_layers):
		#print ("================={}=================".format(idx_to_l))
		layer = mdl.layers[idx_to_l]
		layer_name = layer.name
		#print ("\tLayer:", layer_name)
		
		# Determine input tensors
		layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
					   for layer_aux in network_dict['input_layers_of'][layer.name]]
		#layer_input = network_dict[layer_name]

		if len(layer_input) == 1:
			layer_input = layer_input[0]
			
		# Insert layer if name matches the regular expression
		if idx_to_l in indices_to_tls:
			#print ('In here', idx_to_l)
			l_class_name = type(layer).__name__
			#new_layer = insert_layer_factory(layer_name) ## => currenlty, support only conv2d and dense layers
			###
			if is_target(l_class_name, targeting_clname_pattns): # is our target (dense and conv2d), and now also the lstm
				if model_util.is_FC(l_class_name):
					#print ("In fc")
					w,b = layer.get_weights()
					t_w = tf.placeholder(dtype = w.dtype, shape = w.shape)    
					t_b = tf.constant(b)
					#print ("dot", layer_input.shape, t_w.shape, t_b.shape, tf.tensordot(layer_input, t_w, [[len(layer_input.shape)-1],[0]]).shape)
					# this is a neat way, but the probelm is memory explosion... -> process by batches
					x = tf.add(tf.tensordot(layer_input, t_w, [[len(layer_input.shape)-1],[0]]), t_b, name = layer_name) 
					if act_func is not None:
						x = act_func(x)
					#print ("x", x)
					t_ws.append(t_w)
				elif model_util.is_C2D(l_class_name):
					w,b = layer.get_weights()
					t_w = tf.placeholder(dtype = w.dtype, shape = w.shape)
					t_b = tf.constant(b, dtype = b.dtype)
					x = tf.nn.conv2d(layer_input, 
							t_w,
							strides = list(layer.get_config()['strides'])*2, 
							padding = layer.get_config()['padding'].upper(), 
							data_format = 'NCHW',  
							name = layer_name)
					x = tf.nn.bias_add(x, t_b, data_format = 'NCHW')
					if act_func is not None: # tf.nn.relu
						x = act_func(x)
					t_ws.append(t_w)
				elif model_util.is_LSTM(l_class_name): ### *** SHOULD FIX ###
					### -> set weight inside...???
					#from lstm_layer import LSTM_Layer
					from tensorflow.keras.layers import LSTM
					### -> split! => append kernel and recurrent kernel weights and ...slice it here
					w_kernel, w_recurr_kernel, b = layer.get_weights()
					w_k_rk_shape = (w_kernel.shape[0] + w_recurr_kernel.shape[0], w_kernel.shape[1])
					t_w_k_rk = tf.placeholder(dtype = w_kernel.dtype, shape = w_k_rk_shape)
					t_b = tf.constant(b, dtype = b.dtype)
					n_unit = int(w_kernel.shape[1]/4)

					new_lstm = LSTM(n_unit,
						kernel_initializer=tf.constant_initializer(t_w_k_rk),
						recurrent_initializer=tf.constant_initializer(recurr_kernel_w),
						bias_initializer=tf.constant_initializer(bias),
						return_state=True)#, 
						#input_shape = input_shape)
					print (new_lstm)	
					skip = ['units', 'kernel_initializer', 'recurrent_initializer', 'bias_initializer', 'input_shape', 'return_state']
					for k,v in self.init_lstm_layer.__dict__.items():
						if k not in skip:
							new_lstm.__dict__.update({k:v})

					# lstm_layer here ..?
					
					t_w = tf.placeholder(dtype = w_kernel.dtype, shape = w_kernel.shape)
					t_w_kernel = tf.placeholder(dtype = w_kernel.dtype, shape = w_kernel.shape)
					t_w_recurr_kernel = tf.placeholder(dtype = w_recurr_kernel.dtype, shape = w_recurr_kernel.shape)
					t_b = tf.constant(b, dtype = b.dtype)			
					def get_value_from_placeholder(p, feed_dict):
						with tf.Session() as sess:
							w = sess.run({p})
					# placeholder -> variable 
					# based on the version -> tf.compat.v1.rnn.LSTMCell
					if tf.__version__.split('.')[0] == 1:
						x = tf.nn.rnn.LSTMCell(layer_input)
					else:
						x = tf.compat.v1.nn.rnn.LSTMCell()
			else:
				msg = "{}th layer {}({}) is not our target".format(idx_to_l, layer_name, l_class_name)
				assert False, msg
		else:
			x = layer(layer_input)

		# Set new output tensor (the original one, or the one of the replaced layer)
		network_dict['new_output_tensor_of'].update({layer_name: x}) # x is the output of the layer "layer_name" (current one)

	last_layer_name = mdl.layers[-1].name
	num_label = int(mdl.layers[-1].output.shape[-1])

	ys = tf.placeholder(dtype = tf.float32, shape = (None, num_label))
	pred_probs = tf.math.softmax(network_dict['new_output_tensor_of'][last_layer_name]) # softmax
	if len(ys.shape) != len(pred_probs.shape):
		if pred_probs.shape[1] == 1:
			pred_probs = tf.squeeze(pred_probs, axis = 1)
	loss_op = tf.keras.metrics.categorical_crossentropy(ys, pred_probs)
	fn = K.function(t_ws + [ys], [network_dict['new_output_tensor_of'][last_layer_name], loss_op])
	return fn, t_ws, ys


def build_k_frame_model(mdl, X, indices_to_tls, act_func = None):
	"""
	** The one that actually used in Arachne. All the others (maybe except build_partially_fronzen_model) will be removed
	"""
	import tensorflow as tf
	from tensorflow.keras.models import Model
	import tensorflow.keras.backend as K
	import re
	import numpy as np
	import run_localise
	from collections.abc import Iterable

	targeting_clname_pattns = ['Dense*', 'Conv*']#, 'LSTM*'] #if not target_all else None
	is_target = lambda clname, targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])

	# dictionary to describe the network graph
	network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

	# Set the input layers of each layer
	for layer in mdl.layers:
		for node in layer._outbound_nodes: # the output node of the current layer (layer)
			layer_name = node.outbound_layer.name # the layer that take node as input 
			# layer_name takes layer.name as input
			if layer_name not in network_dict['input_layers_of']:
				network_dict['input_layers_of'].update({layer_name: [layer.name]})    
			else:
				network_dict['input_layers_of'][layer_name].append(layer.name)

	#min_idx_to_tl = np.min(indices_to_tls) # !!!!!
	min_idx_to_tl = np.min([idx if not isinstance(idx, Iterable) else idx[0] for idx in indices_to_tls])
	#t_mdl = Model(inputs = mdl.input, outputs = mdl.layers[min_idx_to_tl-1].output)
	#prev_output = t_mdl.predict(X)
	num_layers = len(mdl.layers)
	
	# Iterate over all layers after the input
	model_outputs = []
	
	#dtype = mdl.layers[min_idx_to_tl-1].output.dtype
	#x = tf.constant(prev_output, dtype = dtype) # can be changed to a placeholder and take the prev_output freely
	
	# Set the output tensor of the input layer (or more exactly, our starting point)
	if min_idx_to_tl == 0 or min_idx_to_tl - 1 == 0:
		x = tf.constant(X, dtype = tf.float32) #X.dtype)
		layer_name = mdl.layers[0].name
		network_dict['new_output_tensor_of'].update({layer_name: x})
	else:
		t_mdl = Model(inputs = mdl.input, outputs = mdl.layers[min_idx_to_tl-1].output)
		#if conv_in_to_3d:
		#	prev_output = t_mdl.predict(X)
		#else:
		prev_output = t_mdl.predict(X)	
		dtype = mdl.layers[min_idx_to_tl-1].output.dtype
		x = tf.constant(prev_output, dtype = dtype) # can be changed to a placeholder and take the prev_output freely

		network_dict['new_output_tensor_of'].update({mdl.layers[min_idx_to_tl-1].name: x})

	t_ws = []
	for idx_to_l in range(min_idx_to_tl, num_layers):
		#print ("================={}=================".format(idx_to_l))
		layer = mdl.layers[idx_to_l]
		layer_name = layer.name
		#print ("\tLayer:", layer_name)
		
		# Determine input tensors
		layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
					   for layer_aux in network_dict['input_layers_of'][layer.name]]
		#layer_input = network_dict[layer_name]

		if len(layer_input) == 1:
			layer_input = layer_input[0]
			
		# Insert layer if name matches the regular expression
		if idx_to_l in indices_to_tls:
			#print ('In here', idx_to_l)
			l_class_name = type(layer).__name__
			#new_layer = insert_layer_factory(layer_name) ## => currenlty, support only conv2d and dense layers
			###
			if is_target(l_class_name, targeting_clname_pattns): # is our target (dense and conv2d), and now also the lstm
				if model_util.is_FC(l_class_name):
					#print ("In fc")
					w,b = layer.get_weights()
					t_w = tf.placeholder(dtype = w.dtype, shape = w.shape)    
					t_b = tf.constant(b)
					#print ("dot", layer_input.shape, t_w.shape, t_b.shape, tf.tensordot(layer_input, t_w, [[len(layer_input.shape)-1],[0]]).shape)
					# this is a neat way, but the probelm is memory explosion... -> process by batches
					x = tf.add(tf.tensordot(layer_input, t_w, [[len(layer_input.shape)-1],[0]]), t_b, name = layer_name) 
					#print ("X", x.shape)
					#print (len(layer_input.shape))
					#x = tf.add(tf.matmul(layer_input, t_w), t_b, name = layer_name)
					if act_func is not None:
						x = act_func(x)
					#print ("x", x)
					t_ws.append(t_w)
				else:# model_util.is_C2D(l_class_name):
					#print ("In CONV2D")
					#print (layer)
					#print (layer.get_config())
					# should be conv2d, if not, then something is wrong
					w,b = layer.get_weights()
					t_w = tf.placeholder(dtype = w.dtype, shape = w.shape)
					t_b = tf.constant(b, dtype = b.dtype)
					
					if layer.get_config()['data_format'] == 'channels_first':
						data_format  = 'NCHW'
					else: # channels_last
						data_format = 'NHWC'
					x = tf.nn.conv2d(layer_input, 
							t_w,
							strides = list(layer.get_config()['strides'])*2, 
							padding = layer.get_config()['padding'].upper(), 
							data_format = data_format,  
							name = layer_name)
					x = tf.nn.bias_add(x, t_b, data_format = data_format)
					if act_func is not None: # tf.nn.relu
						x = act_func(x)
					t_ws.append(t_w)
				#elif model_util.is_LSTM(l_class_name): ### *** SHOULD FIX ###
					#w_kernel, w_recurr_kernel, bais = layer.get_weights()
					## lstm_layer here ..?
					#from lstm_layer import LSTM_Layer
					#t_w_kernel = tf.placeholder(dtype = w_kernel.dtype, shape = w_kernel.shape)
					#t_w_recurr_kernel = tf.placeholder(dtype = w_recurr_kernel.dtype, shape = w_recurr_kernel.shape)
					#t_b = tf.constant(b, dtype = b.dtype)			
					#def get_value_from_placeholder(p, feed_dict):
						#with tf.Session() as sess:
							#w = sess.run({p})
					## placeholder -> variable 
					## based on the version -> tf.compat.v1.rnn.LSTMCell
					#if tf.__version__.split('.')[0] == 1:
						#x = tf.nn.rnn.LSTMCell(layer_input)
					#else:
						#x = tf.compat.v1.nn.rnn.LSTMCell()
			else:
				msg = "{}th layer {}({}) is not our target".format(idx_to_l, layer_name, l_class_name)
				assert False, msg
		else:
			#print ('LAYER', layer, layer.name, layer.input)
			#print ('layer input', layer_input)
			x = layer(layer_input)

		# Set new output tensor (the original one, or the one of the replaced layer)
		network_dict['new_output_tensor_of'].update({layer_name: x}) # x is the output of the layer "layer_name" (current one)
		#print ("\t", "--", network_dict['new_output_tensor_of'])
		# Save tensor in output list if it is output in initial model
		#if layer_name in model.output_names:
		#    model_outputs.append(x)

	#return Model(inputs=model.inputs, outputs=model_outputs)
	last_layer_name = mdl.layers[-1].name
	num_label = int(mdl.layers[-1].output.shape[-1])

	ys = tf.placeholder(dtype = tf.float32, shape = (None, num_label))
	pred_probs = tf.math.softmax(network_dict['new_output_tensor_of'][last_layer_name]) # softmax
	if len(ys.shape) != len(pred_probs.shape):
		if pred_probs.shape[1] == 1:
			pred_probs = tf.squeeze(pred_probs, axis = 1)
	loss_op = tf.keras.metrics.categorical_crossentropy(ys, pred_probs)
	fn = K.function(t_ws + [ys], [network_dict['new_output_tensor_of'][last_layer_name], loss_op])
	#fn = K.function(t_ws + [ys], [network_dict['new_output_tensor_of'][last_layer_name]])	
	#fn = K.function(t_ws + [ys], [loss_op])
#	print (loss_op)
#	print ("========")
#	print (network_dict['new_output_tensor_of'][last_layer_name])
#	print ("\n")
#	for t in t_ws + [ys]:
#		print (t)
#	print ("++++++")
#	for k,v in network_dict['new_output_tensor_of'].items():
#		print (k, v)
#	print ("-----")
#	for k,v in network_dict['input_layers_of'].items():
#		print (k, v)
	return fn, t_ws, ys
	

def build_tf_model(model, X, indices_to_target_layers):
	"""
	"""
	import numpy as np
	import tensorflow.keras.backend as K
	from tensorflow.keras.models import load_model, Model
	from run_localise import is_FC, is_C2D

	targeting_clname_pattns = ['Dense*', 'Conv*'] #if not target_all else None
	is_target = lambda clname, targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])

	idx_min_tl = np.min(indices_to_target_layers)

	t_model = Model(inputs = model.input, outputs = model.layers[idx_min_tl].output)
	output_of_front = t_model.predict(X)

	#tf.constant(output_of_front)	
	for idx_to_l, layer in enumerate(t_model.layers):
		org_idx_to_l = idx_to_l + idx_min_tl
		layer_config = layer.get_config()
		l_class_name = type(layer).__name__

		if org_idx_to_l in indices_to_target_layers:
			# use placeholder
			#a = K.placeholder(shape=(None,), dtype='int32')
			if is_target(l_class_name, targeting_clname_pattns): # is our target
				if is_FC(l_class_name):
					# ... Linear s....
					pass
				else:
					assert is_C2D(l_class_name), "{}".format(l_class_name)
			else:
				msg = "{}th layer {} is not our target".format(org_idx_to_l, l_class_name)
				assert False, msg
				# ... 
				

			pass
		else: # currently supporting only conv2d and dense
			# generate the same layer from config.... 
			if is_target(l_class_name, targeting_clname_pattns): # is our target
				if is_FC(l_class_name):
					# ... Linear s....
					pass
				else:
					assert is_C2D(l_class_name), "{}".format(l_class_name)
			else:
				msg = "{}th layer {} is not our target".format(org_idx_to_l, l_class_name)
				assert False, msg
	pass


def build_keras_model_front_v2(path_to_keras_model, idx_to_target_w = -1):
	"""
	idx_to_target_w = to which layer (default = -1 (the last one)) - 1 
	"""
	from tensorflow.keras.models import load_model, Model

	idx = 0
	loaded_model = load_model(path_to_keras_model)
	front_layers = loaded_model.layers[:idx_to_target_w]

	model = Model(inputs = front_layers[0].input, outputs = front_layers[-1].output)

	return model, front_layers


def build_torch_model_front_v2(path_to_torch_model, idx_to_target_w = -1):
	"""
	"""
	import torch
	import torch.nn as nn
	import torchvision.models as models

	vgg_model = models.vgg19(pretrained=True)
	vgg_model.classifier[6] = nn.Linear(4096, 2)
	model = vgg_model.load_state_dict(torch.load(path_to_keras_model))

	model.classifier = model.classifier[:idx_to_target_w]
	model.cuda()
	model.eval()

	return model



def build_simple_fm_tf_mode(path_to_keras_model, inputs, weight_shape, w_gather = True):
	"""
	build an emtpy graph for models trained with CIFAR-10
	Ret (tf.Graph):
		return a build tf.Graph instance
	"""
	model, _ = build_keras_model_front_v2(path_to_keras_model)
	#model.summary()
	outputs = model.predict(inputs)

	empty_graph = build_fm_graph_only_the_last(outputs, 
		num_label = 10, weight_shape = weight_shape,
		w_gather = w_gather)

	return empty_graph


def build_simple_cm_tf_mode(path_to_keras_model, inputs, weight_shape, w_gather = True):
	"""
	build an emtpy graph for models trained with FM
	Ret (tf.Graph):
		return a build tf.Graph instance
	"""
	model, _ = build_keras_model_front_v2(path_to_keras_model)
	outputs = model.predict(inputs)

	empty_graph = build_cpm_graph_only_the_last(outputs,
		num_label = 10, 
		weight_shape = weight_shape, 
		name_of_target_weight = "fw3",
		w_gather = w_gather)

	return empty_graph


def build_lfw_tf_mode(path_to_torch_model, 
	inputs, 
	weight_shape, 
	w_gather = True, 
	is_train = True,
	indices = None,
	use_raw = False):
	"""
	build an empty graph for RQ6
	Ret (tf.Graph):
		return a build tf.Graph instance
	"""
	from utils.data_util import get_lfw_data
	import numpy as np

	data_dict = get_lfw_data(is_train = is_train)
	outputs = data_dict['at']
	if indices is not None: 
		if is_train or use_raw: # normal usage
			outputs = outputs[indices]
		else: # test -> should go back to the original indices (for the experiement)
			outputs = outputs[2 * np.asarray(indices)]
	else:
		if not is_train: 
			outputs = outputs[::2]
			
	empty_graph = build_cpm_graph_only_the_last(outputs,
		num_label = 2, 
		weight_shape = weight_shape, 
		name_of_target_weight = "fw3",
		w_gather = w_gather)

	return empty_graph


def build_cpm_graph_only_the_last(featuresnp,
	num_label = 10, 
	weight_shape = 4096,
	name_of_target_weight = "fw3",
	w_gather = True):#None):
	"""
	build an emtpy graph for models trained with C10
	Ret (tf.Graph):
		return a build tf.Graph instance
	"""
	import tensorflow as tf

	graph = tf.Graph()
	with graph.as_default():
		labels = tf.placeholder(tf.float32, shape = [None, num_label], name = "labels")
		
		if w_gather:
			featuresnp_const = tf.constant(featuresnp, dtype = tf.float32, name = 'vgg16_output_const')
			indices_to_slice = tf.placeholder(tf.int32, shape = (None,), name = 'indices_to_slice')
			if featuresnp_const.shape[0] != tf.shape(indices_to_slice)[0]:
				target_slice = tf.gather(featuresnp_const, indices_to_slice, name = "Gathered") 
			else:
				target_slice = tf.identity(featuresnp_const, name="Gathered")
		else:
			target_slice = tf.constant(featuresnp, dtype = tf.float32, name = "Gathered")#	

		# Logits Layer
		fb3 = tf.placeholder(tf.float32, shape = [num_label], name = "fb3")
		w3 = tf.placeholder(tf.float32, shape = [weight_shape, num_label], name = "fw3")
		logits = tf.add(tf.matmul(target_slice, w3), fb3, name = 'logits')	

		# add gradient to logits
		predcs = tf.nn.softmax(logits, name = "predc")

		# Define loss and optimiser
		per_label_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
			logits = logits, 
			labels = labels, 
			name = "per_label_loss",
			dim = -1) 
			
		loss_op = tf.reduce_mean(per_label_loss, name = 'loss')	

		doutput_dw_tensor_lst = []
		for idx_to_slice in range(num_label):
			partial_doutput_dw = tf.gradients(predcs[:,idx_to_slice], w3, name = 'partial_doutput_dw_{}'.format(idx_to_slice))
			doutput_dw_tensor_lst.append(partial_doutput_dw)
			
		doutput_dw = tf.concat(doutput_dw_tensor_lst, 0, name = "doutput_dw") # axis = 0
		dloss_dw3 = tf.gradients(loss_op, w3, name = 'dloss_dw3')

		correct_predc = tf.equal(tf.argmax(predcs, 1), tf.argmax(labels, 1), name = "correct_predc")
		acc_op = tf.reduce_mean(tf.cast(correct_predc, tf.float32), name = "acc")

	return graph


def build_fm_graph_only_the_last(featuresnp, num_label = 10, weight_shape = 100, w_gather = True):
	"""
	Ret (tf.Graph):
		return a build tf.Graph instance
	"""
	import tensorflow as tf

	graph = tf.Graph()
	with graph.as_default():
		labels = tf.placeholder(tf.float32, shape = [None, num_label], name = "labels")
	
		if w_gather:
			featuresnp_const = tf.constant(featuresnp, dtype = tf.float32, name = 'fm_output_const')
			indices_to_slice = tf.placeholder(tf.int32, shape = (None,), name = 'indices_to_slice')

			if featuresnp_const.shape[0] != tf.shape(indices_to_slice)[0]:
				target_slice_0 = tf.gather(featuresnp_const, indices_to_slice, name = "Gathered_0")
				target_slice = tf.reshape(target_slice_0, [-1, target_slice_0.shape[-1]], name = "Gathered")
			else:
				target_slice_0 = tf.identity(featuresnp_const, name="Gathered_0")
				target_slice = tf.reshape(target_slice_0, [-1, target_slice_0.shape[-1]], name = "Gathered")
		else:
			target_slice_0 = tf.constant(featuresnp, dtype = tf.float32, name = 'Gathered_0')
			target_slice = tf.reshape(target_slice_0, [-1, target_slice_0.shape[-1]], name = "Gathered")

		### from hereafter, the placeholder parts (modifiable part)
		### add the same structure here 
		# Logits Layer
		fb3 = tf.placeholder(tf.float32, shape = [num_label], name = "fb3") 
		w3 = tf.placeholder(tf.float32, shape = [weight_shape, num_label], name = "fw3")
		logits = tf.add(tf.matmul(target_slice, w3), fb3, name = 'logits')	

		# add gradient to logits
		predcs = tf.nn.softmax(logits, name = "predc")

		# Define loss and optimiser
		per_label_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
			logits = logits, 
			labels = labels, 
			name = "per_label_loss",
			dim = -1) 

		loss_op = tf.reduce_mean(per_label_loss, name = 'loss')	

		doutput_dw_tensor_lst = []
		for idx_to_slice in range(num_label): # compute for each label
			partial_doutput_dw = tf.gradients(predcs[:,idx_to_slice], w3, name = 'partial_doutput_dw_{}'.format(idx_to_slice))
			doutput_dw_tensor_lst.append(partial_doutput_dw)
			
		doutput_dw = tf.concat(doutput_dw_tensor_lst, 0, name = "doutput_dw") # axis = 0
		dloss_dw3 = tf.gradients(loss_op, w3, name = 'dloss_dw3')

		# Evaluate model
		correct_predc = tf.equal(tf.argmax(predcs, 1), tf.argmax(labels, 1), name = "correct_predc")
		acc_op = tf.reduce_mean(tf.cast(correct_predc, tf.float32), name = "acc")

	return graph



def generate_empty_graph(which, 
	inputs, 
	num_label, 
	path_to_keras_model = None, 
	w_gather = True,
	indices = None,
	is_train = False,
	use_raw = False):
	"""
	Prepare an empty graph frame for a given model for patching.
	"""
	import time
	assert path_to_keras_model is not None

	from utils import apricot_rel_util
	from tensorflow.keras.models import load_model

	if which == 'cnn1':
		empty_graph = build_simple_cm_tf_mode(path_to_keras_model, inputs, 256, w_gather = w_gather)
	elif which == 'cnn2':
		empty_graph = build_simple_cm_tf_mode(path_to_keras_model, inputs, 2048, w_gather = w_gather)
	elif which == 'cnn3':
		empty_graph = build_simple_cm_tf_mode(path_to_keras_model, inputs, 10, w_gather = w_gather)
	elif which == 'simple_fm':
		empty_graph = build_simple_fm_tf_mode(path_to_keras_model, inputs, 100, w_gather = w_gather)
	elif which == 'simple_cm':
		empty_graph = build_simple_cm_tf_mode(path_to_keras_model, inputs, 512, w_gather = w_gather)
	else: # lfw_vgg
		empty_graph = build_lfw_tf_mode(path_to_keras_model, inputs, 4096, w_gather = w_gather, 
			indices = indices, is_train = is_train, use_raw = use_raw) 

	return empty_graph




