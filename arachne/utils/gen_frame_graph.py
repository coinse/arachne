"""
empty graph (tf.Graph) generation script
"""
import os
from utils import model_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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
	if min_idx_to_tl == 0:
		layer_name = mdl.layers[0].name
		# if the previous layer (layer_name) is an input layer
		if model_util.is_Input(type(mdl.layers[0]).__name__): 
			# set model_input as the output of this input layer
			network_dict['new_output_tensor_of'].update({layer_name: model_input}) 
		else: # if it is not (happen when using Sequential())
			_input_layer_name = 'input_layer' # -> insert one addiitonal input layer
			# x is the output of _input_layer_name
			network_dict['new_output_tensor_of'].update({_input_layer_name: model_input}) 
			# the input's of layer_name is _input_layer_name
			network_dict['input_layers_of'].update({layer_name: [_input_layer_name]}) 
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
		# x is the output of the layer "layer_name" (current one)
		network_dict['new_output_tensor_of'].update({layer_name: x}) 

	last_layer_name = mdl.layers[-1].name
	mdl = Model(inputs = model_input, # this is a constant
		outputs = network_dict['new_output_tensor_of'][last_layer_name])
	return mdl


def build_k_frame_model(mdl, X, indices_to_tls, act_func = None):
	"""
	"""
	import tensorflow as tf
	from tensorflow.keras.models import Model
	import tensorflow.keras.backend as K
	import re
	import numpy as np
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

	min_idx_to_tl = np.min([idx if not isinstance(idx, Iterable) else idx[0] for idx in indices_to_tls])
	num_layers = len(mdl.layers)
	
	# Iterate over all layers after the input
	# Set the output tensor of the input layer (or more exactly, our starting point)
	if min_idx_to_tl == 0 or min_idx_to_tl - 1 == 0:
		x = tf.constant(X, dtype = tf.float32) #X.dtype)
		layer_name = mdl.layers[0].name
		network_dict['new_output_tensor_of'].update({layer_name: x})
	else:
		t_mdl = Model(inputs = mdl.input, outputs = mdl.layers[min_idx_to_tl-1].output)
		prev_output = t_mdl.predict(X)	
		dtype = mdl.layers[min_idx_to_tl-1].output.dtype
		x = tf.constant(prev_output, dtype = dtype) 

		network_dict['new_output_tensor_of'].update({mdl.layers[min_idx_to_tl-1].name: x})

	t_ws = []
	for idx_to_l in range(min_idx_to_tl, num_layers):
		layer = mdl.layers[idx_to_l]
		layer_name = layer.name
		
		# Determine input tensors
		layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
					   for layer_aux in network_dict['input_layers_of'][layer.name]]
		if len(layer_input) == 1:
			layer_input = layer_input[0]
			
		# Insert layer if name matches the regular expression
		if idx_to_l in indices_to_tls:
			l_class_name = type(layer).__name__
			if is_target(l_class_name, targeting_clname_pattns):
				if model_util.is_FC(l_class_name):
					w,b = layer.get_weights()
					t_w = tf.placeholder(dtype = w.dtype, shape = w.shape)    
					t_b = tf.constant(b)
					# this is a neat way, but the probelm is memory explosion... -> process by batches
					x = tf.add(tf.tensordot(
						layer_input, t_w, [[len(layer_input.shape)-1],[0]]), t_b, name = layer_name) 
					if act_func is not None:
						x = act_func(x)
					t_ws.append(t_w)
				else: # model_util.is_C2D(l_class_name):
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
			else:
				msg = "{}th layer {}({}) is not our target".format(idx_to_l, layer_name, l_class_name)
				assert False, msg
		else:
			x = layer(layer_input)

		# Set new output tensor (the original one, or the one of the replaced layer)
		# x is the output of the layer "layer_name" (current one)
		network_dict['new_output_tensor_of'].update({layer_name: x}) 

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