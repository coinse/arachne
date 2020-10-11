"""
empty graph (tf.Graph) generation script
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def build_keras_model_front_v2(path_to_keras_model):
	"""
	"""
	from tensorflow.keras.models import load_model, Model

	idx = 0
	loaded_model = load_model(path_to_keras_model)
	front_layers = loaded_model.layers[:-1]

	model = Model(inputs = front_layers[0].input, outputs = front_layers[-1].output)

	return model, front_layers


def build_torch_model_front_v2(path_to_torch_model):
	"""
	"""
	import torch
	import torch.nn as nn
	import torchvision.models as models

	vgg_model = models.vgg19(pretrained=True)
	vgg_model.classifier[6] = nn.Linear(4096, 2)
	model = vgg_model.load_state_dict(torch.load(path_to_keras_model))

	model.classifier = model.classifier[:-1]
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


def compute_fisher_infomat(predcs, w, ys, num_input):
	"""
	For EWC.
	Compute fisher information matrix
	"""
	ders_lst = []
	for i in range(num_input):
		idx_to_y = ys[i] - 1
		ders = tf.gradients(tf.log(predcs[i,idx_to_y]), w) # [i] or i
		ders_lst.append(ders)

	fisher_mat = tf.reduce_mean(tf.square(tf.concat(ders_lst, 0)), axis = 0, name = "fisher_mat") # axis = 0
	return fisher_mat



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
		#####
		fisher_info_mat = compute_fisher_infomat(predcs, w3, labels, featuresnp_const.shape[0])
		#####

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

		# Evaluate model
		correct_predc = tf.equal(tf.argmax(predcs, 1), tf.argmax(labels, 1), name = "correct_predc")
		acc_op = tf.reduce_mean(tf.cast(correct_predc, tf.float32), name = "acc")

		#####
		fisher_info_mat = compute_fisher_infomat(predcs, w3, labels, featuresnp.shape[0])
		#####

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




