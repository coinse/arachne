"""
contain model handling functions:
	evaluate, load, update
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import time

def generate_session(graph = None):
	"""
	Generat session using the graph
	"""
	if graph is not None:
		sess = tf.Session(graph = graph)
	else:
		sess = tf.Session()

	return sess


def get_tensor(tensor_name, graph, sess = None):
	"""
	"""
	if graph is not None:
		target_tensor = graph.get_tensor_by_name("%s:0" % tensor_name)
	else:
		if sess is not None:
			target_tensor = sess.graph.get_tensor_by_name("%s:0" % tensor_name)
		else:
			target_tensor = tf.get_default_graph().get_tensor_by_name("%s:0" % tensor_name)

	return target_tensor


def run(
	tensor_name_lst,
	input_data, 
	output_data, 
	#keep_prob_val = 1.0,
	input_tensor_name = "inputs", 
	output_tensor_name = "labels", 
	#keep_prob_name = "keep_prob",
	sess = None, 
	empty_graph = None,
	plchldr_feed_dict = None):
	"""
	Ret:
		return the result of running the tensors
	"""
	#t0 = time.time()
	if sess is None:
		graph = empty_graph
		sess = tf.Session(graph = graph) # initialise the session with a given graph
	else:
		graph = empty_graph
	
	feed_dict = {}
	zipped_tensor_name_and_value = zip(\
		[input_tensor_name, output_tensor_name],#, keep_prob_name],
		[input_data, output_data])#, keep_prob_val])

	for tensor_name, tensor_value in zipped_tensor_name_and_value:
		if tensor_name is not None and tensor_value is not None:
			target_tensor = graph.get_tensor_by_name("%s:0" % (tensor_name))
			feed_dict[target_tensor] = tensor_value

	if plchldr_feed_dict is not None:
		feed_dict.update(plchldr_feed_dict)

	tensor_lst = []
	for tensor_name in tensor_name_lst:
		if isinstance(tensor_name, str):
			tensor = graph.get_tensor_by_name("%s:0" % (tensor_name))
		else: # is tensor
			tensor = tensor_name 
		tensor_lst.append(tensor)
	
	if bool(feed_dict):
		results = sess.run(tensor_lst, feed_dict = feed_dict)
	else:
		results = sess.run(tensor_lst)

	return results, sess


def compute_gradient_new(
	num_label,
	loss_tensor_name,
	weight_tensor_name,
	input_data, output_data,
	input_tensor_name = "inputs", output_tensor_name = "labels", 
	indices_to_slice_tensor_name = "indices_to_slice",
	sess = None,
	empty_graph = None,
	plchldr_feed_dict = None,
	base_indices_to_cifar10 = None, 
	use_pretr_front = False):
	"""
	gradient_tensor_name: e.g., dloss_dw2/MatMul_1_grad/MatMul_1, dlab_dw3/MatMul_2_grad/MatMul_1:0
	"""
	# Run predict_tensor_name and corr_predict_tensor_name. 
	# For the reamining arguments, the arguments for feed_dict,
	# use the default value
	if use_pretr_front:
		if indices_to_slice_tensor_name is not None:
			indices_to_slice_vgg16_tensor = empty_graph.get_tensor_by_name('%s:0' % (indices_to_slice_tensor_name))
			if base_indices_to_cifar10 is None:
				plchldr_feed_dict[indices_to_slice_vgg16_tensor] = list(range(len(output_data)))
			else:
				plchldr_feed_dict[indices_to_slice_vgg16_tensor] = base_indices_to_cifar10
			input_tensor_name = None
		
		input_tensor_name = None
	else:
		input_tensor_name = 'inputs'

	loss_tensor = empty_graph.get_tensor_by_name('%s:0' % (loss_tensor_name))
	weight_tensor = empty_graph.get_tensor_by_name('%s:0' % (weight_tensor_name))

	dlab_dw3 = tf.gradients(loss_tensor, weight_tensor, name = 'dlab_dw3')
	gradient, sess  = run(
		dlab_dw3,
		input_data, 
		output_data, 
		#keep_prob_val = 1.0,
		input_tensor_name = input_tensor_name,
		output_tensor_name = output_tensor_name, 
		#keep_prob_name = keep_prob_name,
		sess = sess, 
		empty_graph = empty_graph,
		plchldr_feed_dict = plchldr_feed_dict)
	
	return gradient, sess


#def compute_ats(
def get_output_vector(
	num_label,
	activation_tensor_name,
	input_data, #keep_prob_val = 1.0,
	input_tensor_name = "inputs", output_tensor_name = "labels", 
	#keep_prob_name = "keep_prob", 
	indices_to_slice_tensor_name = 'indices_to_slice',
	sess = None,
	empty_graph = None,
	plchldr_feed_dict = None,
	use_pretr_front = False,
	base_indices_to_cifar10 = None):
	"""
	activation_tensor_name: e.g., Relu, Relu_1, Relu_2, Relu_3
	"""
	# Run predict_tensor_name and corr_predict_tensor_name. 
	# For the reamining arguments, the arguments for feed_dict,
	# use the default value
	if isinstance(activation_tensor_name, list):
		activation_tensor_names = activation_tensor_name
	else:
		activation_tensor_names = [activation_tensor_name]

	indices_lst = [list(range(len(input_data)))]

	ats = None
	for indices in indices_lst:
		# set current  plchldr_feed_dict
		if plchldr_feed_dict is not None:
			curr_plchldr_feed_dict = plchldr_feed_dict.copy()
		else:
			curr_plchldr_feed_dict = {}

		if use_pretr_front:
			if indices_to_slice_tensor_name is not None:

				indices_to_slice_vgg16_tensor = empty_graph.get_tensor_by_name('%s:0' % (indices_to_slice_tensor_name))
				if base_indices_to_cifar10 is None:
					curr_plchldr_feed_dict[indices_to_slice_vgg16_tensor] = indices
				else:
					curr_plchldr_feed_dict[indices_to_slice_vgg16_tensor] = np.asarray(base_indices_to_vgg16)[indices]

				input_tensor_name = None
			else:
				if indices_to_slice_tensor_name in curr_plchldr_feed_dict.keys():
					del curr_plchldr_feed_dict[indices_to_slice_tensor_name]
			input_tensor_name = None
		else:
			input_tensor_name = 'inputs'

		_ats, sess  = run(
			activation_tensor_names,
			input_data[indices], None, #keep_prob_val = 1.0,
			input_tensor_name = input_tensor_name,
			output_tensor_name = output_tensor_name, 
			#keep_prob_name = keep_prob_name,
			sess = sess, 
			empty_graph = empty_graph,
			plchldr_feed_dict = curr_plchldr_feed_dict)

		_ats = np.asarray(_ats)
		_ats = _ats.reshape(_ats.shape[1:]) # this actually

		if ats is None:
			ats = _ats
		else:
			ats = np.append(ats, _ats, axis = 0)
			
	return ats, sess


def predict(
	test_X, test_y, num_label,
	predict_tensor_name = "predc", 
	corr_predict_tensor_name = "correct_predc",
	indices_to_slice_tensor_name = 'indices_to_slice',
	sess = None, 
	empty_graph = None,
	plchldr_feed_dict = None,
	use_pretr_front = False,
	compute_loss = False,
	base_indices_to_cifar10 = None):
	"""
	"""
	#t0 = time.time()
	indices_lst = [list(range(len(test_y)))]
	
	if predict_tensor_name is not None and corr_predict_tensor_name is not None:
		tensors_to_run = [predict_tensor_name, corr_predict_tensor_name] 
	elif predict_tensor_name is None:
		tensors_to_run = [corr_predict_tensor_name] 
	elif corr_predict_tensor_name is None:
		tensors_to_run = [predict_tensor_name] 

	if compute_loss:
		tensors_to_run += ['per_label_loss']

	predictions = None; correct_predictions = None; per_label_losses = None
	for indices in indices_lst:
		#t1 = time.time()
		# set current feed_dict
		if plchldr_feed_dict is not None:
			curr_plchldr_feed_dict = plchldr_feed_dict.copy()
		else:
			curr_plchldr_feed_dict = {}

		if use_pretr_front: ## LOOK AGAIN
			if indices_to_slice_tensor_name is not None:
				indices_to_slice_vgg16_tensor = empty_graph.get_tensor_by_name('%s:0' % (indices_to_slice_tensor_name))
				if base_indices_to_cifar10 is None:
					curr_plchldr_feed_dict[indices_to_slice_vgg16_tensor] = indices
				else:
					curr_plchldr_feed_dict[indices_to_slice_vgg16_tensor] = np.asarray(base_indices_to_cifar10)[indices]
				input_tensor_name = None
			else:
				if indices_to_slice_tensor_name in curr_plchldr_feed_dict.keys():
					del curr_plchldr_feed_dict[indices_to_slice_tensor_name]

			input_tensor_name = None
		else:
			input_tensor_name = 'inputs'

		vs, sess = run(
			tensors_to_run,
			test_X[indices] if test_X is not None else None, 
			test_y[indices] if test_y is not None else None, 
			input_tensor_name = input_tensor_name, 
			sess = sess, 
			empty_graph = empty_graph,
			plchldr_feed_dict = curr_plchldr_feed_dict)

		if not compute_loss:
			if predict_tensor_name is not None and corr_predict_tensor_name is not None:
				_predictions, _correct_predictions = vs
			elif predict_tensor_name is None:
				_correct_predictions = vs[0]
				_predictions = None
			elif corr_predict_tensor_name is None:
				_predictions = vs[0]
				_correct_predictions = None	
		else:
			if predict_tensor_name is not None and corr_predict_tensor_name is not None:
				_predictions, _correct_predictions, _per_label_loss = vs
			elif predict_tensor_name is None:
				_correct_predictions, _per_label_loss = vs
				_predictions = None
			elif corr_predict_tensor_name is None:
				_predictions, _per_label_loss = vs
				_correct_predictions = None

			## loss update
			if per_label_losses is None:
				per_label_losses = _per_label_loss
			else:
				per_label_losses = np.append(per_label_losses, _per_label_loss, axis = 0)

		## prediction update
		if _predictions is not None:
			if predictions is None:
				predictions = _predictions
			else:
				predictions = np.append(predictions, _predictions, axis = 0)

		## corr_prediction update
		if _correct_predictions is not None:
			if correct_predictions is None:
				correct_predictions = _correct_predictions
			else:
				correct_predictions = np.append(correct_predictions, _correct_predictions, axis = 0)
	#t_end = time.time()
	#print ("Time for pred: {}".format(t_end - t0))
	if not compute_loss:
		return sess, (predictions, correct_predictions)
	else:
		return sess, (predictions, correct_predictions, per_label_losses)

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


def is_Input(lname):
	"""
	"""
	import re
	pattns = ['InputLayer']
	return any([bool(re.match(t,lname)) for t in pattns])


def is_Attention(lname):
	"""
	"""
	import re
	pattns = ['LSTM']
	return any([bool(re.match(t,lname)) for t in pattns])


def generate_base_mdl(mdl, X, indices_to_tls = None, batch_size = None): 
	from gen_frame_graph import build_mdl_lst
	from utils.data_util import return_chunks
	
	indices_to_tls = sorted(indices_to_tls)
	if batch_size is None:
		mdl = build_mdl_lst(mdl, X, indices_to_tls)
		mdl_lst = [mdl]
	else:
		chunks = return_chunks(len(X), batch_size = batch_size)
		mdl_lst = []
		for chunk in chunks:
			mdl = build_mdl_lst(mdl, X[chunk], indices_to_tls)
			mdl_lst.append(mdl)

	return mdl_lst

def compute_pred_and_loss(mdl_lst, ys, tws, batch_size = None):
	"""
	comptue k functon for ys and tws 
	"""
	append_vs = lambda vs_1, vs_2: vs_2 if vs_1 is None else np.append(vs_1, vs_2, axis = 0)
	if len(mdl_lst) == 1:
		mdl = mdl_lst[0]
		#predictions = mdl()
	else:
		from utils.data_util import return_chunks
		num = len(ys)
		chunks = return_chunks(num, batch_size)

		outputs_1 = None; outputs_2 = None
		for mdl, chunk in zip(mdl_lst, chunks):
			a_outputs_1, a_outputs_2 = mdl(tws + [ys[chunk]])
			outputs_1 = append_vs(outputs_1, a_outputs_1)
			outputs_2 = append_vs(outputs_2, a_outputs_2)
		outputs = [outputs_1, outputs_2]
	return outputs
