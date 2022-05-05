"""
methods related to generating and using a keras backend function for a model
"""
import numpy as np

def compute_predictions(k_fn_mdl_lst, ys, tws, batch_size = None, into_label = False):
	"""
	"""
	predictions = compute_kfunc(k_fn_mdl_lst, ys, tws, batch_size)[0]
	if into_label:
		predictions = np.argmax(predictions, axis = 1)
	return predictions
	

def compute_losses(k_fn_mdl_lst, ys, tws, batch_size = None):
	"""
	"""
	losses = compute_kfunc(k_fn_mdl_lst, ys, tws, batch_size)[1]
	return losses


def compute_preds_and_losses(k_fn_mdl_lst, ys, tws, batch_size = None):
	"""
	"""
	preds_and_losses = compute_kfunc(k_fn_mdl_lst, ys, tws, batch_size = batch_size)
	return preds_and_losses


def compute_kfunc(k_fn_mdl_lst, ys, tws, batch_size = None):
	"""
	comptue k functon for ys and tws 
	"""
	append_vs = lambda vs_1, vs_2: vs_2 if vs_1 is None else np.append(vs_1, vs_2, axis = 0)
	if len(k_fn_mdl_lst) == 1:
		k_fn_mdl = k_fn_mdl_lst[0]
		outputs = k_fn_mdl(tws + [ys])
	else:
		num = len(ys)
		chunks = return_chunks(num, batch_size)
		outputs_1 = None; outputs_2 = None
		for k_fn_mdl, chunk in zip(k_fn_mdl_lst, chunks):
			a_outputs_1, a_outputs_2 = k_fn_mdl(tws + [ys[chunk]])
			outputs_1 = append_vs(outputs_1, a_outputs_1)
			outputs_2 = append_vs(outputs_2, a_outputs_2)
		outputs = [outputs_1, outputs_2]
	return outputs


def return_chunks(num, batch_size = None):
	num_split = int(np.round(num/batch_size))
	if num_split == 0:
		num_split = 1
	chunks = np.array_split(np.arange(num), num_split)
	return chunks


def generate_base_mdl(mdl, X, indices_to_tls = None, batch_size = None, act_func = None):
	from utils.gen_frame_graph import build_k_frame_model
	
	indices_to_tls = sorted(indices_to_tls)
	if batch_size is None:	
		k_fn_mdl, _, _  = build_k_frame_model(mdl, X, indices_to_tls, act_func = act_func)
		k_fn_mdl_lst = [k_fn_mdl]
	else:
		num = len(X)
		chunks = return_chunks(num, batch_size = batch_size)
		k_fn_mdl_lst = []
		for chunk in chunks:
			k_fn_mdl, _, _  = build_k_frame_model(mdl, X[chunk], indices_to_tls, act_func = act_func)
			k_fn_mdl_lst.append(k_fn_mdl)

	return k_fn_mdl_lst


def gen_pred_and_loss_ops(pred_shape, pred_dtype, y_shape, y_dtype, loss_func):
	"""
	"""
	import tensorflow as tf
	import tensorflow.keras.backend as K

	pred_tensor = tf.keras.Input(dtype = pred_dtype, shape = pred_shape[1:])
	pred_probs = tf.math.softmax(pred_tensor) 
	y_tensor = tf.keras.Input(dtype = y_dtype, shape = y_shape[1:]) 

	if len(pred_probs.shape) > 2:
		pred_probs = tf.reshape(pred_probs, [-1] + list(y_shape[1:]))
		
	if loss_func == 'categorical_cross_entropy':
		loss_op = tf.keras.metrics.categorical_crossentropy(y_tensor, pred_probs)	
	elif loss_func == 'binary_crossentropy':
		pred_shape = pred_probs.shape
		assert len(pred_shape) in [1,2], pred_shape
		if len(pred_shape) == 1:
			probs_to_be_zero = tf.ones_like(pred_probs) - pred_probs
			pred_probs_2d = tf.stack([probs_to_be_zero, pred_probs], axis = 1)
			y_zero_tensor = tf.ones_like(y_tensor) - y_tensor	
			y_tensor_2d = tf.stack([y_zero_tensor, y_tensor], axis = 1)

			loss_op = tf.keras.metrics.binary_crossentropy(y_tensor_2d, pred_probs_2d)
		else:
			loss_op = tf.keras.metrics.binary_crossentropy(y_tensor, pred_probs)	
	else:
		print ("{} not supported yet".format(loss_func))

	fn = K.function([pred_tensor, y_tensor], [loss_op])
	return fn
