"""
to apply stored patches to a model
"""
from utils import model_util
import numpy as np
import tensorflow as tf

def run_model(mdl, X, y, is_multi_label = True, ret_raw = False):
	"""
	"""
	import pandas as pd
	
	predcs = mdl.predict(X)  
	if len(predcs.shape) >= 3:
		predcs = np.squeeze(predcs, axis = 1)
	if is_multi_label:
		pred_labels = np.argmax(predcs, axis = 1)
	else:
		pred_labels = np.round(predcs).flatten()
		predcs = predcs.flatten()
		
	aft_preds = []
	if not ret_raw:
		aft_preds_column = ['index', 'true', 'pred', 'flag']
	else:
		aft_preds_column = ['index', 'true', 'pred', 'pred_v', 'flag']
	for i, (true_label, pred_label) in enumerate(zip(y, pred_labels)):
		if not ret_raw:
			aft_preds.append([i, true_label, pred_label, true_label == pred_label])
		else:
			aft_preds.append([i, true_label, pred_label, predcs[i], true_label == pred_label])
	aft_pred_df = pd.DataFrame(aft_preds, columns = aft_preds_column)
	return aft_pred_df


def gen_and_run_model(mdl, path_to_patch, 
	X, y, num_label, 
	has_lstm_layer = False, is_multi_label = True, 
	need_act = False, batch_size = None):
	"""
	** this is the part that should be fixed (this is a temporary fix)
	"""
	import pandas as pd
	import utils.kfunc_util as kfunc_util
	import utils.data_util as data_util
	from collections.abc import Iterable

	act_func = tf.nn.relu if need_act else None
	patch = pd.read_pickle(path_to_patch)
	indices_to_tls = sorted(
		list(patch.keys()), key = lambda v:v[0] if isinstance(v, Iterable) else v)
	if is_multi_label:
		formated_y = data_util.format_label(y, num_label)	
	else:
		formated_y = y

	if not has_lstm_layer:
		k_fn_mdl_lst = kfunc_util.generate_base_mdl(
			mdl, X, 
			indices_to_tls = indices_to_tls, 
			batch_size = batch_size, 
			act_func = act_func)

		predictions = kfunc_util.compute_kfunc(
			k_fn_mdl_lst, formated_y, 
			[patch[idx] for idx in indices_to_tls], 
			batch_size = batch_size)[0]	
	else:
		from utils.gen_frame_graph import build_mdl_lst
		from tensorflow.keras.models import Model
		
		# compute previous outputs
		min_idx_to_tl = np.min(
			[idx if not isinstance(idx, Iterable) else idx[0] for idx in indices_to_tls])
		prev_l = mdl.layers[min_idx_to_tl-1 if min_idx_to_tl > 0 else 0]
		if model_util.is_Input(type(prev_l).__name__): # previous layer is an input layer
			prev_outputs = X
		else: # otherwise, compute the output of the previous layer
			t_mdl = Model(inputs = mdl.input, outputs = prev_l.output)	
			prev_outputs = t_mdl.predict(X)
		##
		k_fn_mdl = build_mdl_lst(mdl, prev_outputs.shape[1:],indices_to_tls)
		init_weights = {}
		init_biases = {}
		for idx_to_tl in indices_to_tls:
			idx_to_tl = idx_to_tl[0] if isinstance(idx_to_tl, tuple) else idx_to_tl
			ws = mdl.layers[idx_to_tl].get_weights()
			lname = type(mdl.layers[idx_to_tl]).__name__
			if model_util.is_FC(lname) or model_util.is_C2D(lname):
				init_weights[idx_to_tl] = ws[0]
				init_biases[idx_to_tl] = ws[1]
			elif model_util.is_LSTM(lname):
				# get only the kernel and recurrent kernel, not the bias
				for i in range(2):
					init_weights[(idx_to_tl, i)] = ws[i]
				init_biases[idx_to_tl] = ws[-1]
			else:
				print ("Not supported layer: {}".format(lname))
				assert False

		chunks = data_util.return_chunks(len(X), batch_size = batch_size)
		predictions = model_util.predict_with_new_delat(
			k_fn_mdl, patch, min_idx_to_tl, init_biases, init_weights, prev_outputs, chunks)

	if len(predictions.shape) > len(formated_y.shape) and predictions.shape[1] == 1:
		predictions = np.squeeze(predictions, axis = 1)

	if is_multi_label:
		pred_labels = np.argmax(predictions, axis = 1) 
	else:
		pred_labels = np.round(predictions).flatten()

	aft_preds = []
	aft_preds_column = ['index', 'true', 'pred', 'flag']
	for i, (true_label, pred_label) in enumerate(zip(y, pred_labels)):
		aft_preds.append([i, true_label, pred_label, true_label == pred_label])
	
	aft_pred_df = pd.DataFrame(aft_preds, columns = aft_preds_column)
	return aft_pred_df
