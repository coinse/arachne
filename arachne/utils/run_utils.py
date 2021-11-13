"""
to apply stored patches to a model
"""
from utils import model_util
from gen_frame_graph import generate_empty_graph
import numpy as np
import tensorflow as tf

def get_weights(model):
	"""
	"""
	kernel_and_bias_pairs = []
	ws = model.get_weights()
	for i, w in enumerate(ws):
		if i % 2 == 0: 
			if len(w.shape) == 4:
				kernel_and_bias_pairs.append([np.transpose(w, (1,2,0,3))])
			else:
				kernel_and_bias_pairs.append([w])
		else: # bias
			kernel_and_bias_pairs[-1].append(w)
	
	return kernel_and_bias_pairs


def run_model(mdl, X, y, which_data):
	"""
	"""
	import pandas as pd

	predcs = mdl.predict(X) if which_data != 'fashion_mnist' else mdl.predict(X).reshape(len(X),-1)
	pred_labels = np.argmax(predcs, axis = 1)
	
	aft_preds = []
	aft_preds_column = ['index', 'true', 'pred', 'flag']
	for i, (true_label, pred_label) in enumerate(zip(y, pred_labels)):
		aft_preds.append([i, true_label, pred_label, true_label == pred_label])
	
	aft_pred_df = pd.DataFrame(aft_preds, columns = aft_preds_column)
	return aft_pred_df

def gen_and_run_model(mdl, path_to_patch, X, y, num_label, input_reshape = False, need_act = False, batch_size = None):
	"""
	"""
	import pandas as pd
	import utils.kfunc_util as kfunc_util
	import utils.data_util as data_util

	act_func = tf.nn.relu if need_act else None
	patch = pd.read_pickle(path_to_patch)
	indices_to_tls = sorted(list(patch.keys()))
	
	k_fn_mdl_lst = kfunc_util.generate_base_mdl(
		mdl, X, indices_to_tls = indices_to_tls, batch_size = batch_size, act_func = act_func)

	formated_y = data_util.format_label(y, num_label)	
	predictions = kfunc_util.compute_kfunc(
		k_fn_mdl_lst, formated_y, [patch[idx] for idx in indices_to_tls], batch_size = batch_size)[0]	
	if len(predictions.shape) == 3:
		predictions = np.squeeze(predictions,axis = 1)
	pred_labels = np.argmax(predictions, axis = 1)

	aft_preds = []
	aft_preds_column = ['index', 'true', 'pred', 'flag']
	for i, (true_label, pred_label) in enumerate(zip(y, pred_labels)):
		aft_preds.append([i, true_label, pred_label, true_label == pred_label])
	
	aft_pred_df = pd.DataFrame(aft_preds, columns = aft_preds_column)
	return aft_pred_df
	


def record_predcs(y, pred_labels, filename, target_indices):
	"""
	"""
	import csv
	filename = filename.replace(".json", "")
	with open(filename, 'w') as f:
		csvWriter = csv.writer(f)
		csvWriter.writerow(['index', 'true', 'pred'])
		for idx, true_label, pred_label in zip(target_indices, y, pred_labels):
			csvWriter.writerow([idx, true_label, pred_label])

