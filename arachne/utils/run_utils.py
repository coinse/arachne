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
	

def gen_and_run_model(which, X, y, num_label, model, kernel_and_bias_pairs, 
	sess = None, is_train = False, indices = None, use_raw = False):
	"""
	"""
	import tensorflow as tf
	empty_graph = generate_empty_graph(which, 
		X, 
		num_label, 
		path_to_keras_model = model, 
		w_gather = False,
		is_train = is_train,
		indices = indices,
		use_raw = use_raw)
		
	if sess is None:
		sess = tf.Session(graph = empty_graph)	

	y = np.eye(num_label)[y]
	sess, (predictions, correct_predictions) = model_util.predict(
		None, y, num_label,
		predict_tensor_name = "predc", 
		corr_predict_tensor_name = "correct_predc",
		indices_to_slice_tensor_name = None,
		sess = sess, 
		empty_graph = empty_graph,
		plchldr_feed_dict =  {'fw3:0':np.float32(kernel_and_bias_pairs[-1][0]), 'fb3:0':kernel_and_bias_pairs[-1][1]},
		use_pretr_front = True,
		compute_loss = False)

	pred_labels = np.argmax(predictions, axis = 1)
	num = len(correct_predictions)
	corr_cnt = np.sum(correct_predictions)

	return pred_labels, corr_cnt, num, sess


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

