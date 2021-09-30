"""
methods related to generating and using a keras backend function for a model,
"""
import numpy as np
#BATCH_SIZE = 5096

def compute_predictions(k_fn_mdl_lst, ys, tws):
	"""
	"""
	pred_probas = compute_kfunc(k_fn_mdl_lst, ys, tws)
	predictions = np.argmax(pred_probas, axis = 1)
	#print ("Final", predictions.shape)
	return predictions
	

def compute_losses(k_fn_mdl_lst, ys, tws):
	"""
	"""
	losses = compute_kfunc(k_fn_mdl_lst, ys, tws)
	return losses


def compute_kfunc(k_fn_mdl_lst, ys, tws):
	"""
	comptue k functon for ys and tws 
	"""
	if len(k_fn_mdl_lst) == 1:
		k_fn_mdl = k_fn_mdl_lst[0]
		outputs, _ = k_fn_mdl(tws + [ys])
	else:
		num = len(ys)
		chunks = return_chunks(num)
		outputs = None
		for k_fn_mdl, chunk in zip(k_fn_mdl_lst, chunks):
			a_outputs, _ = k_fn_mdl(tws + [ys[chunk]])
			if outputs is None:
				outputs = a_outputs
			else:
				outputs = np.append(outputs, a_outputs, axis = 0)
	return outputs


def return_chunks(num, batch_size = None):
	num_split = int(np.round(num/batch_size))
	if num_split == 0:
		num_split = 1
	chunks = np.array_split(np.arange(num), num_split)
	return chunks


def generate_base_mdl(mdl, X, indices_to_tls = None, batch_size = None, act_func = None):
	from gen_frame_graph import build_k_frame_model
	
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