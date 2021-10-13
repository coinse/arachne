"""
methods related to generating and using a keras backend function for a model,
"""
import numpy as np
#BATCH_SIZE = 5096

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


#def compute_kfunc(k_fn_mdl_lst, ys, tws, idx, batch_size = None):
#	"""
#	comptue k functon for ys and tws 
#	"""
#	if len(k_fn_mdl_lst) == 1:
#		k_fn_mdl = k_fn_mdl_lst[0]
#		outputs = k_fn_mdl(tws + [ys])[idx]
#	else:
#		num = len(ys)
#		chunks = return_chunks(num, batch_size)
#		outputs = None
#		for k_fn_mdl, chunk in zip(k_fn_mdl_lst, chunks):
#			a_outputs = k_fn_mdl(tws + [ys[chunk]])[idx]
#			if outputs is None:
#				outputs = a_outputs
#			else:
#				outputs = np.append(outputs, a_outputs, axis = 0)
#	return outputs


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
