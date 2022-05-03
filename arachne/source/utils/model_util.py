"""
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np

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
	pattns = ['.*LSTM*']
	return any([bool(re.match(t,lname)) for t in pattns])

def is_Input(lname):
	"""
	"""
	import re
	pattns = ['InputLayer']
	return any([bool(re.match(t,lname)) for t in pattns])


def get_loss_func(is_multi_label = True):
	"""
	here, we will only return either cross_entropy or binary_crossentropy
	"""
	loss_func = 'categorical_cross_entropy' if is_multi_label else 'binary_crossentropy'
	return loss_func 


def predict_with_new_delat(
	fn_mdl, deltas, min_idx_to_tl, 
	init_biases, init_weights, 
	prev_outputs, chunks):
	"""
	predict with the model patched using deltas
	"""
	from collections.abc import Iterable
	for idx_to_tl, delta in deltas.items(): 
		# either idx_to_tl or (idx_to_tl, i)
		if isinstance(idx_to_tl, Iterable):
			idx_to_t_mdl_l, idx_to_w = idx_to_tl
		else:
			idx_to_t_mdl_l = idx_to_tl
	
		# index of idx_to_tl (from deltas) in the local model
		local_idx_to_l = idx_to_t_mdl_l - min_idx_to_tl + 1 
		lname = type(fn_mdl.layers[local_idx_to_l]).__name__
		if is_FC(lname) or is_C2D(lname):
			fn_mdl.layers[local_idx_to_l].set_weights([delta, init_biases[idx_to_t_mdl_l]])
		elif is_LSTM(lname):
			if idx_to_w == 0: # kernel
				new_kernel_w = delta # use the full 
				new_recurr_kernel_w = init_weights[(idx_to_t_mdl_l, 1)]
			elif idx_to_w == 1:
				new_recurr_kernel_w = delta
				new_kernel_w = init_weights[(idx_to_t_mdl_l, 0)]
			else:
				print ("{} not allowed".format(idx_to_w), idx_to_t_mdl_l, idx_to_tl)
				assert False
			# set kernel, recurr kernel, bias
			fn_mdl.layers[local_idx_to_l].set_weights(
				[new_kernel_w, new_recurr_kernel_w, init_biases[idx_to_t_mdl_l]])
		else:
			print ("{} not supported".format(lname))
			assert False

	predictions = None
	for chunk in chunks:
		_predictions = fn_mdl.predict(prev_outputs[chunk], batch_size = len(chunk))
		if predictions is None:
			predictions = _predictions
		else:
			predictions = np.append(predictions, _predictions, axis = 0)

	return predictions
