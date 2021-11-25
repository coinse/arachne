"""
Gradient loss localiser & random selection localiser
"""
import sys
sys.path.insert(0, "../utils")
import utils.model_util as model_util
import time

def where_to_fix_from_gl(
	indices_to_wrong, 
	loss_tensor_name,
	weight_tensor_name,
	num_label,
	input_data, output_data,
	empty_graph = None,
	plchldr_feed_dict = None,
	path_to_keras_model = None,
	top_n = 1):
	"""
	"""
	import numpy as np

	target_output_data = output_data[indices_to_wrong]
	target_input_data = input_data[indices_to_wrong]

	(gradient_value,), sess = model_util.compute_gradient_new(
				num_label,
				loss_tensor_name,
				weight_tensor_name,
				target_input_data, target_output_data,
				input_tensor_name = "inputs", 
				output_tensor_name = "labels", 
				indices_to_slice_tensor_name = "indices_to_slice",
				sess = None,
				empty_graph = empty_graph,
				plchldr_feed_dict = plchldr_feed_dict,
				base_indices_to_cifar10 = indices_to_wrong, 
				use_pretr_front = path_to_keras_model is not None)
	sess.close()

	if top_n == 1: # multiple places can be chosen
		indices_to_places_to_fix_and_grads = []
		max_value = np.max(np.abs(gradient_value))
		row_indices, column_indices = np.where(np.abs(gradient_value) == max_value)	
		for row_idx, column_idx in zip(row_indices, column_indices):
			idx_to_largest_grad = (row_idx, column_idx)
			indices_to_places_to_fix_and_grads.append([idx_to_largest_grad, max_value])
	else:
		from scipy.stats import rankdata	

		if top_n < 0 or top_n > gradient_value.shape[0] * gradient_value.shape[1]: 
			#top_n = gradient_value.shape[0] * gradient_value.shape[1] # this means, taking all of them as the target
			top_n = gradient_value.reshape(-1,).shape[0] # the number of total neural weights in the target weight var

		if top_n < gradient_value.reshape(-1,).shape[0]: #gradient_value.shape[0] * gradient_value.shape[1]:
			flatten_gradients = np.abs(gradient_value).reshape(-1,)
			ranks_of_gradients = list(rankdata(-flatten_gradients, method = 'max')) # in decending order
			top_ranks = np.sort(ranks_of_gradients)[:top_n]	

			indices_to_top_n_gradients = []
			for top_rank in set(top_ranks):#range(top_n):
				flatten_indices, = np.where(ranks_of_gradients == top_rank)
				matching_grad = flatten_gradients[flatten_indices[0]]	

				for flatten_index in flatten_indices:	
					idx_to_grad_row = int(np.floor(flatten_index / (gradient_value.shape[1])))
					idx_to_grad_column = int(flatten_index - idx_to_grad_row * (gradient_value.shape[1]))	
					idx_to_grad = (idx_to_grad_row, idx_to_grad_column)	

					indices_to_top_n_gradients.append([idx_to_grad, matching_grad])	

			indices_to_places_to_fix_and_grads = indices_to_top_n_gradients
		else: # meaning, it is likely they are too many and also, since we took them all, we do not need to sort any
			# take absolute values of them
			abs_gradient_value = np.abs(gradient_value)
			indices_to_grad = [(i,j) for i in range(abs_gradient_value.shape[0]) for j in range(abs_gradient_value.shape[1])]
			indices_to_places_to_fix_and_grads = list(zip(indices_to_grad, abs_gradient_value.reshape(-1,)))
				
	nodes_to_lookat = list(map(lambda v:v[0], indices_to_places_to_fix_and_grads))

	return nodes_to_lookat, indices_to_places_to_fix_and_grads


def where_to_fix_from_random(
	tensor_name_file,
	number_of_place_to_fix,
	empty_graph = None):
	"""
	randomly select places to fix
	"""
	import random
	import numpy as np
	from utils.data_util import read_tensor_name

	gradient_tensor_name = read_tensor_name(tensor_name_file)['t_weight']
	gradient_tensor = model_util.get_tensor(gradient_tensor_name, empty_graph)
	gradient_tensor_shape = tuple([int(v) for v in gradient_tensor.shape])

	all_indices = [(i,j) for i in range(gradient_tensor_shape[0]) for j in range(gradient_tensor_shape[1])]	
	if number_of_place_to_fix > 0 and number_of_place_to_fix < len(all_indices):
		selected_indices = np.random.choice(np.arange(gradient_tensor_shape[0]*gradient_tensor_shape[1]), 
			number_of_place_to_fix, replace = False)
		indices_to_places_to_fix = [all_indices[idx] for idx in selected_indices]
	else:
		indices_to_places_to_fix = all_indices
	
	return indices_to_places_to_fix


	
