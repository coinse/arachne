"""
Localiser is used to identify the likely to be the most influential parts, a set of nodes,
of a given weight(target) layer.
"""

from search.base import Base_Searcher
import time
from search import other_localisers

class Localiser(Base_Searcher):
	"""docstring for Localiser"""
	random = __import__('random')

	def __init__(self, 
		inputs, labels, 
		num_label,
		predictions, # newly added for FI for any layer
		indices_to_target_layer, # newly added for F1 for any layer -> a list of target layer
		tensor_name_file,
		empty_graph = None, 
		which = None, 
		init_weight_value = None,
		nodes_to_lookat = None,
		path_to_keras_model = None,
		base_indices_to_cifar10 = None):
		"""
		Arguments:
		"""
		super(Localiser, self).__init__(
			inputs, labels,
			num_label,
			tensor_name_file,
			empty_graph = None,
			which = which)

		self.path_to_keras_model = path_to_keras_model
		# generate and set empty_graph that is main frame of the our model to patch
		if self.empty_graph is None:
			self.generate_empty_graph()
			self.initialise_feed_dict()

		self.init_weight_value = init_weight_value

		self.nodes_to_lookat = nodes_to_lookat
		self.base_indices_to_cifar10 = base_indices_to_cifar10
		self.d_output_weight = None

		self.prev_vector_value = None
		self.init_weight_and_prev_input()
		##
		self.predictions = predictions
		self.idx_to_target_layer = idx_to_target_layer


	def generate_empty_graph(self):
		"""
		"""
		from gen_frame_graph import generate_empty_graph
		self.empty_graph = generate_empty_graph(self.which, 
			self.inputs, self.num_label,  
			path_to_keras_model = self.path_to_keras_model, 
			w_gather = True)

	#def 

	def compute_prev_vector(self, feed_dict):
		"""
		"""
		self.prev_vector_value, sess  = self.model_util.get_output_vector(
			self.num_label,
			self.tensors['t_prev_v'], # 
			self.inputs, 
			indices_to_slice_tensor_name = 'indices_to_slice',
			sess = None,
			empty_graph = self.empty_graph,
			plchldr_feed_dict = feed_dict,
			use_pretr_front = self.path_to_keras_model is not None,
			base_indices_to_cifar10 = self.base_indices_to_cifar10)
		sess.close()

		return self.prev_vector_value, sess


	def compute_forward_impact(self, pos, weight_value = None):
		"""
		idx = the position in weight value to fix
		sens = Ik x Wj,k x Delta(change)
			-> since Delta should be the similart to all,
			we simplify the sensibility as Ik x Wj,k (this is only valid, because
			it is on last layer)
		"""
		assert self.prev_vector_value is not None

		if weight_value is None:
			weight_value_to_use = self.init_weight_value
		else:
			weight_value_to_use = weight_value

		avg_sens = self.np.mean(self.np.abs(weight_value_to_use[pos] * self.prev_vector_value[:,pos[0]]))	

		return avg_sens

	def compute_forward_impact_on_any_layer(self, pos_arr, weight_value = None):
		"""
		can be applied to any FNN layer and also, attention NNs. 
		But, on CNN ..? (think again)
		weight_value = an array of neural weights of a target layer
		
		pos = [for t, for t+1] (0th -> for t layer (prev), 1th -> for t+1 layer)

		(our target: w_(i,j) between L_t, and L_(t+1), t = t'th target layer)
		FI_(i,j) = w_(i,j) * v_i * grad(n_j, o_k), k = an index to the prediction

		Here, we will compute th
		"""
		from sklearn.preprocessing import Normalizer
		norm_scaler = Normalizer(norm = 'l1')
		if weight_value is None:
			weight_value_to_use = self.init_weight_value
		else:
 			weight_value_to_use = weight_value

		# from front part		
		#from_front =  self.prev_vector_value[:,pos[:,0]] * weight_value_to_use[pos[:,0],pos[:,1]]
		from_front_arr = []
		for pos in pos_arr:
			from_front_raw =  self.np.matmul(self.prev_vector_value, weight_value_to_use[:,pos[1]])
			from_front_abs = self.np.abs(from_front_raw)
			sum_front = self.np.sum(from_front_abs)
			from_front = from_front_abs/sum_front
			from_front_arr.append(from_front)

		from_front_arr = self.np.asarray(from_front_arr)
		print ("before mean:", from_front_arr.shape)
		from_front_arr = self.np.mean(from_front_arr, axis = 0)
		# print ("Normed front", from_front_arr.shape)

		# from behind part
		# get the gradient that has been computed before
		# if all layers .. then e.g., output_{idx} for idx in range(num_layer)
		# compute here, or add gradient computation tensor to each layer, and use get_output_vectore to just retrieve the tensor value
		#output_tensor_name = "output_{}".format(...) # some output_{idx_to_target + 1}
		output_tensor_name = "Gathered" # some output_{idx_to_target + 1}
		# get the index of the final output
		pos_of_pred_labels = self.np.asarray(
			list(zip(self.np.arange(self.predictions.shape[0]), self.np.argmax(self.predictions, axis = 1))))
		
		predc_tensor = self.empty_graph.get_tensor_by_name('predc:0')
		output_tensor = self.empty_graph.get_tensor_by_name('{}:0'.format(output_tensor_name))

		# d(pred)/d(output)
		tensor_grad = tf.gradients(
			predc_tensor[pos_of_pred_labels[:,0],pos_of_pred_labels[:,1]], 
			output_tensor,
			name = 'output_grad')

		(gradient,), sess  = self.model_util.run(
			tensor_grad,
			self.inputs, 
			self.labels, 
			empty_graph = self.empty_graph,
			plchldr_feed_dict = self.curr_feed_dict)

		sess.close()
		# take (absolute & norm) & average => to maintain the ratio of the impact 
		gradient = self.np.abs(gradient)
		norm_gradient = Normalizer(norm = 'l1').fit_transform(gradient)
		gradient_value_from_behind = self.np.mean(norm_gradient, axis = 0)
		from_behind_arr = gradient_value_from_behind # pos... what if pos is 3-d 
		
		print ("From behind", from_behind_arr.shape)
		#
		FI_arr = from_front_arr[:,pos[:,0]] * from_behind_arr[:,pos[:,1]]
		print ("FI", FI_arr.shape)
		return FI_arr
		

	## newly added to compute the forward impact of a neural weight on any layer
	def compute_forward_impact_on_random_neuron(self, 
		pos,
		predicted,  
		indices_to_wrong,
		weight_value = None, 
		chunk_size = 0):
		"""
		Compute a forward impact of a random neural weight,
		or an array of foward impact on the random weight neuron specified by
		the argument pos (position).

		=> although, the ideal one will target all, here, only a specific weight value 
			-> the one self.weight_value . self.idx_to_target_layer
		Args:
			pos: a position to a neural weight (list)
				or an array of positions to specific nerual weights
			predicted: idx to target label or indices to target labels
		"""
		#assert predicted is not None and len(predicted)/self.num_label == len(pos), "{} & Length: pos({}) vs predicted({})".format(
		#	predicted,
		#	len(pos),
		#	len(predicted) if predicted is not None else -1)
		#print ("Predictions", predicted)
		#print (predicted.shape)
		#print (pos)
		
		if weight_value is None:
			weight_value = self.init_weight_value

		if self.d_output_weight is None:
			(self.d_output_weight,), sess = self.model_util.compute_gradient(
				self.num_label,
				"doutput_dw",
				self.inputs, self.labels, keep_prob_val = 1.0,
				input_tensor_name = "inputs", 
				output_tensor_name = "labels", 
				keep_prob_name = "keep_prob",
				indices_to_slice_tensor_name = "indices_to_slice",
				sess = None,
				empty_graph = self.empty_graph,
				plchldr_feed_dict = self.curr_feed_dict,
				is_cifar10 = self.path_to_keras_model is not None,
				base_indices_to_cifar10 = self.np.arange(len(indices_to_wrong)),
				chunk_size = chunk_size)
			sess.close()

		# first compute grandient descent of a target nerual weight relative to the output (o_l = l = a classified label
		predicted_labels = self.np.argmax(predicted, axis = 1)
		print ("Predicted label", predicted_labels)
		predicted_to_pair_with_pos = self.np.full(pos.shape[0], predicted_labels)#[0])
		#print ("Predicted label extended to pair with pos", predicted_to_pair_with_pos.shape)
		#predicted_to_pair_with_pos = np.zeros((predicted_labels.shape[0], pos.shape[0]))
		#for i in range(predicted_to_pair_with_pos.shape[0]):
		#	predicted_to_pair_with_pos[i] = predicted_labels
		#predicted_to_pair_with_pos = predicted_to_pair_with_pos.reshape(-1,)

		#print (weight_value.shape)
		#print (self.prev_vector_value.shape)
		values_of_target_weight = weight_value[pos[:,0], pos[:,1]]
		#values_of_target_act = self.prev_vector_value[pos[:,0], pos[:,1]]
			
		#print (values_of_target_weight)
		#print (values_of_target_act)
		#print ("Two value", values_of_target_weight.shape, self.d_output_weight.shape, predicted_to_pair_with_pos.shape, pos.shape)
		#print ("Two value", values_of_target_act.shape, self.d_output_weight.shape, predicted_to_pair_with_pos.shape, pos, pos.shape)

		gds_of_target_weight_to_output = self.d_output_weight[predicted_to_pair_with_pos, pos[:,0], pos[:,1]]
		#print (gds_of_target_weight_to_output)
		#print (gds_of_target_weight_to_output.shape)
		#print ("===============================================================")
		#print (values_of_target_weight)
		#print (values_of_target_weight.shape)
		return self.np.abs(gds_of_target_weight_to_output * values_of_target_weight)


	def init_weight_and_prev_input(self):
		"""
		Initialise the collection of activation traces, which is stored in self.ats_collection
		self.ats_collection(dict):
			key = an index to the target node that has been initialised with a random value 
				to compute the change of activation traces caused by the target node
			value = a list of ats (nested list) or can be np.drarray
		"""
		assert self.path_to_keras_model is not None

		if self.init_weight_value is None:
			if self.which != 'lfw_vgg':
				kernel_and_bias_pairs = self.apricot_rel_util.get_weights(self.path_to_keras_model,start_idx = 0)
			else:
				kernel_and_bias_pairs = self.torch_rel_util.get_weights(self.path_to_keras_model,start_idx = 0)
			weights = {}
			#for i, (w,b) in enumerate(kernel_and_bias_pairs):
			#	weights["fw{}".format(i+1)] = w
			#	weights["fb{}".format(i+1)] = b	
			weights["fw3"] = kernel_and_bias_pairs[-1][0] # only the last
			weights["fb3"] = kernel_and_bias_pairs[-1][1]
			self.init_weight_value = weights[self.tensors['t_weight']]

		if self.nodes_to_lookat is None:
			# we will look all of the possible node in the target weight
			self.nodes_to_lookat = [(i,j) for i in range(self.init_weight_value.shape[0]) for j in range(self.init_weight_value.shape[1])]

		import time
		# compute input vector to the target weight node
		self.compute_prev_vector(self.curr_feed_dict)	


	def run(self, 
		idx_to_wrong, ####
		#num_of_top_nodes = 0, 
		predicted = None,
		indices_to_wrong = None,
		from_where_to_fix_nw_down = None,
		pareto_ret_all = False):
		"""
		d_gradients(dict):
			key = index to a single weight node
			value = gradient loss value of the weight node
		"""
		nodes_to_lookat, indices_to_cands_and_grads = from_where_to_fix_nw_down
		curr_nodes_to_lookat = nodes_to_lookat
		d_gradients = {k:v for k, v in indices_to_cands_and_grads} 

		assert curr_nodes_to_lookat is not None
		# compute forward impact for the given nodes, i.e., nodes in curr_nodes_to_lookat 
		forward_impact = {}
		for index_to_node in curr_nodes_to_lookat:
			a_forward_impact = self.compute_forward_impact(index_to_node)
			forward_impact[index_to_node] = a_forward_impact
		
		### compute forwrad impact all
		forward_impacts = self.compute_forward_impact_on_any_layer(np.asarray(curr_nodes_to_lookat))
		forward_impacts = {curr_nodes_to_lookat[i]:forward_impacts[i] for i in range(len(curr_nodes_to_lookat))}
		### compute forwrad impact all end
		nodes_with_grads = list(d_gradients.keys()) # key = indices to target nodes, value = gradient
		t4_1 = time.time()
		print (nodes_with_grads)
		print (forward_impacts)
		import sys; sys.exit()
		costs = self.np.asarray([[d_gradients[node], forward_impact[node]] for node in curr_nodes_to_lookat])

		ret_lst = []
		ret_front_lst = []
		while len(curr_nodes_to_lookat) > 0:
			_costs = costs.copy()
			is_efficient = self.np.arange(costs.shape[0])
			next_point_index = 0 # Next index in the is_efficient array to search for

			while next_point_index < len(_costs):
				nondominated_point_mask = self.np.any(_costs > _costs[next_point_index], axis=1)
				nondominated_point_mask[next_point_index] = True
				is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
				_costs = _costs[nondominated_point_mask]
				next_point_index = self.np.sum(nondominated_point_mask[:next_point_index])+1
						
			current_ret = [tuple(v) for v in self.np.asarray(curr_nodes_to_lookat)[is_efficient]]
			if not pareto_ret_all or len(ret_lst) == 0: # here, we intened to return only the front if pareto_ret_all is True
				ret_lst.extend(current_ret)

			ret_front_lst.append(current_ret)
			# remove selected items (non-dominated ones)
			curr_nodes_to_lookat = self.np.delete(curr_nodes_to_lookat, is_efficient, 0)
			costs = self.np.delete(costs, is_efficient, 0)
			if not pareto_ret_all: # go on or break
				break
				
		return ret_lst, ret_front_lst





