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
		#indices_to_target_layer, # newly added for F1 for any layer -> a list of target layer
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
		#self.idx_to_target_layer = idx_to_target_layer


	def generate_empty_graph(self):
		"""
		"""
		from gen_frame_graph import generate_empty_graph
		self.empty_graph = generate_empty_graph(self.which, 
			self.inputs, self.num_label,  
			path_to_keras_model = self.path_to_keras_model, 
			w_gather = True)


	def compute_prev_vector(self, feed_dict):
		"""
		"""
		print ("Computeing prev")
		print ("\t", feed_dict.keys())
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
		
		#print ("original", (self.np.abs(weight_value_to_use[pos] * self.prev_vector_value[:,pos[0]]).shape))
		avg_sens = self.np.mean(self.np.abs(weight_value_to_use[pos] * self.prev_vector_value[:,pos[0]]))	

		return avg_sens

	def compute_forward_impact_on_any_layer(self, weight_value = None):
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
		import tensorflow as tf
		import time 

		if weight_value is None:
			weight_value_to_use = self.init_weight_value
		else:
 			weight_value_to_use = weight_value
		
		#print ("All targets", pos_arr.shape)
		print ("weight", weight_value_to_use.shape)
		print ("prev", self.prev_vector_value.shape)
		#print ("we", set(pos_arr[:,1]))
		#print ("\t", self.np.max(pos_arr[:,0]))
		# from front part		
		#from_front =  self.prev_vector_value[:,pos[:,0]] * weight_value_to_use[pos[:,0],pos[:,1]]
		from_front_arr = []
		i = 0
		#from tqdm import tqdm
		##t1 = time.time()
		#for pos in tqdm(pos_arr[:2]):
		#	from_front_raw = self.np.multiply(self.prev_vector_value, weight_value_to_use[:,pos[1]])
		#	if i == 0:
		#		print ("+", from_front_raw.shape, self.prev_vector_value.shape, weight_value_to_use[:,pos[1]].shape)
		#		#print (self.prev_vector_value * weight_value_to_use[:,pos[1]])
		#	from_front_abs = self.np.abs(from_front_raw)
		#	normed_front = norm_scaler.fit_transform(from_front_abs)
		#	#normed_front = from_front_abs	
		#	#print ("N", normed_front.shape)
		#	# retrive only our target
		#	from_front = normed_front[:,pos[0]]
		#	if i ==0:
		#		print ("-", from_front.shape)
		#		i+=1
		#	
		#	from_front_arr.append(self.np.mean(from_front))
		##t2 = time.time()
		##print ("Time", t2 - t1)
		##

		curr_plchldr_feed_dict = self.curr_feed_dict.copy()
		indices_to_slice_tensor = self.empty_graph.get_tensor_by_name('%s:0' % ("indices_to_slice"))
		curr_plchldr_feed_dict[indices_to_slice_tensor] = list(range(len(self.labels)))

		prev_tensor = self.empty_graph.get_tensor_by_name("{}:0".format(self.tensors['t_prev_v']))
		weight_tensor = self.empty_graph.get_tensor_by_name("{}:0".format(self.tensors['t_weight']))

		print (self.empty_graph.get_tensor_by_name("logits:0"))
		print (prev_tensor)
		print (weight_tensor)
		print ("F", curr_plchldr_feed_dict.keys())
		sess = None
		temp_tensors = []
		#print ("target", pos_arr[:20])
		#for pos in tqdm(pos_arr[:20]):
		#	output_tensor = tf.math.multiply(prev_tensor, weight_tensor[:,pos[-1]])# since we have to normalise, instead of pos[0], take all
		#	output_tensor = tf.math.abs(output_tensor)
		#	print ("-", output_tensor)
		#	sum_tensor = tf.math.reduce_sum(output_tensor, axis = -1) #
		#	print ("sum", sum_tensor)
		#	# norm
		#	output_tensor = tf.transpose(tf.div_no_nan(tf.transpose(output_tensor), sum_tensor))
		#	print ("--", output_tensor)
		##	output_tensor = output_tensor[:,pos[0]]
		##	#output_tensor = tf.math.divide(output_tensor, sum_tensor)[:,pos[0]]
		##	print ("+", output_tensor)
		##	output_tensor = tf.math.reduce_mean(output_tensor, axis = 0)
		##	#output_tensor = tf.linalg.norm(output_tensor, ord = 1, axis = 1)[0]
		##	#print ("++", output_tensor)
		##	temp_tensors.append(output_tensor)
		#	temp_tensors.append(sum_tensor)

		####
		for idx in range(weight_value_to_use.shape[-1]):
			output_tensor = tf.math.multiply(prev_tensor, weight_tensor[:,idx])
			output_tensor = tf.math.abs(output_tensor)
			print ("-", output_tensor)
			sum_tensor = tf.math.reduce_sum(output_tensor, axis = -1) #
			print ("sum", sum_tensor)
			# norm
			output_tensor = tf.transpose(tf.div_no_nan(tf.transpose(output_tensor), sum_tensor))
			print ("--", output_tensor)
			#output_tensor = output_tensor[:,pos[0]]
			#print ("+", output_tensor)
			output_tensor = tf.math.reduce_mean(output_tensor, axis = 0) # compute an average for given inputs
			#output_tensor = tf.linalg.norm(output_tensor, ord = 1, axis = 1)[0]
			print ("++", output_tensor)
			temp_tensors.append(output_tensor)
		####

		print ("start")
		print (temp_tensors[0])
		t1 = time.time()
		outs, sess  = self.model_util.run(
			temp_tensors,
			self.inputs, 
			self.labels, 
			input_tensor_name = None, output_tensor_name = None,
			empty_graph = self.empty_graph,
			plchldr_feed_dict = curr_plchldr_feed_dict)
		#sess.close()
		t2 = time.time()
		print ("Time", t2 - t1)
		print (len(outs))
		#print(len(outs[0]))
		#print(outs[0].shape)
		outs = self.np.asarray(outs).T
		print (outs.shape)
		#from_front = self.np.mean(outs, axis = 1) # compute an average for given inputs
		from_front = outs
		#from_front = self.np.asarray(from_front_arr)
		print ("Front", from_front.shape)
		#print ("\t", from_front[0])
		#print ("\t", from_front[1])
		#print ("\t", self.np.sum(from_front[:,0]))
		print ("\t", self.np.sum(from_front, axis = 0))
		#import sys; sys.exit()
		
		# from behind part
		# get the gradient that has been computed before
		# if all layers .. then e.g., output_{idx} for idx in range(num_layer)
		# compute here, or add gradient computation tensor to each layer, and use get_output_vectore to just retrieve the tensor value
		#output_tensor_name = "output_{}".format(...) # some output_{idx_to_target + 1}
		output_tensor_name = "predc" # some output_{idx_to_target + 1}
		# get the index of the final output
		pos_of_pred_labels = self.np.asarray(
			list(zip(self.np.arange(self.predictions.shape[0]), self.np.argmax(self.predictions, axis = 1))))
		
		predc_tensor = self.empty_graph.get_tensor_by_name('predc:0')
		output_tensor = self.empty_graph.get_tensor_by_name('{}:0'.format(output_tensor_name))

		# d(pred)/d(output)
		print ("Inputs", len(self.inputs))
		print ("target predc", pos_of_pred_labels.shape)
		print ("\tex", pos_of_pred_labels[:5])
		print (predc_tensor)
		print (output_tensor)
		print (pos_of_pred_labels[:,0].shape, pos_of_pred_labels[:,1].shape)
		print (pos_of_pred_labels[:,0])
		print (pos_of_pred_labels[:,1])
		#for pos in pos_arr:
		tensor_grad = tf.gradients(
			predc_tensor, #[:,pos_of_pred_labels[:,1]], 
			output_tensor,
			name = 'output_grad')

		(gradient,), sess  = self.model_util.run(
			tensor_grad,
			self.inputs, 
			self.labels,
			input_tensor_name = None, output_tensor_name = None,
			sess = sess, 
			empty_graph = self.empty_graph,
			plchldr_feed_dict = curr_plchldr_feed_dict)

		sess.close()
		# take (absolute & norm) & average => to maintain the ratio of the impact 
		print ("tensor grad", tensor_grad)
		print ("\t", gradient.shape)
		gradient = self.np.abs(gradient)
		norm_gradient = Normalizer(norm = 'l1').fit_transform(gradient)
		print ("after norm", norm_gradient.shape)
		gradient_value_from_behind = self.np.mean(norm_gradient, axis = 0) # compute the average for given inputs
		from_behind = gradient_value_from_behind # pos... what if pos is 3-d 
		
		print ("From behind", from_behind.shape)
		print ("\t", from_behind)
		print ("\t", self.np.sum(from_behind, axis = 0))
	
		#FIs = []
		#for pos in pos_arr:
		#	FIs.append(from_front[pos[0]] * from_behind[pos[1]])
		#	#FIs.append(from_front[pos[0]])
		##
		FIs = self.np.multiply(from_front, from_behind)
		print ("FI shape", FIs.shape)
		##
		print ("Max: {}, min:{}".format(self.np.max(FIs), self.np.min(FIs)))
		return FIs
		

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
		i = 0
		print (curr_nodes_to_lookat[:20])
		for index_to_node in curr_nodes_to_lookat:
			a_forward_impact = self.compute_forward_impact(index_to_node)
			if i < 20:
				print ("+", a_forward_impact)
				i+= 1
			forward_impact[index_to_node] = a_forward_impact
			
		### compute forwrad impact all
		print ("Length of nodes", len(list(forward_impact.keys())))
		print (len(curr_nodes_to_lookat))
		forward_impacts_2 = self.compute_forward_impact_on_any_layer().reshape(-1,) #self.np.asarray(curr_nodes_to_lookat))
		forward_impacts_2 = {curr_nodes_to_lookat[i]:forward_impacts_2[i] for i in range(len(curr_nodes_to_lookat))}
		
		print ("compare", forward_impact[curr_nodes_to_lookat[0]], forward_impacts_2[curr_nodes_to_lookat[0]])
		#import sys; sys.exit()
		### compute forwrad impact all end
		#nodes_with_grads = list(d_gradients.keys()) # key = indices to target nodes, value = gradient
		##
		forward_impact = forward_impacts_2
		##
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
		
		#print (ret_lst)	
		#import sys; sys.exit()	
		return ret_lst, ret_front_lst





