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
		self.prev_vector_value, sess  = self.model_util.get_output_vector(
			self.num_label,
			self.tensors['t_prev_v'],
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

		nodes_with_grads = list(d_gradients.keys()) # key = indices to target nodes, value = gradient
		t4_1 = time.time()

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





