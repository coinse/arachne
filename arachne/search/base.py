"""
"""
class Base_Searcher(object):
	"""docstring for Base_Searcher"""

	np = __import__('numpy')
	os = __import__('os')
	importlib = __import__('importlib')
	model_util = importlib.import_module('utils.model_util')
	data_util = importlib.import_module('utils.data_util')
	apricot_rel_util = importlib.import_module('utils.apricot_rel_util')
	torch_rel_util = importlib.import_module('utils.torch_rel_util')

	which_tensors = [
		't_prev_v', 't_weight', 't_bias', 
		't_prediction','t_correct_prediction', 
		't_lab_loss']
		
	var_placeholders = {'cnn1':['fw3', 'fb3'],
					'cnn2':['fw3', 'fb3'],
					'cnn3':['fw3', 'fb3'],
					'simple_fm':['fw3', 'fb3'],
					'simple_cm':['fw3', 'fb3'],
					'lfw_vgg':['fw3', 'fb3']}

	def __init__(self, 
		inputs, labels,
		num_label,
		tensor_name_file,
		empty_graph = None,
		which = None):

		"""
		"""
		super(Base_Searcher, self).__init__()

		# data related initialisation
		self.inputs = inputs
		self.labels = labels
		self.num_label = num_label
		self.which = which
		self.empty_graph = empty_graph
		print ("Set empty graph", self.empty_graph, empty_graph)

		# initialise the names of the tensors used in Base_Searcher
		self.tensor_name_file = tensor_name_file
		self.tensors = self.set_target_tensor_names(self.tensor_name_file)

		if self.empty_graph is not None:
			self.initialise_feed_dict()
		else:
			#self.sess = None
			self.init_feed_dict = None 
			self.curr_feed_dict = None 
		

	@classmethod
	def get_which_tensors(cls):
		"""
		Return a list of tensors to look at
		"""
		return cls.which_tensors


	def set_target_tensor_names(self, tensor_name_file):
		"""
		Read a given tensor_name_file and set target tensors
		"""
		assert self.os.path.exists(tensor_name_file), "%s does not exist" % (tensor_name_file)

		with open(tensor_name_file) as f:
			lines = [line.strip() for line in f.readlines()]

		tensors = {}
		for line in lines:
			terms = line.split(",")
			assert len(terms) >= 2, "%s should contain at least two terms" % (line)

			which_tensor = terms[0]
		
			assertion_violate_msg = "%s is not one of %s" % (which_tensor, ",".join(Base_Searcher.which_tensors))
			assert which_tensor in Base_Searcher.which_tensors, assertion_violate_msg

			tensors[which_tensor] = terms[1] if len(terms) == 2 else terms[1:]

		return tensors


	def gen_feed_dict(self, values_dict):
		"""
		Arguments:
			values_dict: key = name of a placeholder, value = initial value
		Ret (dict):
			returm an initialised feed_dict
		"""
		assertion_violate_msg = "%s vs %s" % (','.join(list(values_dict.keys())), ','.join(Base_Searcher.var_placeholders[self.which]))
		assert sorted(list(values_dict.keys())) == sorted(Base_Searcher.var_placeholders[self.which]), assertion_violate_msg

		feed_dict = {}
		for placeholder_name in Base_Searcher.var_placeholders[self.which]:
			placeholder_tensor = self.model_util.get_tensor(placeholder_name, self.empty_graph)
			feed_dict[placeholder_tensor] = values_dict[placeholder_name]

		return feed_dict


	def initialise_feed_dict(self, path_to_keras_model = None):
		"""
		Initialise value_dict using self.meta_file and self.model_name
		and get the matching variable values with the placeholders
		"""
		values_dict = {}
		if self.which != 'lfw_vgg':
			kernel_and_bias_pairs = self.apricot_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)#6)#4)
		else:
			kernel_and_bias_pairs = self.torch_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)
		# fb1,fw1, fb2,fw2, fb3,fw3
		weights = {}
		#for i, (w,b) in enumerate(kernel_and_bias_pairs):
		#	weights["fw{}".format(i+1)] = w
		#	weights["fb{}".format(i+1)] = b	
		weights["fw3"] = kernel_and_bias_pairs[-1][0] # only the last
		weights["fb3"] = kernel_and_bias_pairs[-1][1]
		values_dict = weights

		feed_dict = self.gen_feed_dict(values_dict)

		self.init_feed_dict = feed_dict.copy()
		self.curr_feed_dict = feed_dict.copy()



