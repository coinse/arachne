"""
"""
import time

class Searcher(object):
	"""docstring for Searcher"""

	np = __import__('numpy')
	os = __import__('os')
	importlib = __import__('importlib')
	model_util = importlib.import_module('utils.model_util')
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
		indices_to_correct, indices_to_wrong,
		num_label,
		tensor_name_file,
		max_search_num = 200,
		initial_predictions = None,
		empty_graph = None,
		which = None,
		path_to_keras_model = None,
		is_empty_one = False,
		w_gather = False,
		at_indices = None,
		use_ewc = False):

		"""
		"""
		super(Searcher, self).__init__()

		# data related initialisation
		self.num_label = num_label
		self.inputs = inputs
		from collections.abc import Iterable
		if not isinstance(labels[0], Iterable):
			from utils.data_util import format_label
			self.ground_truth_labels = lables
			self.labels = format_label(labels, self.num_label)
		else:
			self.labels = labels
			self.ground_truth_labels = self.np.argmax(self.labels, axis = 1)

		self.model_name = "model"
		self.model_name_format = self.model_name + ".{}"

		self.w_gather = w_gather
		self.indices_to_correct = indices_to_correct
		self.indices_to_wrong = indices_to_wrong

		# model related initialisation
		self.path_to_keras_model = path_to_keras_model

		self.which = which
		self.tensors = self.set_target_tensor_names(tensor_name_file)

		self.at_indices = at_indices
		if empty_graph is None:
			self.generate_empty_graph()
		else:
			self.empty_graph = empty_graph
		
		self.sess = None
		# initialise the names of the tensors used in Searcher
		if not is_empty_one: 
			self.initialise_feed_dict() 
			self.set_initial_predictions(initial_predictions)
			self.maximum_fitness = 0.0# the maximum fitness value	
			# set search relate parameters
			self.max_search_num = max_search_num	
			self.indices_to_sampled_correct = None

		# EWC
		self.use_ewc = use_ewc
		#if self.use_ewc:
		#	self.ewc_inst = self.set_ewc_inst(
		#		init_weight,
		#		var_lambda = None)
		#else:
		self.ewc_inst = None


	@classmethod
	def get_which_tensors(cls):
		"""
		Return a list of tensors to look at
		"""
		return cls.which_tensors

	def generate_empty_graph(self):
		"""
		Generate an empyt graph frame for current searcher
		"""
		from gen_frame_graph import generate_empty_graph
		self.empty_graph = generate_empty_graph(self.which, 
			self.inputs, 
			self.num_label, 
			path_to_keras_model = self.path_to_keras_model, 
			w_gather = self.w_gather,
			indices = self.at_indices)
		self.inputs = None # no longer needed	

	def set_indices_to_wrong(self, indices_to_wrong):
		"""
		set new self.indices_to_wrong
		"""
		self.indices_to_wrong = indices_to_wrong

	def set_indices_to_correct(self, indices_to_correct):
		"""
		set new self.indices_to_correct
		"""
		self.indices_to_correct = indices_to_correct


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
		
			assertion_violate_msg = "%s is not one of %s" % (which_tensor, ",".join(Searcher.which_tensors))
			assert which_tensor in Searcher.which_tensors, assertion_violate_msg

			tensors[which_tensor] = terms[1] if len(terms) == 2 else terms[1:]

		return tensors


	def gen_feed_dict(self, values_dict):
		"""
		Arguments:
			values_dict: key = name of a placeholder, value = initial value
		Ret (dict):
			returm an initialised feed_dict
		"""
		assertion_violate_msg = "%s vs %s" % (','.join(list(values_dict.keys())),\
			','.join(Searcher.var_placeholders[self.which]))
		
		feed_dict = {}

		for placeholder_name in values_dict.keys():
			placeholder_tensor = self.model_util.get_tensor(
				placeholder_name, 
				self.empty_graph, 
				sess = None)

			feed_dict[placeholder_tensor] = values_dict[placeholder_name]

		return feed_dict


	def initialise_feed_dict(self):
		"""
		Initialise value_dict using self.meta_file and self.model_name
		and get the matching variable values with the placeholders
		"""
		import time
		values_dict = {}
		if self.which != 'lfw_vgg':
			kernel_and_bias_pairs = self.apricot_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)#6)#4)
		else:
			kernel_and_bias_pairs = self.torch_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)

		# fb1,fw1, fb2,fw2, fb3,fw3
		weights = {}
		weights["fw3"] = kernel_and_bias_pairs[-1][0]
		weights["fb3"] = kernel_and_bias_pairs[-1][1]
		values_dict = weights

		feed_dict = self.gen_feed_dict(values_dict)
	
		self.init_feed_dict = feed_dict.copy()
		self.curr_feed_dict = feed_dict.copy()


	def compute_initial_predictions(self, inputs = None, labels = None, num_label = None):
		"""
		"""
		if inputs is None:
			inputs = self.inputs
		if labels is None:
			labels = self.labels
		if num_label is None:
			num_label = self.num_label

		sess, (initial_predictions, _) = self.model_util.predict(
			inputs, labels, num_label,
			predict_tensor_name = self.tensors['t_prediction'], 
			corr_predict_tensor_name = None,
			indices_to_slice_tensor_name = 'indices_to_slice' if self.w_gather else None,
			sess = None, 
			empty_graph = self.empty_graph,
			plchldr_feed_dict = self.curr_feed_dict.copy(),
			use_pretr_front = self.path_to_keras_model is not None)
		# here, close the session
		sess.close()

		return initial_predictions


	def set_initial_predictions(self, initial_predictions = None):
		"""
		Set initial prediction results of the model that Searcher tries to optimise
		"""
		if initial_predictions is None:
			is_data_all_prepared = self.inputs is not None and self.labels is not None and self.num_label is not None,
			is_model_available = self.meta_file is not None and self.model_name is not None
			is_tensor_name_set = self.tensors['t_prediction'] is not None	

			assertion_violate_msg = "None check: inputs:%s, labels:%s, num_label:%s, \
				meta_file: %s, model_name:%s, t_prediction:%s" % (str(self.inputs is not None),
					str(self.labels is not None), str(self.num_label is not None),
					str(self.meta_file is not None), str(self.model_name is not None),
					str(self.tensors['t_prediction'] is not None))	

			assert is_data_all_prepared and is_tensor_name_set, assertion_violate_msg	
			# does not have to be on-the-fly one as it is only initialsation

			initial_predictions = self.compute_initial_predictions(
				inputs = self.inputs, 
				labels = self.labels, 
				num_label = self.num_label)

			self.initial_predictions = initial_predictions
		else:
			self.initial_predictions = initial_predictions


	def set_ewc_var_lambda(self, var_lambda):
		"""
		"""
		assert self.use_ewc is not None and self.ewc_inst is not None
		self.ewc_inst.var_lambda = var_lambda


	def set_ewc_inst(self,
		init_weight,
		var_lambda = None):
		"""
		generate EWC handler
		"""
		assert self.use_ewc

		from search import ewc

		if var_lambda is None:
			var_lambda = len(self.indices_to_correct)/len(self.indices_to_wrong)

		self.ewc_inst = ewc.EWC_Loss(
			init_weight, 
			self.initial_predictions, 
			self.labels,
			self.indices_to_correct, 
			self.indices_to_wrong, 
			self.empty_graph,
			self.curr_feed_dict,
			self.sess,
			weight_tensor_name = self.tensors['t_weight'],
			var_lambda = var_lambda,
			mode = 1)


	def move(self, target_tensor_name, delta, new_model_name, 
		update_op = 'add', values_dict = None):
		"""
		"""
		t0 = time.time()
		inputs = self.inputs
		labels = self.labels
		
		# set new feed_dict for the variable placeholders
		if values_dict is not None:
			feed_dict = self.gen_feed_dict(values_dict)
		else:
			# if values_dict is not given, use self.curr_feed_dict as an initial feed_dict
			feed_dict = self.curr_feed_dict.copy()
		assert feed_dict is not None, feed_dict
		#
		# update target tensor: get the target tensor & update its value with delta
		target_tensor = self.model_util.get_tensor(target_tensor_name, self.empty_graph)
		assert target_tensor in feed_dict.keys(), "Target tensor %s should already exists" % (target_tensor_name)
		#
		if update_op == 'add':
			feed_dict[target_tensor] += delta
		elif update_op == 'sub':
			feed_dict[target_tensor] -= delta
		else:# set
			feed_dict[target_tensor] = delta
		#	
		sess, (predictions, correct_predictions, all_losses) = self.model_util.predict(
			inputs, labels, self.num_label,
			predict_tensor_name = self.tensors['t_prediction'], 
			corr_predict_tensor_name = self.tensors['t_correct_prediction'],
			indices_to_slice_tensor_name = 'indices_to_slice' if self.w_gather else None,
			sess = self.sess, 
			empty_graph =  self.empty_graph,
			plchldr_feed_dict = feed_dict,
			use_pretr_front = self.path_to_keras_model is not None,
			compute_loss = True)
		
		if not self.use_ewc:
			losses_of_correct = all_losses[self.indices_to_correct]
			##
			indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.)[0]
			num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
			new_losses_of_correct = num_corr_true + self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1))
			##	

			losses_of_wrong = all_losses[self.indices_to_wrong]
			##
			indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.)[0]
			num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
			new_losses_of_wrong = num_wrong_true + self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1))
			##
			combined_losses	= (new_losses_of_correct, new_losses_of_wrong)	

			return sess, (predictions, correct_predictions, combined_losses)
		else:### under construction
			assert self.ewc_inst is not None, "should be called in de.eval"

			ewc_loss_v = self.ewc_inst.ewc_loss(delta, all_losses)
			return sess, (predictions, correct_predictions, ewc_loss_v)


	def get_results_of_target(self, indices_to_target, empty_graph, plchldr_feed_dict):
		"""
		Return the results of the target (can be accessed by indices_to_target)
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		sess, (_, correct_predictions) = self.model_util.predict(
			self.inputs, self.labels, self.num_label,
			predict_tensor_name = None,
			corr_predict_tensor_name = self.tensors['t_correct_prediction'],
			indices_to_slice_tensor_name = 'indices_to_slice' if self.w_gather else None,
			sess = self.sess, 
			empty_graph =  empty_graph,
			plchldr_feed_dict = plchldr_feed_dict,
			use_pretr_front = self.path_to_keras_model is not None,
			compute_loss = False)
		
		target_corr_predcs = correct_predictions[indices_to_target]

		num_of_total_target = len(target_corr_predcs)
		assert num_of_total_target == len(indices_to_target), "%d vs %d" % (num_of_total_target, len(indices_to_target))
		correctly_classified = self.np.sum(target_corr_predcs)

		return (correctly_classified, correctly_classified / num_of_total_target)


	def get_number_of_patched(self, 
		empty_graph = None, plchldr_feed_dict = None):
		"""
		Return a number of patched initially wrongly classified outputs 
		(can be accessed by self.indices_to_wrong)
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		if empty_graph is None:
			empty_graph = self.empty_graph

		if plchldr_feed_dict is None:
			plchldr_feed_dict = self.curr_feed_dict

		correctly_classified, perc_correctly_classifed = self.get_results_of_target(
			self.indices_to_wrong, 
			#sess, 
			empty_graph, 
			plchldr_feed_dict)


		return (correctly_classified, perc_correctly_classifed)


	def get_number_of_violated(self, #sess = None, 
		empty_graph = None, plchldr_feed_dict = None):
		"""
		Return a number of patched initially correctly classified outputs 
		(can be accessed by self.indices_to_correct)
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		if empty_graph is None:
			empty_graph = self.empty_graph

		if plchldr_feed_dict is None:
			plchldr_feed_dict = self.curr_feed_dict

		target_indices = self.indices_to_correct

		correctly_classified, perc_correctly_classifed = self.get_results_of_target(
			target_indices, 
			#sess, 
			empty_graph, 
			plchldr_feed_dict)

		num_of_initially_correct = len(target_indices)

		return (num_of_initially_correct - correctly_classified, 1.0 - perc_correctly_classifed)


	def get_num_patched_and_broken(self, predictions):
		"""
		compute fitness using a given predictions and correctly
		Ret (boolean, float):
			boolean: True if restriction is not violated, otherwise, False
			float: new fitness value
		"""
		# get the final clasification results of the newly computed predictions
		new_classifcation_results = self.np.argmax(predictions, axis = 1)	
		num_violated = self.np.sum((new_classifcation_results != self.ground_truth_labels)[self.indices_to_correct])
		num_patched = self.np.sum((new_classifcation_results == self.ground_truth_labels)[self.indices_to_wrong])

		return num_violated, num_patched	


	def check_early_stop(self, 
		new_weight_value, 
		fitness_value, 
		model_name = None, sess = None, empty_graph = None):
		"""
		Check whether early stop is possible or not
		Arguments:
			model_name: the name of model to examine
		Ret (bool):
			True (early stop)
			False (not yet)
		"""
		if model_name is None:
			model_name = self.model_name

		if empty_graph is None:
			empty_graph = self.empty_graph

		if empty_graph is not None:
			plchldr_feed_dict = self.curr_feed_dict.copy()
			target_tensor = self.model_util.get_tensor(self.tensors['t_weight'], empty_graph)

			assert target_tensor in self.curr_feed_dict.keys(), "Target tensor %s should already exists" % (target_tensor_name)

			plchldr_feed_dict[target_tensor] = new_weight_value
		else:
			plchldr_feed_dict = None

		num_of_patched, perc_num_of_patched = self.get_number_of_patched(
			empty_graph = empty_graph, 
			plchldr_feed_dict = plchldr_feed_dict)

		num_of_violated, perc_num_of_violated = self.get_number_of_violated(
			empty_graph = empty_graph, 
			plchldr_feed_dict = plchldr_feed_dict)

		t3 = time.time()

		if num_of_patched == len(self.indices_to_wrong) and num_of_violated == 0:
			print("In early stop checking:%d, %d" % (num_of_patched, num_of_violated))
			print ("\t fitness values", fitness_value)
			print ("\t", num_of_patched == len(self.indices_to_wrong) and num_of_violated == 0)

			return True, num_of_patched
		else:
			return False, num_of_patched


	def summarise_results(self, model_name, sess = None):
		"""
		Print out the current result of model_name
		"""
		print ("***Patching results***\n")

		num_of_patched, perc_num_of_patched = self.get_number_of_patched(
			empty_graph = self.empty_graph)

		num_of_violated, perc_num_of_violated = self.get_number_of_violated(
			empty_graph = self.empty_graph)

		print ("\tFor initially wrongly classified:%d -> %d(%f)" % (len(self.indices_to_wrong), \
			len(self.indices_to_wrong) - num_of_patched, perc_num_of_patched))
		print ("\tFor initially correctly classified(violation):%d -> %d(%f)" % (len(self.indices_to_correct),\
			len(self.indices_to_correct)- num_of_violated, perc_num_of_violated))


