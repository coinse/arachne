"""
"""
import time


class Searcher(object):
	"""docstring for Searcher"""

	np = __import__('numpy')
	os = __import__('os')
	importlib = __import__('importlib')
	kfunc_util = importlib.import_module('utils.kfunc_util')
	model_util = importlib.import_module('utils.model_util')
	data_util = importlib.import_module('utils.data_util')
	gen_frame_graph = importlib.import_module('utils.gen_frame_graph')
	def __init__(self, 
		inputs, labels,
		indices_to_correct, indices_to_wrong,
		num_label,
		indices_to_target_layers, 
		max_search_num = 200,
		initial_predictions = None,
		path_to_keras_model = None,
		batch_size = None, 
		is_multi_label = True,
		is_lstm = False,
		act_func = None):

		"""
		"""
		super(Searcher, self).__init__()

		# data related initialisation
		self.num_label = num_label
		self.inputs = inputs
		from collections.abc import Iterable
		if is_multi_label:
			if not isinstance(labels[0], Iterable):
				from utils.data_util import format_label
				self.ground_truth_labels = labels
				self.labels = format_label(labels, self.num_label)
			else:	
				self.labels = labels
				self.ground_truth_labels = self.np.argmax(self.labels, axis = 1)
		else:
			self.labels = labels
			self.ground_truth_labels = labels
	
		self.lstm_mdl = is_lstm
		self.model_name = "model"
		self.model_name_format = self.model_name + ".{}"

		self.indices_to_correct = indices_to_correct
		self.indices_to_wrong = indices_to_wrong

		# model related initialisation
		self.path_to_keras_model = path_to_keras_model
		self.indices_to_target_layers = indices_to_target_layers 
		self.targeted_layer_names = None
		self.batch_size = batch_size
		self.act_func = act_func # will be latter used for GTSRB
		self.is_multi_label = is_multi_label

		if not self.lstm_mdl:
			self.set_base_model_v1()
		else:
			self.set_base_model_v2()
		self.set_target_weights()

		# initialise the names of the tensors used in Searcher
		if initial_predictions is None:
			self.set_initial_predictions()# initial_predictions)
		else:
			self.initial_predictions = initial_predictions
		self.maximum_fitness = 0.0 # the maximum fitness value	
		
		# set search relate parameters
		self.max_search_num = max_search_num	
		self.indices_to_sampled_correct = None


	def set_base_model_v1(self):
		"""
		Generate an empyt graph frame for current searcher
		"""
		from tensorflow.keras.models import load_model
		mdl = load_model(self.path_to_keras_model)
		self.mdl = mdl
		print ("Number of layers in model: {}".format(len(self.mdl.layers)))

		self.fn_mdl_lst = self.kfunc_util.generate_base_mdl(	
			self.mdl, self.inputs, indices_to_tls = self.indices_to_target_layers, 
			batch_size = self.batch_size, act_func = self.act_func)


	def set_base_model_v2(self):
		"""
		generate a list of Model instances -> used with move_lstm
		"""
		from tensorflow.keras.models import load_model
		from collections.abc import Iterable
		from tensorflow.keras.models import Model
		import tensorflow as tf

		mdl = load_model(self.path_to_keras_model, compile = False)
		self.mdl = mdl
		#print ("Number of layers in model: {}".format(len(self.mdl.layers)))
		
		# set targete layer names
		self.targeted_layer_names = {}
		for idx_to_tl in self.indices_to_target_layers:
			self.targeted_layer_names[idx_to_tl] = type(self.mdl.layers[idx_to_tl]).__name__

		# compute previous outputs
		self.min_idx_to_tl = self.np.min(
			[idx if not isinstance(idx, Iterable) else idx[0] 
				for idx in self.indices_to_target_layers])

		self.input_layer_added = False
		if self.min_idx_to_tl == 0:
			if not self.model_util.is_Input(type(self.mdl.layers[0]).__name__):
				self.input_layer_added = True
				
		prev_l = self.mdl.layers[self.min_idx_to_tl-1 if self.min_idx_to_tl > 0 else 0]
		if self.model_util.is_Input(type(prev_l).__name__): 
			# previous layer is an input layer
			self.prev_outputs = self.inputs
		else: # otherwise, compute the output of the previous layer
			t_mdl = Model(inputs = self.mdl.input, outputs = prev_l.output)
			print ("temporaray model")
			print (t_mdl.summary())		
			self.prev_outputs = t_mdl.predict(self.inputs)

		# set base model
		# a list that contains a single model
		self.fn_mdl_lst = [
			self.gen_frame_graph.build_mdl_lst(self.mdl, self.prev_outputs.shape[1:], 
			sorted(self.indices_to_target_layers))]

		# set chunks
		self.chunks = self.data_util.return_chunks(
			len(self.inputs), 
			batch_size = self.batch_size)

		# also set softmax and loss op
		self.k_fn_loss = None 

	def set_target_weights(self):
		"""
		"""
		self.init_weights = {}
		self.init_biases = {}
		for idx_to_tl in self.indices_to_target_layers:
			ws = self.mdl.layers[idx_to_tl].get_weights()
			lname = type(self.mdl.layers[idx_to_tl]).__name__
			if (self.model_util.is_FC(lname) or self.model_util.is_C2D(lname)):
				self.init_weights[idx_to_tl] = ws[0]
				self.init_biases[idx_to_tl] = ws[1]
			elif self.model_util.is_LSTM(lname):
				for i in range(2): # get only the kernel and recurrent kernel, not the bias
					self.init_weights[(idx_to_tl, i)] = ws[i]
				self.init_biases[idx_to_tl] = ws[-1]
			else:
				print ("Not supported layer: {}".format(lname))
				assert False
	
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

	def set_initial_predictions(self):
		"""
		"""
		if self.mdl is None:
			self.set_base_model()
		self.initial_predictions = self.mdl.predict(self.inputs)


	def get_results_of_target(self, indices_to_target):
		"""
		Return the results of the target (can be accessed by indices_to_target)
			-> results are compute for currnet self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		predictions = self.mdl.predict(self.inputs)
		correct_predictions = self.np.argmax(predictions, axis = 1)
		correct_predictions = correct_predictions == self.np.argmax(self.labels, axis = 1)

		target_corr_predcs = correct_predictions[indices_to_target]

		num_of_total_target = len(target_corr_predcs)
		msg = "%d vs %d" % (num_of_total_target, len(indices_to_target))
		assert num_of_total_target == len(indices_to_target), msg
		correctly_classified = self.np.sum(target_corr_predcs)

		return (correctly_classified, correctly_classified / num_of_total_target)


	def move(self, deltas, update_op = 'set'):
		"""
		*** should be checked and fixed
		"""
		import tensorflow as tf
	
		labels = self.labels
		deltas_as_lst = [deltas[idx_to_tl] 
			for idx_to_tl in self.indices_to_target_layers if idx_to_tl in deltas.keys()] 
		predictions, losses_of_all = self.kfunc_util.compute_preds_and_losses(
			self.fn_mdl_lst, labels, deltas_as_lst, batch_size = self.batch_size)

		if len(predictions.shape) > len(labels.shape) and predictions.shape[1] == 1:
			predictions = self.np.squeeze(predictions, axis = 1)
		
		if self.is_multi_label:
			correct_predictions = self.np.argmax(predictions, axis = 1)
			correct_predictions = correct_predictions == self.np.argmax(labels, axis = 1)
		else:
			correct_predictions = self.np.round(predictions).flatten() == labels

		losses_of_correct = losses_of_all[self.indices_to_correct]
		indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.)[0]
		num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
		new_losses_of_correct = num_corr_true + self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1))

		losses_of_wrong = losses_of_all[self.indices_to_wrong]
		indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.)[0]
		num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
		new_losses_of_wrong = num_wrong_true + self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1))

		combined_losses	= (new_losses_of_correct, new_losses_of_wrong)
		#print (self.is_multi_label, 
		# self.lstm_mdl, self.num_label, num_corr_true, 
		# self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1)), 
		# num_wrong_true, self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1)))
		return predictions, correct_predictions, combined_losses
		

	def predict_with_new_delta(self, deltas):
		"""
		predict with the model patched using deltas
		"""
		from collections.abc import Iterable
		# prepare a new model to run by updating the weights from deltas
		
		# we only have a one model as this one accept any lenghts of an input,
		# which is actually the output of the previous layers
		fn_mdl = self.fn_mdl_lst[0] 
		for idx_to_tl, delta in deltas.items(): # either idx_to_tl or (idx_to_tl, i)
			if isinstance(idx_to_tl, Iterable):
				idx_to_t_mdl_l, idx_to_w = idx_to_tl
			else:
				idx_to_t_mdl_l = idx_to_tl
			
			# index of idx_to_tl (from deltas) in the local model
			local_idx_to_l = idx_to_t_mdl_l - self.min_idx_to_tl + 1 
			lname = type(fn_mdl.layers[local_idx_to_l]).__name__
			if self.model_util.is_FC(lname) or self.model_util.is_C2D(lname):
				fn_mdl.layers[local_idx_to_l].set_weights([delta, self.init_biases[idx_to_t_mdl_l]])
			elif self.model_util.is_LSTM(lname):
				if idx_to_w == 0: # kernel
					new_kernel_w = delta # use the full 
					new_recurr_kernel_w = self.init_weights[(idx_to_t_mdl_l, 1)]
				elif idx_to_w == 1:
					new_recurr_kernel_w = delta
					new_kernel_w = self.init_weights[(idx_to_t_mdl_l, 0)]
				else:
					print ("{} not allowed".format(idx_to_w), idx_to_t_mdl_l, idx_to_tl)
					assert False
				# set kernel, recurr kernel, bias
				fn_mdl.layers[local_idx_to_l].set_weights(
					[new_kernel_w, new_recurr_kernel_w, self.init_biases[idx_to_t_mdl_l]])
			else:
				print ("{} not supported".format(lname))
				assert False

		predictions = None
		for chunk in self.chunks:
			_predictions = fn_mdl.predict(self.prev_outputs[chunk], batch_size = len(chunk))
			if predictions is None:
				predictions = _predictions
			else:
				predictions = self.np.append(predictions, _predictions, axis = 0)
		return predictions


	def move_lstm(self, deltas):
		"""
		*** should be checked and fixed
		--> need to fix this...
		delatas -> key: idx_to_tl & inner_key: index to the weight
				or key: (idx_to_tl, i) & inner_key
				value -> the new value
		"""
		#import time

		labels = self.labels
		predictions = self.predict_with_new_delta(deltas)
		#print (predictions)
		# due to the data dimention of fashion_mnist (..) 
		if predictions.shape != labels.shape:
			to_this_shape = labels.shape
			predictions = self.np.reshape(predictions, to_this_shape)

		#t1 = time.time()
		if self.is_multi_label:
			correct_predictions = self.np.argmax(predictions, axis = 1)
			correct_predictions = correct_predictions == self.np.argmax(labels, axis = 1)
		else:
			correct_predictions = self.np.round(predictions).flatten() == labels

		# set self.k_fn_loss if it has been done yet.
		if self.k_fn_loss is None:
			loss_func = self.model_util.get_loss_func(is_multi_label = self.is_multi_label)
			self.k_fn_loss = self.kfunc_util.gen_pred_and_loss_ops(
				predictions.shape, predictions.dtype, labels.shape, labels.dtype, loss_func)

		losses_of_all = self.k_fn_loss([predictions, labels])[0]
		#t2 = time.time()
		#print ("Time for pred prob and loss: {}".format(t2 - t1))

		losses_of_correct = losses_of_all[self.indices_to_correct]
		indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.)[0]
		num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
		new_losses_of_correct = num_corr_true + self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1))

		losses_of_wrong = losses_of_all[self.indices_to_wrong]
		indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.)[0]
		num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
		new_losses_of_wrong = num_wrong_true + self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1))
		
		combined_losses	= (new_losses_of_correct, new_losses_of_wrong)
		return predictions, correct_predictions, combined_losses
		

	def get_results_of_target(self, deltas, indices_to_target):
		"""
		Return the results of the target (can be accessed by indices_to_target)
			-> results are compute for currnet self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		deltas_as_lst = [deltas[idx_to_tl] for idx_to_tl in self.indices_to_target_layers]
		predictions = self.kfunc_util.compute_predictions(
			self.fn_mdl_lst, self.labels, deltas_as_lst, batch_size = self.batch_size)
	
		if self.is_multi_label:
			correct_predictions = self.np.argmax(predictions, axis = -1)
			y_labels = self.np.argmax(self.labels, axis = 1)
			if correct_predictions.shape != y_labels.shape:
				correct_predictions = correct_predictions.reshape(y_labels.shape)
			correct_predictions = correct_predictions == y_labels
		else:
			correct_predictions = self.np.round(predictions).flatten() == self.labels

		target_corr_predcs = correct_predictions[indices_to_target]
		num_of_total_target = len(target_corr_predcs)
		msg = "%d vs %d" % (num_of_total_target, len(indices_to_target))
		assert num_of_total_target == len(indices_to_target), msg
		correctly_classified = self.np.sum(target_corr_predcs)

		return (correctly_classified, correctly_classified / num_of_total_target)


	def get_results_of_target_lstm(self, deltas, indices_to_target):
		"""
		Return the results of the target (can be accessed by indices_to_target)
			-> results are compute for currnet self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		predictions = self.predict_with_new_delta(deltas)
		if predictions.shape != self.labels.shape:
			to_this_shape = self.labels.shape
			predictions = self.np.reshape(predictions, to_this_shape)

		if self.is_multi_label:
			correct_predictions = self.np.argmax(predictions, axis = 1)
			correct_predictions = correct_predictions == self.np.argmax(self.labels, axis = 1)
		else:
			correct_predictions = self.np.round(predictions).flatten() == self.labels

		target_corr_predcs = correct_predictions[indices_to_target]

		num_of_total_target = len(target_corr_predcs)
		msg = "%d vs %d" % (num_of_total_target, len(indices_to_target))
		assert num_of_total_target == len(indices_to_target), msg
		correctly_classified = self.np.sum(target_corr_predcs)

		return (correctly_classified, correctly_classified / num_of_total_target)


	def get_number_of_patched(self, deltas): 
		"""
		Return a number of patched initially wrongly classified outputs 
		(can be accessed by self.indices_to_wrong)
		=> compute for currnent self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		if self.lstm_mdl:
			correctly_classified, perc_correctly_classifed = self.get_results_of_target_lstm(
				deltas, 
				self.indices_to_wrong)
		else:
			correctly_classified, perc_correctly_classifed = self.get_results_of_target(
				deltas, 
				self.indices_to_wrong) 

		return (correctly_classified, perc_correctly_classifed)


	def get_number_of_violated(self, deltas): 
		"""
		Return a number of patched initially correctly classified outputs 
		(can be accessed by self.indices_to_correct)
		=> compute for currnent self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		target_indices = self.indices_to_correct
		if self.lstm_mdl:
			correctly_classified, perc_correctly_classifed = self.get_results_of_target_lstm(
				deltas, 
				target_indices)
		else:
			correctly_classified, perc_correctly_classifed = self.get_results_of_target(
				deltas,
				target_indices) 

		num_of_initially_correct = len(target_indices)
		return (num_of_initially_correct - correctly_classified, 1.0 - perc_correctly_classifed)


	def get_num_patched_and_broken(self, predictions):
		"""
		compute fitness using a given predictions and correctly
		Ret (boolean, float):
			boolean: True if restriction is not violated, otherwise, False
			float: new fitness value
		"""
		if self.is_multi_label:
			new_classifcation_results = self.np.argmax(predictions, axis = 1)
		else:
			new_classifcation_results = self.np.round(predictions).flatten()

		# correct -> incorrect
		num_violated = self.np.sum(
			(new_classifcation_results != self.ground_truth_labels)[self.indices_to_correct])
		# incorrect (wrong) -> corret
		num_patched = self.np.sum(
			(new_classifcation_results == self.ground_truth_labels)[self.indices_to_wrong])

		return num_violated, num_patched	


	def check_early_stop(self, fitness_value, new_weights, model_name = None):
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
		
		num_of_patched, perc_num_of_patched = self.get_number_of_patched(new_weights)
		num_of_violated, perc_num_of_violated = self.get_number_of_violated(new_weights)

		if num_of_patched == len(self.indices_to_wrong) and num_of_violated == 0:
			print("In early stop checking:%d, %d" % (num_of_patched, num_of_violated))
			print ("\t fitness values", fitness_value)
			print ("\t", num_of_patched == len(self.indices_to_wrong) and num_of_violated == 0)
			return True, num_of_patched
		else:
			print ("in early stop checking", "{} ({}), {} ({})".format(
				num_of_patched, perc_num_of_patched, num_of_violated, perc_num_of_violated))
			return False, num_of_patched


	def summarise_results(self, deltas):
		"""
		Print out the current result of model_name
		=> compute for currnent self.mdl
		"""
		print ("***Patching results***\n")

		num_of_patched, perc_num_of_patched = self.get_number_of_patched(deltas)
		num_of_violated, perc_num_of_violated = self.get_number_of_violated(deltas)

		print ("\tFor initially wrongly classified:%d -> %d(%f)" % (
			len(self.indices_to_wrong), \
			len(self.indices_to_wrong) - num_of_patched, perc_num_of_patched))
		print ("\tFor initially correctly classified(violation):%d -> %d(%f)" % (
			len(self.indices_to_correct),\
			len(self.indices_to_correct)- num_of_violated, perc_num_of_violated))


