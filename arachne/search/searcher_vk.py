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
	#apricot_rel_util = importlib.import_module('utils.apricot_rel_util')
	#torch_rel_util = importlib.import_module('utils.torch_rel_util')

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
		indices_to_target_layers, 
		max_search_num = 200,
		initial_predictions = None,
		path_to_keras_model = None,
		at_indices = None, 
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
			#self.lstm_mdl = False # curretly
		else:
			self.labels = labels
			self.ground_truth_labels = labels
			#self.lstm_mdl = True
		self.lstm_mdl = is_lstm
		print ("This model is a lstm model: {}".format(self.lstm_mdl))
		self.model_name = "model"
		self.model_name_format = self.model_name + ".{}"

		self.indices_to_correct = indices_to_correct
		self.indices_to_wrong = indices_to_wrong

		# model related initialisation
		self.path_to_keras_model = path_to_keras_model

		#self.which = which
		#self.tensors = self.set_target_tensor_names(tensor_name_file)
		self.indices_to_target_layers = indices_to_target_layers # !!!!!! here, include (idx_to_tl, idx_to_w (0 or 1))
		self.targeted_layer_names = None
		##
		self.batch_size = batch_size
		self.act_func = act_func # will be latter used for GTSRB
		self.is_multi_label = is_multi_label
		#self.set_base_model()
		if not self.lstm_mdl:
			self.set_base_model_v2()
		else:
			self.set_base_model_v3()
		self.set_target_weights()

		self.at_indices = at_indices ## related to RQ6 ... not sure what this is

		#self.sess = None
		# initialise the names of the tensors used in Searcher
		if initial_predictions is None:
			self.set_initial_predictions()# initial_predictions)
		else:
			self.initial_predictions = initial_predictions
		self.maximum_fitness = 0.0 # the maximum fitness value	
		# set search relate parameters
		self.max_search_num = max_search_num	
		self.indices_to_sampled_correct = None

	# @classmethod
	# def get_which_tensors(cls):
	# 	"""
	# 	Return a list of tensors to look at
	# 	"""
	# 	return cls.which_tensors

	#def generate_empty_graph(self):
	def set_base_model(self):
		"""
		Generate an empyt graph frame for current searcher
		"""
		from tensorflow.keras.models import load_model, Model
		#idx_to_first_tl = self.np.min(self.indices_to_target_layers)
		mdl = load_model(self.path_to_keras_model)
		#self.mdl = Model(inputs = mdl.inputs, outputs = mdl.layers[idx_to_first_tl-1].output)
		self.mdl = mdl
		print ("Number of layers in model: {}".format(len(self.mdl.layers)))

		###
		from gen_frame_graph import build_k_frame_model
		# fn, t_ws (for weights), ys (for labels)
		print ("INPUT FORMAT", self.inputs.shape) 
		self.fn_mdl, _, _  = build_k_frame_model(self.mdl, self.inputs, self.indices_to_target_layers)
	

	def set_base_model_v2(self):
		"""
		Generate an empyt graph frame for current searcher
		"""
		from tensorflow.keras.models import load_model
		mdl = load_model(self.path_to_keras_model)
		self.mdl = mdl
		print ("Number of layers in model: {}".format(len(self.mdl.layers)))

		self.fn_mdl_lst = self.kfunc_util.generate_base_mdl(	
			self.mdl, self.inputs, indices_to_tls = self.indices_to_target_layers, # here, indices_to_target_layers is a (true) list of layers
			batch_size = self.batch_size, act_func = self.act_func)
		# store prev output here


	def set_base_model_v3(self):
		"""
		generate a list of Model instances -> used with move_v3
		here, set previous ouputs and model
		"""
		from tensorflow.keras.models import load_model
		from collections.abc import Iterable
		from tensorflow.keras.models import Model
		import tensorflow as tf

		mdl = load_model(self.path_to_keras_model, compile = False)
		self.mdl = mdl
		print ("Number of layers in model: {}".format(len(self.mdl.layers)))
		# set targete layer names
		self.targeted_layer_names = {}
		for idx_to_tl in self.indices_to_target_layers:
			self.targeted_layer_names[idx_to_tl] = type(self.mdl.layers[idx_to_tl]).__name__

		# compute previous outputs
		self.min_idx_to_tl = self.np.min([idx if not isinstance(idx, Iterable) else idx[0] for idx in self.indices_to_target_layers])
		self.input_layer_added = False
		if self.min_idx_to_tl == 0:
			if not self.model_util.is_Input(type(self.mdl.layers[0]).__name__):
				self.input_layer_added = True

		print ("index", self.min_idx_to_tl-1 if self.min_idx_to_tl > 0 else 0, "and min idx", self.min_idx_to_tl)
		prev_l = self.mdl.layers[self.min_idx_to_tl-1 if self.min_idx_to_tl > 0 else 0]
		if self.model_util.is_Input(type(prev_l).__name__): # previous layer is an input layer
			self.prev_outputs = self.inputs
		else: # otherwise, compute the output of the previous layer
			t_mdl = Model(inputs = self.mdl.input, outputs = prev_l.output)
			print ("temporaray model")
			print (t_mdl.summary())		
			self.prev_outputs = t_mdl.predict(self.inputs)

		# set base model
		from gen_frame_graph import build_mdl_lst
		# a list that contains a single model
		self.fn_mdl_lst = [build_mdl_lst(self.mdl, self.prev_outputs.shape[1:], sorted(self.indices_to_target_layers))]

		# set chunks
		self.chunks = self.data_util.return_chunks(len(self.inputs), batch_size = self.batch_size)

		# also set softmax and loss op
		self.k_fn_loss = None 

	def set_target_weights(self):
		"""
		"""
		if self.mdl is None:
			self.set_base_model(self.path_to_keras_model)
		
		self.init_weights = {}
		self.init_biases = {}
		for idx_to_tl in self.indices_to_target_layers:
			#self.init_weights[idx_to_tl] = self.mdl.layers[idx_to_tl].get_weights()[0]
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

	# def set_target_tensor_names(self, tensor_name_file):
	# 	"""
	# 	Read a given tensor_name_file and set target tensors
	# 	"""
	# 	assert self.os.path.exists(tensor_name_file), "%s does not exist" % (tensor_name_file)
	# 	with open(tensor_name_file) as f:
	# 		lines = [line.strip() for line in f.readlines()]
	# 	tensors = {}
	# 	for line in lines:
	# 		terms = line.split(",")
	# 		assert len(terms) >= 2, "%s should contain at least two terms" % (line)
	# 		which_tensor = terms[0]	
	# 		assertion_violate_msg = "%s is not one of %s" % (which_tensor, ",".join(Searcher.which_tensors))
	# 		assert which_tensor in Searcher.which_tensors, assertion_violate_msg
	# 		tensors[which_tensor] = terms[1] if len(terms) == 2 else terms[1:]
	# 	return tensors

	# def gen_feed_dict(self, values_dict):
	# 	"""
	# 	Arguments:
	# 		values_dict: key = name of a placeholder, value = initial value
	# 	Ret (dict):
	# 		returm an initialised feed_dict
	# 	"""
	# 	assertion_violate_msg = "%s vs %s" % (','.join(list(values_dict.keys())),\
	# 		','.join(Searcher.var_placeholders[self.which]))
	#	
	# 	feed_dict = {}
	#
	# 	for placeholder_name in values_dict.keys():
	# 		placeholder_tensor = self.model_util.get_tensor(
	# 			placeholder_name, 
	# 			self.empty_graph, 
	# 			sess = None)
	#
	# 		feed_dict[placeholder_tensor] = values_dict[placeholder_name]
	#
	# 	return feed_dict

	# to repair for any weights, then this method should be fixed
	# 
	# def initialise_feed_dict(self):
	# 	"""
	# 	Initialise value_dict using self.meta_file and self.model_name
	# 	and get the matching variable values with the placeholders
	# 	"""
	# 	import time
	# 	values_dict = {}
	# 	if self.which != 'lfw_vgg':
	# 		kernel_and_bias_pairs = self.apricot_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)#6)#4)
	# 	else:
	# 		kernel_and_bias_pairs = self.torch_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)
	#
	# 	# fb1,fw1, fb2,fw2, fb3,fw3
	# 	weights = {}
	# 	weights["fw3"] = kernel_and_bias_pairs[-1][0]
	# 	weights["fb3"] = kernel_and_bias_pairs[-1][1]
	# 	values_dict = weights
	#
	# 	feed_dict = self.gen_feed_dict(values_dict)
	#
	# 	self.init_feed_dict = feed_dict.copy()
	# 	self.curr_feed_dict = feed_dict.copy()


	# def compute_initial_predictions(self, inputs = None, labels = None, num_label = None, 
	# 	base_indices_to_vgg16 = None):
	# 	"""
	# 	"""
	# 	if inputs is None:
	# 		inputs = self.inputs
	# 	if labels is None:
	# 		labels = self.labels
	# 	if num_label is None:
	# 		num_label = self.num_label
	#
	# 	sess, (initial_predictions, _) = self.model_util.predict(
	# 		inputs, labels, num_label,
	# 		predict_tensor_name = self.tensors['t_prediction'], 
	# 		corr_predict_tensor_name = None,
	# 		indices_to_slice_tensor_name = 'indices_to_slice' if self.w_gather else None,
	# 		sess = None, 
	# 		empty_graph = self.empty_graph,
	# 		plchldr_feed_dict = self.curr_feed_dict.copy(),
	# 		use_pretr_front = self.path_to_keras_model is not None)
	# 	# here, close the session
	# 	sess.close()
	#
	# 	return initial_predictions


	# def set_initial_predictions(self, initial_predictions = None):
	# 	"""
	# 	Set initial prediction results of the model that Searcher tries to optimise
	# 	"""
	# 	if initial_predictions is None:
	# 		is_data_all_prepared = self.inputs is not None and self.labels is not None and self.num_label is not None,
	# 		is_model_available = self.meta_file is not None and self.model_name is not None
	# 		is_tensor_name_set = self.tensors['t_prediction'] is not None	
	#
	# 		assertion_violate_msg = "None check: inputs:%s, labels:%s, num_label:%s, \
	# 			meta_file: %s, model_name:%s, t_prediction:%s" % (str(self.inputs is not None),
	# 				str(self.labels is not None), str(self.num_label is not None),
	# 				str(self.meta_file is not None), str(self.model_name is not None),
	# 				str(self.tensors['t_prediction'] is not None))	
	#
	# 		assert is_data_all_prepared and is_tensor_name_set, assertion_violate_msg	
	# 		# does not have to be on-the-fly one as it is only initialsation
	#
	# 		initial_predictions = self.compute_initial_predictions(
	# 			inputs = self.inputs, 
	# 			labels = self.labels, 
	# 			num_label = self.num_label,
	# 			base_indices_to_vgg16 = None)
	#
	# 		self.initial_predictions = initial_predictions
	# 	else:
	# 		self.initial_predictions = initial_predictions

	def set_initial_predictions(self):
		"""
		"""
		if self.mdl is None:
			self.set_base_model()
		self.initial_predictions = self.mdl.predict(self.inputs)

	#def move(self, target_tensor_name, delta, new_model_name, 
	#	update_op = 'add', values_dict = None):
	def move_v1(self, deltas, update_op = 'set'):
		"""
		"""
		import tensorflow as tf
		#t0 = time.time()
		inputs = self.inputs
		labels = self.labels

		import time
		t1 = time.time()
		for idx_to_tl in self.indices_to_target_layers:
			_t0 = time.time()
			w_org, b = self.mdl.layers[idx_to_tl].get_weights() # get current weights
			_t1 = time.time()
			#print ("++", _t1 - _t0)
			if update_op == 'add':
				self.mdl.layers[idx_to_tl].set_weights([w_org + deltas[idx_to_tl], b])
			elif update_op == 'sub':
				self.mdl.layers[idx_to_tl].set_weights([w_org - deltas[idx_to_tl], b])
			else: # set
				self.mdl.layers[idx_to_tl].set_weights([deltas[idx_to_tl], b])
		t2 = time.time()

		t1 =time.time()	
		predictions = self.mdl.predict(inputs)
		correct_predictions = self.np.argmax(predictions, axis = 1)
		correct_predictions = correct_predictions == self.np.argmax(labels, axis = 1)
		t2 =time.time()
		print (correct_predictions.shape, self.np.sum(correct_predictions))
		print ('Time for pred: {}'.format(t2 - t1))
		t1 = time.time()
		pred_probs = tf.math.softmax(predictions) # softmax
		loss_op = tf.keras.metrics.categorical_crossentropy(labels, pred_probs)
		with tf.Session() as sess:
			losses_of_all = sess.run(loss_op)
		t2 = time.time()
		#print ("Time for sess: {}".format(t2 - t1))
		#print ("Loss", losses_of_all.shape)
		#print (np.max(self.indices_to_correct))
		losses_of_correct = losses_of_all[self.indices_to_correct]
		
		##
		indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.)[0]
		num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
		new_losses_of_correct = num_corr_true + self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1))
		##

		losses_of_wrong = losses_of_all[self.indices_to_wrong]
		##
		indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.)[0]
		num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
		new_losses_of_wrong = num_wrong_true + self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1))
		##
		combined_losses	= (new_losses_of_correct, new_losses_of_wrong)

		return predictions, correct_predictions, combined_losses
		#return sess, (predictions, correct_predictions, combined_losses)
		

	def get_results_of_target(self, indices_to_target):
		"""
		Return the results of the target (can be accessed by indices_to_target)
			-> results are compute for currnet self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		# sess, (_, correct_predictions) = self.model_util.predict(
		# 	self.inputs, self.labels, self.num_label,
		# 	predict_tensor_name = None,
		# 	corr_predict_tensor_name = self.tensors['t_correct_prediction'],
		# 	indices_to_slice_tensor_name = 'indices_to_slice' if self.w_gather else None,
		# 	sess = self.sess, 
		# 	empty_graph =  empty_graph,
		# 	plchldr_feed_dict = plchldr_feed_dict,
		# 	use_pretr_front = self.path_to_keras_model is not None,
		# 	compute_loss = False)

		predictions = self.mdl.predict(self.inputs)
		correct_predictions = self.np.argmax(predictions, axis = 1)
		correct_predictions = correct_predictions == self.np.argmax(self.labels, axis = 1)
		##

		##

		target_corr_predcs = correct_predictions[indices_to_target]

		num_of_total_target = len(target_corr_predcs)
		assert num_of_total_target == len(indices_to_target), "%d vs %d" % (num_of_total_target, len(indices_to_target))
		correctly_classified = self.np.sum(target_corr_predcs)

		return (correctly_classified, correctly_classified / num_of_total_target)


	def move_v2(self, deltas, update_op = 'set'):
		"""
		*** should be checked and fixed
		"""
		import tensorflow as tf
		import time
		#t0 = time.time()
		#inputs = self.inputs
		labels = self.labels
		t1 = time.time()
		deltas_as_lst = [deltas[idx_to_tl] for idx_to_tl in self.indices_to_target_layers if idx_to_tl in deltas.keys()] 
		#predictions_o, losses_of_all_o = self.fn_mdl_lst[0](deltas_as_lst + [labels])
		#**
		#predictions = self.kfunc_util.compute_predictions(self.fn_mdl_lst, labels, deltas_as_lst, batch_size = self.batch_size)
		#losses_of_all = self.kfunc_util.compute_losses(self.fn_mdl_lst, labels, deltas_as_lst, batch_size = self.batch_size)
		predictions, losses_of_all = self.kfunc_util.compute_preds_and_losses(self.fn_mdl_lst, labels, deltas_as_lst, batch_size = self.batch_size)
		#print (predictions.shape, predictions_o.shape)
		#print (losses_of_all.shape, losses_of_all_o.shape)
		#print (losses_of_all[0], losses_of_all_o[0])
		#print (all((predictions == predictions_o).flatten()))
		#print (all((losses_of_all == losses_of_all_o).flatten()))
		#import sys; sys.exit()
		#**
		#
		if len(predictions.shape) > len(labels.shape) and predictions.shape[1] == 1:
			predictions = self.np.squeeze(predictions, axis = 1)
		
		#if len(predictions.shape) >= 2 and predictions.shape[-1] > 1: # first check whether this task is multi-class classification
		if self.is_multi_label:
			correct_predictions = self.np.argmax(predictions, axis = 1)
			correct_predictions = correct_predictions == self.np.argmax(labels, axis = 1)
		else:
			correct_predictions = self.np.round(predictions).flatten() == labels

		#print (correct_predictions.shape, self.np.sum(correct_predictions))
		t2 =time.time()
		#print ("Time for move: {}".format(t2 - t1))
		#print ("Time for sess: {}".format(t2 - t1))
		#print ("Loss", losses_of_all.shape)å
		#print (np.max(self.indices_to_correct))

		losses_of_correct = losses_of_all[self.indices_to_correct]
		##
		indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.)[0]
		num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
		new_losses_of_correct = num_corr_true + self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1))
		##

		losses_of_wrong = losses_of_all[self.indices_to_wrong]
		##
		indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.)[0]
		num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
		new_losses_of_wrong = num_wrong_true + self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1))
		##
		combined_losses	= (new_losses_of_correct, new_losses_of_wrong)
		#print ("==", "**", self.is_multi_label, self.lstm_mdl, self.num_label, num_corr_true, self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1)), num_wrong_true, self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1)))
		return predictions, correct_predictions, combined_losses
		#return sess, (predictions, correct_predictions, combined_losses)
		
	def predict_with_new_delat(self, deltas):
		"""
		predict with the model patched using deltas
		"""
		from collections.abc import Iterable
		import time 
		t1 = time.time()
		# prepare a new model to run by updating the weights from deltas
		fn_mdl = self.fn_mdl_lst[0] # we only have a one model as this one accept any lenghts of an input, which is actually the output of the previous layers
		#print ("targets", deltas.keys())
		for idx_to_tl, delta in deltas.items(): # either idx_to_tl or (idx_to_tl, i)
			#print ("** delta: {} **".format(idx_to_tl))
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
				fn_mdl.layers[local_idx_to_l].set_weights([new_kernel_w, new_recurr_kernel_w, self.init_biases[idx_to_t_mdl_l]])
			else:
				print ("{} not supported".format(lname))
				assert False
		#import sys; sys.exit()
		t2 = time.time()
		#print ("Time for setting weights: {}".format(t2 - t1))

		predictions = None
		#print ("number of chunks", len(self.chunks), len(self.chunks[0]))
		#print (self.chunks[0].shape)
		#print (self.prev_outputs.shape)
		#print (fn_mdl.summary())
		for chunk in self.chunks:
			_t1 =time.time()
			#_predictions = fn_mdl(self.prev_outputs[chunk], training = False)
			#print (type(chunk))
			#print (chunk.shape)
			#print (self.prev_outputs.shape)
			#print (chunk[:10])
			#print (self.prev_outputs[[0,1,2]].shape)
			#print ("prev otuput", self.prev_outputs[chunk].shape) 
			_predictions = fn_mdl.predict(self.prev_outputs[chunk], batch_size = len(chunk))
			_t2= time.time()
			#print ("time for pure predict: {}".format(_t2 - _t1))
			if predictions is None:
				predictions = _predictions
			else:
				predictions = self.np.append(predictions, _predictions, axis = 0)

		t3 = time.time()
		#print ("Time for predictions: {}".format(t3 - t2))
		#import sys;sys.exit()
		return predictions


	def move_v3(self, deltas):
		"""
		*** should be checked and fixed
		--> need to fix this...
		delatas -> key: idx_to_tl & inner_key: index to the weight
				or key: (idx_to_tl, i) & inner_key
				value -> the new value
		"""
		import time

		labels = self.labels
		predictions = self.predict_with_new_delat(deltas)
		#print (predictions)
		# due to the data dimention of fashion_mnist, 
		if predictions.shape != labels.shape:
			to_this_shape = labels.shape
			predictions = self.np.reshape(predictions, to_this_shape)

		# the softmax or any other activation function should be inserted here!!!!! ... but, again think abou it
		# since the applicaiton itself is argmax, the tendency itself doesn't change ... ok 
		#if len(predictions.shape) >= 2 and predictions.shape[-1] > 1: # first check whether this task is multi-class classification
		t1 = time.time()
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
		t2 = time.time()
		#print ("Time for pred prob and loss: {}".format(t2 - t1))
		losses_of_correct = losses_of_all[self.indices_to_correct]
		##
		indices_to_corr_false = self.np.where(correct_predictions[self.indices_to_correct] == 0.)[0]
		num_corr_true = len(self.indices_to_correct) - len(indices_to_corr_false)
		new_losses_of_correct = num_corr_true + self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1))
		##

		losses_of_wrong = losses_of_all[self.indices_to_wrong]
		##
		indices_to_wrong_false = self.np.where(correct_predictions[self.indices_to_wrong] == 0.)[0]
		num_wrong_true = len(self.indices_to_wrong) - len(indices_to_wrong_false)
		new_losses_of_wrong = num_wrong_true + self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1))
		##
		print ("==", "**", self.is_multi_label, self.lstm_mdl, self.num_label, num_corr_true, self.np.sum(1/(losses_of_correct[indices_to_corr_false] + 1)), num_wrong_true, self.np.sum(1/(losses_of_wrong[indices_to_wrong_false] + 1)))
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
		# sess, (_, correct_predictions) = self.model_util.predict(
		# 	self.inputs, self.labels, self.num_label,
		# 	predict_tensor_name = None,
		# 	corr_predict_tensor_name = self.tensors['t_correct_prediction'],
		# 	indices_to_slice_tensor_name = 'indices_to_slice' if self.w_gather else None,
		# 	sess = self.sess, 
		# 	empty_graph =  empty_graph,
		# 	plchldr_feed_dict = plchldr_feed_dict,
		# 	use_pretr_front = self.path_to_keras_model is not None,
		# 	compute_loss = False)

		#predictions = self.mdl.predict(self.inputs)
		## v2
		deltas_as_lst = [deltas[idx_to_tl] for idx_to_tl in self.indices_to_target_layers]
		#predictions, _ = self.fn_mdl(deltas_as_lst + [self.labels])
		## **
		predictions = self.kfunc_util.compute_predictions(self.fn_mdl_lst, self.labels, deltas_as_lst, batch_size = self.batch_size)
		#if len(predictions.shape) >= 2 and predictions.shape[-1] > 1: # first check whether this task is multi-class classification
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
		assert num_of_total_target == len(indices_to_target), "%d vs %d" % (num_of_total_target, len(indices_to_target))
		correctly_classified = self.np.sum(target_corr_predcs)

		return (correctly_classified, correctly_classified / num_of_total_target)


	def get_results_of_target_v3(self, deltas, indices_to_target):
		"""
		Return the results of the target (can be accessed by indices_to_target)
			-> results are compute for currnet self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		predictions = self.predict_with_new_delat(deltas)
                # due to the data dimention of fashion_mnist,
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
		assert num_of_total_target == len(indices_to_target), "%d vs %d" % (num_of_total_target, len(indices_to_target))
		correctly_classified = self.np.sum(target_corr_predcs)

		return (correctly_classified, correctly_classified / num_of_total_target)

	def get_number_of_patched(self, deltas): #, 
		#empty_graph = None, plchldr_feed_dict = None):
		"""
		Return a number of patched initially wrongly classified outputs 
		(can be accessed by self.indices_to_wrong)
		=> compute for currnent self.mdl
		Ret (int, float):
			int: the number of patched
			float: percentage of the number of patched)
		"""
		if self.lstm_mdl:
			correctly_classified, perc_correctly_classifed = self.get_results_of_target_v3(
				deltas, 
				self.indices_to_wrong)
		else:
			correctly_classified, perc_correctly_classifed = self.get_results_of_target(
				deltas, 
				self.indices_to_wrong) 

		return (correctly_classified, perc_correctly_classifed)


	def get_number_of_violated(self, deltas): #sess = None, 
		#empty_graph = None, plchldr_feed_dict = None):
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
			correctly_classified, perc_correctly_classifed = self.get_results_of_target_v3(
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
		# get the final clasification of labels  of the newly computed predictions 
		#new_classifcation_results = self.np.argmax(predictions, axis = 1)	
		#if len(predictions.shape) >= 2 and predictions.shape[-1] > 1: # first check whether this task is multi-class classification
		if self.is_multi_label:
			new_classifcation_results = self.np.argmax(predictions, axis = 1)
		else:
			new_classifcation_results = self.np.round(predictions).flatten()

		num_violated = self.np.sum((new_classifcation_results != self.ground_truth_labels)[self.indices_to_correct]) # correct -> incorrect
		num_patched = self.np.sum((new_classifcation_results == self.ground_truth_labels)[self.indices_to_wrong]) # incorrect (wrong) -> corret

		return num_violated, num_patched	


	# def check_early_stop(self, 
	# 	new_weight_value, 
	# 	fitness_value, 
	# 	model_name = None, sess = None, empty_graph = None):
	#def check_early_stop(self, fitness_value, model_name = None):
	def check_early_stop(self, fitness_value, new_weights, model_name = None):
		"""
		Check whether early stop is possible or not
		=> compute for currnent self.mdl (actually, the model is set with the best value before calling this)
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
			#empty_graph = self.empty_graph)

		num_of_violated, perc_num_of_violated = self.get_number_of_violated(deltas)
			#empty_graph = self.empty_graph)

		print ("\tFor initially wrongly classified:%d -> %d(%f)" % (len(self.indices_to_wrong), \
			len(self.indices_to_wrong) - num_of_patched, perc_num_of_patched))
		print ("\tFor initially correctly classified(violation):%d -> %d(%f)" % (len(self.indices_to_correct),\
			len(self.indices_to_correct)- num_of_violated, perc_num_of_violated))


