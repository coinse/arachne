"""
Use differential evolution
"""
import numpy as np
from numpy.lib.function_base import delete
from search.searcher_vk import Searcher

class RT_searcher(object):
	"""docstring for DE_searcher"""
	random = __import__('random')
	operator = __import__('operator')
	np = __import__('numpy')

	def __init__(self,
		inputs, labels,
		indices_to_correct, indices_to_wrong,
		num_label,
		indices_to_target_layers,
		path_to_keras_model = None):
		"""
		"""
		# set attributes
		self.inputs = inputs
		self.labels = labels
		self.num_label = num_label
		self.indices_to_correct = indices_to_correct
		self.indices_to_wrong = indices_to_wrong
		self.indices_to_target_layers = indices_to_target_layers
		self.path_to_keras_model = path_to_keras_model


		pass

	def set_base_model(self):
		from tensorflow.keras.models import load_model
		self.mdl = load_model(self.path_to_keras_model, compile = False)


	def reset_keras(self, delete_list = None):
		import tensorflow as tf
		import tensorflow.keras.backend as K
		
		if delete_list is None:
			K.clear_session()
			s = tf.InteractiveSession()
			K.set_session(s)
		else:
			import gc
			#sess = K.get_session()
			K.clear_session()
			#sess.close()
			sess = K.get_session()
			try:
				for d in delete_list:
					del d
			except:
				pass

		print(gc.collect()) # if it's done something you should see a number being outputted

		# use the same config as you used to create the session
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 1
		config.gpu_options.visible_device_list = "0"
		K.set_session(tf.Session(config=config))
		
		# load model agai
		self.set_base_model()
 
	# def eval(self, patch_candidate, places_to_fix): 
	# 	"""
	# 	Evaluate a given trial vector
	# 	Use a given part, a new weight value vector, to update
	# 	a target tensor, here, weight (& bias), and evaluate a new DNN model
	# 	wiht the updated weight tensor

	# 	Argumnets:
	# 		patch_candidate: a trial candiate vector
	# 		places_to_fix: [(idx_to_tl, inner_indices) .. ]
	# 	Ret: (fitness)
	# 	"""
	# 	import time

	# 	deltas = {} # this is deltas for set update op
	# 	for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
	# 		if idx_to_tl not in deltas.keys():
	# 			deltas[idx_to_tl] = self.init_weights[idx_to_tl]
	# 		# since our op is set
	# 		deltas[idx_to_tl][tuple(inner_indices)] = patch_candidate[i]
	# 	###################

	# 	predictions, correct_predictions, loss_v = self.move_v2(deltas, update_op = 'set')

	# 	assert predictions is not None
	# 	assert correct_predictions is not None
	# 	num_violated, num_patched = self.get_num_patched_and_broken(
	# 		predictions) 

	# 	losses_of_correct, losses_of_target = loss_v
	# 	if self.patch_aggr is None:
	# 		final_fitness = losses_of_correct + losses_of_target
	# 	else:
	# 		final_fitness = losses_of_correct + self.patch_aggr * losses_of_target	

	# 	num_still_correct = len(self.indices_to_correct) - num_violated # N_intac
	# 	# log
	# 	#print ("New fitness (num_violatd, num_patched)",
	# 	#	self.patch_aggr, losses_of_correct, losses_of_target, num_violated, num_patched, final_fitness)
			
	# 	return (final_fitness,)

	def freeze_neurons(self):
		"""
		"""
		for idx_to_tl in self.indices_to_target_layers:
			self.mdl.layers[idx_to_tl].freeze 

		num_layers = len(self.mdl.layers)
		for idx_to_l in range(num_layers):
			if idx_to_l not in self.indices_to_target_layers:
				self.mdl.layers[idx_to_l].trainable = False
			#else:
			#	pass

	def compute_mask_to_frozen_in_tl(self, places_to_fix):
		"""
		"""
		masks = {}
		for idx_to_tl, inner_indices in places_to_fix.items():
			ws = self.mdl.layers[idx_to_tl].get_weights()
			w_shape = ws[0].shape
			## generate mask!!!
			# should I sort the mask ..? no, I don't think it is neccessary
			masks[idx_to_tl] = list(set(list(self.np.ndindex(w_shape))) - set(inner_indices))

		return masks

	def get_init_weights(self):
		"""
		"""
		ws = {}
		for idx_to_tl in self.indices_to_target_layers:
			ws[idx_to_tl] = self.mdl.get_weights()[0]
		return ws


	def train(self,
		places_to_fix,
		batch_size = 256,
		name_key = 'best'):
		"""
		free the neurons in the places_to_fix argument
		"""
		import tensorflow as tf
		##

		self.freeze_neurons()
		masks = self.compute_mask_to_frozen_in_tl(places_to_fix)
		init_weights = self.get_init_weights()

		##
		loss_fn = tf.keras.losses.CategoryCrossentropy(from_logits = False) # since predictions is not in the range of 0 and 1
		optimizer = tf.keras.optimizers.Adam()

		## batch 
		n_inputs = len(self.inputs)
		n_splits = int(np.round(n_inputs/batch_size))
		indices_to_batches = np.split(np.arange(n_inputs), n_splits)

		# train
		for curr_batch_indices in indices_to_batches:
			inputs = self.inputs[curr_batch_indices]
			targets = self.labels[curr_batch_indices]
			## train

			# set the weight to the original value
			# Open a GradientTape.
			with tf.GradientTape() as tape:
				# Forward pass.
				predictions = self.mdl(inputs)
				# Compute the loss value for this batch.
				loss_value = loss_fn(targets, predictions)

			# get gradients of loss wrt the *trainable* weights.
			# but, this also means that the gradient itself is computed for the entire weight variable, which is a redundnat cost
			# .... hmmm ... doesn't seem fair;;;
			gradients = tape.gradient(loss_value, self.mdl.trainable_weights)
			
			# update the weights of the model.
			optimizer.apply_gradients(zip(gradients, self.mdl.trainable_weights))			

			# set the weights hack to the initial values
			for idx_to_tl, mask in masks.items():
				# set back to the initial value
				self.mdl.layers[idx_to_tl][mask] = init_weights[idx_to_tl][mask]

		pass


