"""
Use differential evolution
"""
from numpy.lib.function_base import delete
from search.searcher_vk import Searcher

class DE_searcher(Searcher):
	"""docstring for DE_searcher"""
	random = __import__('random')
	operator = __import__('operator')

	importlib = __import__('importlib')
	base = importlib.import_module('deap.base')
	creator = importlib.import_module('deap.creator')
	tools = importlib.import_module('deap.tools')

	def __init__(self,
		inputs, labels,
		indices_to_correct, indices_to_wrong,
		num_label,
		indices_to_target_layers,
		#tensor_name_file,
		mutation = (0.5, 1), 
		recombination = 0.7,
		max_search_num = 200, # 100
		initial_predictions = None,
		#empty_graph = None,
		#which = None,
		path_to_keras_model = None,
		#is_empty_one = False,
		patch_aggr = None,
		batch_size = None,
		act_func = None,
		at_indices = None):

		"""
		"""
		super(DE_searcher, self).__init__( 
				inputs, labels,
				indices_to_correct, indices_to_wrong,
				num_label,
				indices_to_target_layers,
				#tensor_name_file,
				max_search_num = max_search_num,
				initial_predictions = initial_predictions,
				#empty_graph = empty_graph,
				#which = which,
				path_to_keras_model = path_to_keras_model,
				#is_empty_one = is_empty_one,
				batch_size = batch_size,
				act_func = act_func,
				at_indices = at_indices)

		# fitness computation related initialisation
		self.fitness = 0.0

		self.creator.create("FitnessMax", self.base.Fitness, weights = (1.0,)) # maximisation
		self.creator.create("Individual", self.np.ndarray, fitness = self.creator.FitnessMax, model_name = None)

		# store the best performace seen so far
		self.the_best_performance = None
		self.max_num_of_unchanged = int(self.max_search_num / 10) if self.max_search_num is not None else None
		
		if self.max_num_of_unchanged < 10:
			self.max_num_of_unchanged = 10

		self.num_iter_unchanged = 0

		# DE specific
		self.mutation = mutation
		self.recombination = recombination

		# fitness
		self.patch_aggr = patch_aggr


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
		#self.set_base_model()

	# ######### This will be our fitness function ##################3
	# def eval(self, 
	# 	patch_candidate, 
	# 	#init_weight_value, 
	# 	places_to_fix): #, 
	# 	#new_model_name): #, 
	def eval(self, patch_candidate, places_to_fix): 
		"""
		Evaluate a given trial vector
		Use a given part, a new weight value vector, to update
		a target tensor, here, weight (& bias), and evaluate a new DNN model
		wiht the updated weight tensor

		Argumnets:
			patch_candidate: a trial candiate vector
			places_to_fix: [(idx_to_tl, inner_indices) .. ]
		Ret: (fitness)
		"""
		import time
		# generate new weight value from part and init_weight_value
		# new_weight_value = self.np.zeros(init_weight_value.shape, dtype = init_weight_value.dtype)
		# new_weight_value[:] = init_weight_value
		# # replace places_to_fix node with a matching part value
		# for i, place_to_fix in enumerate(places_to_fix):
		# 	new_weight_value[place_to_fix] = patch_candidate[i]

		deltas = {} # this is deltas for set update op
		for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
			if idx_to_tl not in deltas.keys():
				deltas[idx_to_tl] = self.init_weights[idx_to_tl] ## *** HAVE CHANGED TO ACCEPT (idx_to_tl, idx_to_w(0 or 1))
			# since our op is set
			deltas[idx_to_tl][tuple(inner_indices)] = patch_candidate[i]
		###################

		# sess, (predictions, correct_predictions, loss_v) = self.move(
		# 	self.tensors['t_weight'], 
		# 	new_weight_value, 
		# 	new_model_name, 
		# 	update_op = 'set') 
		#predictions, correct_predictions, loss_v = self.move_v1(deltas, update_op = 'set') 
		#print (deltas[idx_to_tl])
		#predictions, correct_predictions, loss_v = self.move_v2(deltas, update_op = 'set')
		predictions, correct_predictions, loss_v = self.move_v3(deltas)

		assert predictions is not None
		assert correct_predictions is not None
		num_violated, num_patched = self.get_num_patched_and_broken(
			predictions) 

		
		losses_of_correct, losses_of_target = loss_v
		if self.patch_aggr is None:
			final_fitness = losses_of_correct + losses_of_target
		else:
			final_fitness = losses_of_correct + self.patch_aggr * losses_of_target	

		num_still_correct = len(self.indices_to_correct) - num_violated # N_intac
		# log
		#print ("New fitness (num_violatd, num_patched)",
		#	self.patch_aggr, losses_of_correct, losses_of_target, num_violated, num_patched, final_fitness)
			
		return (final_fitness,)


	def is_the_performance_unchanged(self, 
		curr_best_patch_candidate, 
		places_to_fix, 
		sess = None):
		"""
		curr_best_performance: fitness of curr_best_patch_candidate:
		Ret (bool):
			True: the performance
		"""
		curr_best_performance = curr_best_patch_candidate.fitness.values[0]

		if self.the_best_performance is None:
			self.the_best_performance = curr_best_performance
			return False

		if self.np.float32(curr_best_performance) == self.np.float32(self.the_best_performance):
			self.num_iter_unchanged += 1
		else:
			self.num_iter_unchanged = 0 # look for subsequent
			if curr_best_performance > self.the_best_performance:
				self.the_best_performance = curr_best_performance
				
		if self.max_num_of_unchanged < self.num_iter_unchanged:
			return True
		else:
			return False


	def set_bounds(self, init_weight_value):
		
		"""
		"""
		min_v = self.np.min(init_weight_value)
		min_v = min_v * 2 if min_v < 0 else min_v / 2
		
		max_v = self.np.max(init_weight_value)
		max_v = max_v * 2 if max_v > 0 else max_v / 2

		#bounds = [(min_v, max_v)]
		bounds = (min_v, max_v)
		return bounds


	# def search(self,
	# 	places_to_fix,
	# 	pop_size_times = 5,
	# 	#sess = None,
	# 	name_key = 'best', 
	# 	max_allowed_no_change = 10):
	# 	#init_weight_value = None):
	def search(self,
		places_to_fix,
		pop_size_times = 5,
		name_key = 'best', 
		max_allowed_no_change = 10):
		"""
		"""
		import time

		# generate and set empty_graph that is main frame of the our model to patch
		#if self.empty_graph is None:
		#	self.generate_empty_graph()
		# set delta part
		# if init_weight_value is None:
		# 	if self.which != 'lfw_vgg':
		# 		kernel_and_bias_pairs = self.apricot_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)
		# 	else:
		# 		kernel_and_bias_pairs = self.torch_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)\
		# 	weights = {}
		# 	#for i, (w,b) in enumerate(kernel_and_bias_pairs):
		# 	#	weights["fw{}".format(i+1)] = w
		# 	#	weights["fb{}".format(i+1)] = b	
		# 	weights["fw3"] = kernel_and_bias_pairs[-1][0] 
		# 	weights["fb3"] = kernel_and_bias_pairs[-1][1]
		# 	init_weight_value = weights[self.tensors['t_weight']]
		
		num_places_to_fix = len(places_to_fix) # the number of places to fix # NDIM of a patch candidate
		#init_target_ws = init_weight_value[self.np.asarray(places_to_fix)[:,0], self.np.asarray(places_to_fix)[:,1]]

		#mean_value = self.np.mean(init_weight_value)
		#std_value = self.np.std(init_weight_value)
		#min_value = self.np.min(init_weight_value)
		#max_value = self.np.max(init_weight_value)
		#bounds = self.set_bounds(num_places_to_fix, init_weight_value) # for clipping
		bounds = []
		for idx_to_tl, _ in places_to_fix: # ** HAVE CHANGED TO HANDLE (idx_to_tl, idx_to_w (0 or 1))
			_init_weight = self.init_weights[idx_to_tl]
			mean_value = self.np.mean(_init_weight)
			std_value = self.np.std(_init_weight)
			#min_value = self.np.min(_init_weight)
			#max_value = self.np.max(_init_weight)
			bounds.append(self.set_bounds(_init_weight)) # for clipping
		##
		print ("Bounds are set")
		print (bounds)
		print (len(bounds))
		# set search parameters
		pop_size = 100 
		toolbox = self.base.Toolbox()

		def sample_from_noraml(loc = mean_value, scale = std_value):
			return self.np.random.normal(loc = mean_value, scale = std_value, size = 1)[0]

		toolbox.register("attr_float", sample_from_noraml, loc = mean_value, scale = std_value)
		toolbox.register("individual", self.tools.initRepeat, self.creator.Individual, toolbox.attr_float, num_places_to_fix)
		toolbox.register("population", self.tools.initRepeat, list, toolbox.individual)
		toolbox.register("select", self.np.random.choice, size = 3, replace=False)
		toolbox.register("evaluate", self.eval)

		# set logbook
		stats = self.tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", self.np.mean)
		stats.register("std", self.np.std)
		stats.register("min", self.np.min)
		stats.register("max", self.np.max)

		logbook = self.tools.Logbook()
		logbook.header = ['gen', 'evals'] + stats.fields
		# logbook setting end

		# if sess is None:
		# 	self.sess = self.model_util.generate_session(graph = self.empty_graph)
		# else:
		# 	self.sess = sess
		#print (toolbox.individual()) 
		#print (type(toolbox.individual()))
		pop = toolbox.population(n = pop_size)

		#print ('===pop===')
		#print ("\t Places to fix ({}): {}".format(
		#	num_places_to_fix, 
		#	",".join([str(plc) for plc in places_to_fix ])))
		
		hof = self.tools.HallOfFame(1, similar = self.np.array_equal)

		# update fitness
		print ("Places to fix", places_to_fix)
		for ind in pop:
			ind.fitness.values = toolbox.evaluate(ind, places_to_fix)
				#ind,
				#init_weight_value, 
				#places_to_fix) 
				#self.model_name) 
			ind.model_name = None 
	
		hof.update(pop)
		best = hof[0]

		record = stats.compile(pop)
		logbook.record(gen = 0, evals = len(pop), **record)
		print (logbook)
		#import sys; sys.exit()
		search_start_time = time.time()
		# search start
		import time
		for iter_idx in range(self.max_search_num):
			#t_t1 = time.time()
			#self.reset_keras([])
			#t_t2 = time.time()
			#print ('Time for reset: {}'.format(t_t2 - t_t1))
			iter_start_time = time.time()

			MU = self.random.uniform(self.mutation[0], self.mutation[1])
			#print ("Iteration (mu = {}): {} -> {} ({})".format(
			#	MU, iter_idx, best.fitness.values[0], best.fitness))
			# for each population
			#print ("Pop len: {}".format(len(pop)))
			for pop_idx, ind in enumerate(pop):
				t0 = time.time()
				# set model name
				new_model_name = self.model_name_format.format("{}-{}".format(iter_idx, pop_idx))

				# select
				target_indices = [_i for _i in range(pop_size) if _i != pop_idx]
				a_idx, b_idx, c_idx = toolbox.select(target_indices)
				a = pop[a_idx]; b = pop[b_idx]; c = pop[c_idx]

				y = toolbox.clone(ind)

				index = self.random.randrange(num_places_to_fix)
				for i, value in enumerate(ind):
					if i == index or self.random.random() < self.recombination:
						y[i] = self.np.clip(a[i] + MU * (b[i] - c[i]), bounds[i][0], bounds[i][1])
						
				y.fitness.values = toolbox.evaluate(y, places_to_fix)
					#y,
					#init_weight_value, 
					#places_to_fix) 
					#new_model_name)
				
				if y.fitness.values[0] >= ind.fitness.values[0]: # better
					pop[pop_idx] = y # upddate
					# set new model name
					pop[pop_idx].model_name = new_model_name 
					# update best
					if best.fitness.values[0] < pop[pop_idx].fitness.values[0]:
						hof.update(pop)
						best = hof[0]
						#print ("New best one is set: {}, fitness: {}, model_name: {}".format(
						#	best, best.fitness.values[0], best.model_name))
				
			hof.update(pop)
			best = hof[0]
			print ("The best one at Gen {}: {},\n\tfitness: {},\n\tmodel_name: {}".format(
				iter_idx,
				best, 
				best.fitness.values[0], 
				best.model_name))

			# logging for this generation
			record = stats.compile(pop)
			logbook.record(gen = iter_idx, evals = len(pop), **record)
			print (logbook)

			# check for early stop
			# new_weight_value = self.np.zeros(
			# 	init_weight_value.shape, 
			# 	dtype = init_weight_value.dtype)
			# new_weight_value[:] = init_weight_value
			# for i, place_to_fix in enumerate(places_to_fix):
			# 	new_weight_value[place_to_fix] = best[i]

			#########################################################
			# update for best value to check for early stop #########
			#########################################################
			deltas = {} # this is deltas for set update op
			for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
				if idx_to_tl not in deltas.keys():
					deltas[idx_to_tl] = self.init_weights[idx_to_tl]
				# since our op is set
				deltas[idx_to_tl][tuple(inner_indices)] = best[i]
			#self.move(deltas, update_op = 'set') ### v1
				
			# is_early_stop_possible, num_of_patched = self.check_early_stop(best.fitness.values[0], model_name = best.model_name) ### v1
			# 	#new_weight_value, 
			# 	#best.fitness.values[0], 
			# 	#model_name = best.model_name,  
			# 	#empty_graph = self.empty_graph)
			is_early_stop_possible, num_of_patched = self.check_early_stop(
				best.fitness.values[0], deltas, model_name = best.model_name)

			print ("Is early stop possible?: {}".format(is_early_stop_possible))
			prev_best = best

			# check for two stop coniditions
			if self.is_the_performance_unchanged(best, 
				places_to_fix):#
				print ("Performance has not been changed over {} iterations".format(
					self.num_iter_unchanged))
				break

			curr_time = time.time()
			local_run_time = curr_time - iter_start_time
			run_time = curr_time - search_start_time
			print ("Time for a single iter: {}, ({})".format(run_time, local_run_time))
				
		#save_path = self.model_name_format.format(self.max_search_num, name_key)#"best")
		save_path = self.model_name_format.format(name_key)
		best.model_name = save_path ##

		# with these two cases, the new model has not been saved
		#if self.empty_graph is not None:
		if True:
			print ("BEST\n", best)
			print ("\tbest fitness:",best.fitness.values[0])
			print (best.model_name, save_path)

			# generate new weight value from part and init_weight_value
			# new_weight_value = self.np.zeros(init_weight_value.shape, 
			# 	dtype = init_weight_value.dtype)
			# new_weight_value[:] = init_weight_value
			# # replace places_to_fix node with a matching part value
			#for i, place_to_fix in enumerate(places_to_fix):
			# 	new_weight_value[place_to_fix] = best[i]
			# self.move(new_weight_value, update_op = 'set')

			deltas = {} # this is deltas for set update op
			for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
				if idx_to_tl not in deltas.keys():
					deltas[idx_to_tl] = self.init_weights[idx_to_tl]
				# since our op is set
				deltas[idx_to_tl][tuple(inner_indices)] = best[i]
			#self.move(deltas, update_op = 'set') ### v1

			#import json 
			#with open(save_path.replace("None","model") + ".json", 'w') as f:
			#	#f.write(json.dumps({'weight':new_weight_value.tolist()}))	
			#	f.write(json.dumps({'weight':deltas.tolist()}))
			import pickle
			#deltas_df = pd.DataFrame({idx_to_tl:vs for idx_to_tl,vs in deltas.items()})
			#deltas_df.to_pickle(save_path.replace("None","model")+".pkl")
			save_path = save_path.replace("None","model")+".pkl"
			print("The model is initially saved here: {}".format(save_path))
			with open(save_path, 'wb') as f:
				pickle.dump(deltas, f)
			# update self.curr_feed_dict (for processing wrongly classified inputs & outputs)
			# target_tensor = self.model_util.get_tensor(
			# 	self.tensors['t_weight'], 
			# 	self.empty_graph)
			#assert target_tensor in self.curr_feed_dict.keys(), "Target tensor %s should already exists" % (target_tensor_name)
			#self.curr_feed_dict[target_tensor] = new_weight_value
			## summarise the results for the best candidates (best)
			#self.summarise_results() # v1
			self.summarise_results(deltas) # v2
		
		#if self.sess is not None:
		#	self.sess.close()

		return best.model_name, save_path



