"""
Use differential evolution
"""
from search.searcher import Searcher

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
		tensor_name_file,
		mutation = (0.5, 1), 
		recombination = 0.7,
		max_search_num = 200, # 100
		initial_predictions = None,
		empty_graph = None,
		which = None,
		path_to_keras_model = None,
		is_empty_one = False,
		w_gather = False,
		patch_aggr = None,
		at_indices = None,
		use_ewc = False):

		"""
		"""
		super(DE_searcher, self).__init__( 
				inputs, labels,
				indices_to_correct, indices_to_wrong,
				num_label,
				tensor_name_file,
				max_search_num = max_search_num,
				initial_predictions = initial_predictions,
				empty_graph = empty_graph,
				which = which,
				path_to_keras_model = path_to_keras_model,
				is_empty_one = is_empty_one,
				w_gather = w_gather,
				at_indices = at_indices,
				use_ewc = use_ewc)

		# fitness computation related initialisation
		self.fitness = 0.0

		if not self.use_ewc:
			self.creator.create("FitnessMax", self.base.Fitness, weights = (1.0,)) # maximisation
		else:
			self.creator.create("FitnessMax", self.base.Fitness, weights = (-1.0,)) # minimisation
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


	######### This will be our fitness function ##################3
	def eval(self, 
		patch_candidate, 
		init_weight_value, 
		places_to_fix, 
		new_model_name): #, 
		"""
		Evaluate a given trial vector
		Use a given part, a new weight value vector, to update
		a target tensor, here, weight (& bias), and evaluate a new DNN model
		wiht the updated weight tensor

		Argumnets:
			patch_candidate: a trial candiate vector
			sess: None -> save model's parameter values & restore
					is not None -> working on the same session until the search end
		Ret: (fitness)
		"""
		import time
		# generate new weight value from part and init_weight_value
		new_weight_value = self.np.zeros(init_weight_value.shape, dtype = init_weight_value.dtype)
		new_weight_value[:] = init_weight_value

		# replace places_to_fix node with a matching part value
		for i, place_to_fix in enumerate(places_to_fix):
			new_weight_value[place_to_fix] = patch_candidate[i]

		sess, (predictions, correct_predictions, loss_v) = self.move(
			self.tensors['t_weight'], 
			new_weight_value, 
			new_model_name, 
			update_op = 'set') 

		assert predictions is not None
		assert correct_predictions is not None
		num_violated, num_patched = self.get_num_patched_and_broken(
			predictions) 

		if not self.use_ewc:
			losses_of_correct, losses_of_target = loss_v
			if self.patch_aggr is None:
				final_fitness = losses_of_correct + losses_of_target
			else:
				final_fitness = losses_of_correct + self.patch_aggr * losses_of_target	

			num_still_correct = len(self.indices_to_correct) - num_violated # N_intac
			# log
			#print ("New fitness (num_violatd, num_patched)",
			#	self.patch_aggr, losses_of_correct, losses_of_target, num_violated, num_patched, final_fitness)
		else:
			final_fitness = loss_v
			# log
			print ("New fitness (num_violatd, num_patched)", num_violated, num_patched, final_fitness)
			
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


	def set_bounds(self,
		num_places_to_fix,
		init_weight_value):
		
		"""
		"""
		min_v = self.np.min(init_weight_value)
		miv_v = min_v * 2 if min_v < 0 else min_v / 2
		
		max_v = self.np.max(init_weight_value)
		max_v = max_v * 2 if max_v > 0 else max_v / 2

		bounds = [(min_v, max_v)] * num_places_to_fix

		return bounds

	
	def search(self,
		places_to_fix,
		pop_size_times = 5,
		sess = None,
		name_key = 'best', 
		max_allowed_no_change = 10,
		init_weight_value = None,
		var_lambda = None):
		"""
		"""
		import time

		# generate and set empty_graph that is main frame of the our model to patch
		if self.empty_graph is None:
			self.generate_empty_graph()

		# set delta part
		if init_weight_value is None:
			if self.which != 'lfw_vgg':
				kernel_and_bias_pairs = self.apricot_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)
			else:
				kernel_and_bias_pairs = self.torch_rel_util.get_weights(self.path_to_keras_model, start_idx = 0)

			weights = {}
			#for i, (w,b) in enumerate(kernel_and_bias_pairs):
			#	weights["fw{}".format(i+1)] = w
			#	weights["fb{}".format(i+1)] = b	
			weights["fw3"] = kernel_and_bias_pairs[-1][0] 
			weights["fb3"] = kernel_and_bias_pairs[-1][1]
			init_weight_value = weights[self.tensors['t_weight']]

		mean_value = self.np.mean(init_weight_value)
		std_value = self.np.std(init_weight_value)

		min_value = self.np.min(init_weight_value)
		max_value = self.np.max(init_weight_value)
		num_places_to_fix = len(places_to_fix) # NDIM
		
		init_target_ws = init_weight_value[self.np.asarray(places_to_fix)[:,0], self.np.asarray(places_to_fix)[:,1]]

		# set search parameters
		pop_size = 100 
		bounds = self.set_bounds(len(places_to_fix), init_weight_value) # for clipping
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

		if sess is None:
			self.sess = self.model_util.generate_session(graph = self.empty_graph)
		else:
			self.sess = sess

		### For EWC extension ###
		if self.use_ewc and self.ewc_inst is None:
			self.ewc_inst = self.set_ewc_inst(
				init_weight_value, 
				var_lambda = var_lambda)
		#########################

		#print (toolbox.individual()) 
		#print (type(toolbox.individual()))
		pop = toolbox.population(n = pop_size)

		#print ('===pop===')
		#print ("\t Places to fix ({}): {}".format(
		#	num_places_to_fix, 
		#	",".join([str(plc) for plc in places_to_fix ])))
		
		hof = self.tools.HallOfFame(1, similar = self.np.array_equal)

		# update fitness
		for ind in pop:
			ind.fitness.values = toolbox.evaluate(
				ind,
				init_weight_value, 
				places_to_fix, 
				self.model_name) 
			ind.model_name = None 

		hof.update(pop)
		best = hof[0]

		record = stats.compile(pop)
		logbook.record(gen = 0, evals = len(pop), **record)
		print (logbook)

		search_start_time = time.time()
		# search start
		import time
		for iter_idx in range(self.max_search_num):
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
						
				y.fitness.values = toolbox.evaluate(
					y,
					init_weight_value, 
					places_to_fix, 
					new_model_name)
				
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
			new_weight_value = self.np.zeros(
				init_weight_value.shape, 
				dtype = init_weight_value.dtype)

			new_weight_value[:] = init_weight_value
			for i, place_to_fix in enumerate(places_to_fix):
				new_weight_value[place_to_fix] = best[i]
				
			is_early_stop_possible, num_of_patched = self.check_early_stop(
				new_weight_value, 
				best.fitness.values[0], 
				model_name = best.model_name,  
				empty_graph = self.empty_graph)

			print ("Is early stop possible?: {}".format(is_early_stop_possible))
			prev_best = best

			# check for two stop coniditions
			if self.is_the_performance_unchanged(best, 
				places_to_fix):#
				print ("Performance has not been changed over {} iterations".format(
					self.num_iter_unchanged))

				break

			curr_time = time.time()
			run_time = curr_time - search_start_time
			
				
		#save_path = self.model_name_format.format(self.max_search_num, name_key)#"best")
		save_path = self.model_name_format.format(name_key)
		best.model_name = save_path

		# with these two cases, the new model has not been saved
		if self.empty_graph is not None:
			print ("BEST\n", best)
			print ("\tbest fitness:",best.fitness.values[0])
			print (best.model_name, save_path)

			# generate new weight value from part and init_weight_value
			new_weight_value = self.np.zeros(init_weight_value.shape, 
				dtype = init_weight_value.dtype)
			new_weight_value[:] = init_weight_value

			# replace places_to_fix node with a matching part value
			for i, place_to_fix in enumerate(places_to_fix):
				new_weight_value[place_to_fix] = best[i]

			self.move(self.tensors['t_weight'], 
				new_weight_value, 
				self.model_name, 
				update_op = 'set')

			import json 
			with open(save_path.replace("None","model") + ".json", 'w') as f:
				f.write(json.dumps({'weight':new_weight_value.tolist()}))	
			
			# update self.curr_feed_dict (for processing wrongly classified inputs & outputs)
			target_tensor = self.model_util.get_tensor(
				self.tensors['t_weight'], 
				self.empty_graph)

			assert target_tensor in self.curr_feed_dict.keys(), "Target tensor %s should already exists" % (target_tensor_name)

			self.curr_feed_dict[target_tensor] = new_weight_value

			self.summarise_results(save_path)
		
		if self.sess is not None:
			self.sess.close()

		return best.model_name



