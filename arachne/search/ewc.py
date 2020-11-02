"""
Elastic weight consoldiation
"""

class EWC_Loss(object):
	"""docstring for EWC_Loss"""

	np = __import__('numpy')
	importlib = __import__('importlib')
	model_util = importlib.import_module('utils.model_util')

	def __init__(self, 
		init_weight, 
		predictions, 
		labels,
		indices_to_taska, 
		indices_to_taskb,
		empty_graph,
		curr_plchldr_feed_dict,
		sess,
		weight_tensor_name = 'fw3',
		var_lambda = 1,
		mode = 0):

		super(EWC_Loss, self).__init__()
		self.init_weight = init_weight
		self.predictions = predictions
		self.num_imgs = len(predictions)
		self.labels = labels
		#self.imgset = imgset
		self.indices_to_taska = indices_to_taska
		self.indices_to_taskb = indices_to_taskb

		self.empty_graph = empty_graph
		self.curr_plchldr_feed_dict = curr_plchldr_feed_dict

		self.weight_tensor_name = weight_tensor_name
		self.var_lambda = var_lambda
		# set self.fisher
		self.compute_fisher_emp(self.empty_graph, 
			self.curr_plchldr_feed_dict, 
			sess, 
			mode = mode)
		assert self.fisher is not None, "fisher should be computed"

#	def compute_fisher(self, predictions):
#		# computer Fisher information for each parameter
#		# initialize Fisher information for most recent task
#		num_inputs = len(imgset)
#		fisher = self.np.zeros(w.shape)
#		
#		# sampling a random class from softmax
#		probs = tf.nn.softmax(self.y)
#		class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])#
#
#		class_ind = np.multinomial(np.log(probs),1)
#		
#		for i in range(num_samples):
#			# select random input image
#			im_ind = self.np.random.randint(imgset.shape[0])
#			# compute first-order derivatives
#			ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
#			# square the derivatives and add to total
#			for v in range(len(self.fisher)):
#				self.fisher[v] += self.np.square(ders[v])	#
#
#		fisher /= num_inputs
#		return fisher	
	
	def compute_fisher_emp(self, 
		empty_graph, 
		_curr_plchldr_feed_dict, 
		sess, 
		mode = 0):
		"""
		"""
		import tensorflow as tf

		indices = list(range(self.num_imgs))
		#
		#for op in sess.graph.get_operations():
		#	print (op.name)

		curr_plchldr_feed_dict = _curr_plchldr_feed_dict.copy()

		if mode == 0:
			vs, _ = self.model_util.run(
				['fisher_mat'],
				None, 
				None, 
				input_tensor_name = None, 
				sess = sess, 
				empty_graph = empty_graph,
				plchldr_feed_dict = curr_plchldr_feed_dict)	

			self.fisher = vs[0]
		else:# OR
			predc_tensor = empty_graph.get_tensor_by_name("predc:0")
			weight_tensor = empty_graph.get_tensor_by_name('{}:0'.format(self.weight_tensor_name))	

			try:
				iter(self.labels[0])
				indices_to_label = self.np.argmax(self.labels, axis = 1)
			except TypeError:
				indices_to_label = self.labels

			full_indices = self.np.asarray(list(zip(self.np.arange(len(self.labels)), indices_to_label)))
			ders_tensor = tf.gradients(tf.reshape(tf.log(tf.gather_nd(predc_tensor, full_indices)), (-1,1)), weight_tensor)
			vs, _ = self.model_util.run(
				ders_tensor,
				None, 
				None, 
				input_tensor_name = None, 
				output_tensor_name = None,
				sess = sess, 
				empty_graph = empty_graph,
				plchldr_feed_dict = curr_plchldr_feed_dict)	
			
			ders_arr = 1/len(self.labels) * vs[0]			
			print ("Ders", ders_arr.shape, len(self.labels))	
			self.fisher = ders_arr ** 2


	def ewc_loss(self,
		new_weight, 
		per_label_losses):
		"""
		per_label_losses => will be given as this is called insider of searcher.move
		"""	
		assert per_label_losses is not None
		loss_taskb = self.np.mean(per_label_losses[self.indices_to_taskb])
		post_taska = self.var_lambda/2 * self.np.sum(
			self.np.multiply(self.fisher, (new_weight - self.init_weight)**2))
		
		print ("({}), Loss A and B: {} and {}".format(self.var_lambda, post_taska, loss_taskb))
		loss = loss_taskb + post_taska

		return loss



