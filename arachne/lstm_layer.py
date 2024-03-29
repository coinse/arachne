import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np 

class LSTM_Layer(object):
	def __init__(self, init_lstm_layer):
		"""
		"""
		self.init_lstm_layer = init_lstm_layer
		self.num_units = init_lstm_layer.units


	@staticmethod
	def get_initial_state(lstm_layer):
		"""
		rerturn the initial state values of a given lstm layer
		"""
		init_hidden_states, init_cell_states = lstm_layer.states
		init_hidden_states = np.asarray(init_hidden_states) if init_hidden_states is not None else None
		init_cell_states = np.asarray(init_cell_states) if init_cell_states is not None else None
		return init_hidden_states, init_cell_states


	def gen_lstm_layer_from_another(self, prev_output):
		"""
		generate a model with a single lstm_layer that returns 
		the sequences of hidden state and cell state 
		based on the exisitng one (self.init_lstm_layer)
		"""
		from tensorflow.keras.layers import LSTM, CuDNNLSTM
		#input_shape = self.init_lstm_layer.input_shape
		time_steps = prev_output.shape[1]
		#print ("time step is {}".format(time_steps))
		assert time_steps is not None, "For this time_steps should be given"
		kernel_w, recurr_kernel_w, bias = self.init_lstm_layer.get_weights()
		# change to accept variout shapes of inputs -> both all & per time-step
		#print ("Input shape", input_shape)
		inputs = tf.keras.Input(shape = (None, None)) 
		h_state_input = tf.keras.Input(shape = (None,))
		c_state_input = tf.keras.Input(shape = (None,))
		#new_lstm = LSTM(
		new_lstm = CuDNNLSTM(
			self.init_lstm_layer.units, 
			kernel_initializer=tf.constant_initializer(kernel_w),
			recurrent_initializer=tf.constant_initializer(recurr_kernel_w),
			bias_initializer=tf.constant_initializer(bias),
			return_sequences = False, 
			return_state=True)

		skip = ['units', 
			'kernel_initializer', 'recurrent_initializer', 'bias_initializer', 
			'input_shape', 'return_state']
		for k,v in self.init_lstm_layer.__dict__.items():
			if k not in skip:
				new_lstm.__dict__.update({k:v})
	
		outs = new_lstm(inputs, initial_state = [h_state_input, c_state_input])
		outs_2 = new_lstm(inputs)

		mdl = Model(inputs = [inputs, h_state_input, c_state_input], outputs = outs)
		mdl2 = Model(inputs = inputs, outputs =outs_2)

		mdl.summary()
		mdl2.summary()

		h_states = []; cell_states = []
		print ("prev", prev_output.shape)
		print (mdl2.summary())
		for t in range(time_steps):
			curr_prev_output = prev_output[:,t:t+1,:]
			if len(mdl2.inputs[0].shape) < len(curr_prev_output.shape):
				curr_prev_output = np.squeeze(curr_prev_output, axis = 1)

			if t == 0:
				_, h_state, cell_state = mdl2.predict(curr_prev_output)
			else:
				_, h_state, cell_state = mdl.predict([curr_prev_output, h_state, cell_state])
			h_states.append(h_state) 
			cell_states.append(cell_state) 

		# shape = (time_steps, batch_size, num_units)
		h_states = np.asarray(h_states) 
		# shape = (batch_size, time_steps, num_units)
		h_states = np.moveaxis(h_states, [0,1], [1,0])
		
		# shape = (time_steps, batch_size, num_units)
		cell_states = np.asarray(cell_states) 
		# shape = (batch_size, time_steps, num_units)
		cell_states = np.moveaxis(cell_states, [0,1], [1,0]) 

		return h_states, cell_states

