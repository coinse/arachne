from os import stat
from tensorflow.keras.layers import LSTM
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


	def gen_lstm_layer_from_another(self):
		"""
		generate a model with a single lstm_layer that returns the sequences of hidden state and cell state 
		based on the exisitng one (self.init_lstm_layer)
		"""
		input_shape = self.init_lstm_layer.input_shape
		time_steps = input_shape[1]
		kernel_w, recurr_kernel_w, bias = self.init_lstm_layer.get_weights()

		inputs = tf.keras.Input(shape = input_shape)

		new_lstm = LSTM(self.init_lstm_layer.units, 
			kernel_initializer=tf.constant_initializer(kernel_w),
			recurrent_initializer=tf.constant_initializer(recurr_kernel_w),
			bias_initializer=tf.constant_initializer(bias),
			#return_sequences=True, 
			return_state=True, 
			input_shape = input_shape)

		#skip = ['units', 'kernel_initializer', 'recurrent_initializer', 'bias_initializer', 'return_sequences', 'return_state', 'input_shape']
		skip = ['units', 'kernel_initializer', 'recurrent_initializer', 'bias_initializer', 'input_shape']
		for k,v in self.init_lstm_layer.__dict__.items():
			if k not in skip:
				new_lstm.__dict__.update({k:v})

		#out = new_lstm(inputs)
		h_states = []; cell_states = []
		h_state = None
		for t in range(time_steps):
			# h_state = (batch_size, num_units), cell_state = (batch_size, num_units)
			_, h_state, cell_state = new_lstm(inputs[:,t,:], initial_state = h_state)
			h_states.append(h_state) 
			cell_states.append(cell_state) 

		h_states = np.asarray(h_states) # shape = (time_steps, batch_size, num_units)
		h_states = np.moveaxis(h_states, [0,1], [1,0]) # shape = (batch_size, time_steps, num_units)
		
		cell_states = np.asarray(cell_states) # shape = (time_steps, batch_size, num_units)
		cell_states = np.moveaxis(cell_states, [0,1], [1,0]) # shape = (batch_size, time_steps, num_units)

		mdl = Model(inputs = inputs, outputs = [h_states, cell_states])
		return mdl, new_lstm

