from os import stat
from tensorflow.keras.layers import LSTM, Lambda, Concatenate
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


	def gen_lstm_layer_from_another(self, prev_output_shape):
		"""
		generate a model with a single lstm_layer that returns the sequences of hidden state and cell state 
		based on the exisitng one (self.init_lstm_layer)
		"""
		#input_shape = self.init_lstm_layer.input_shape
		input_shape = prev_output_shape[1:]
		time_steps = prev_output_shape[1]
		assert time_steps is not None, "For this time_steps should be given"
		kernel_w, recurr_kernel_w, bias = self.init_lstm_layer.get_weights()
		
		inputs = tf.keras.Input(shape = input_shape)
		print (inputs)
		print ("init unit", self.init_lstm_layer.units)
		new_lstm = LSTM(self.init_lstm_layer.units, 
			kernel_initializer=tf.constant_initializer(kernel_w),
			recurrent_initializer=tf.constant_initializer(recurr_kernel_w),
			bias_initializer=tf.constant_initializer(bias),
			return_state=True)#, 
			#input_shape = input_shape)
		print (new_lstm)	
		skip = ['units', 'kernel_initializer', 'recurrent_initializer', 'bias_initializer', 'input_shape', 'return_state']
		for k,v in self.init_lstm_layer.__dict__.items():
			if k not in skip:
				new_lstm.__dict__.update({k:v})

		outs =  []	
		h_states = []; cell_states = []
		h_state = None; cell_state = None
		for t in range(2):#time_steps):
			print ("**", t)
			# h_state = (batch_size, num_units), cell_state = (batch_size, num_units)
			if t == 0:
				#_, h_state, cell_state = new_lstm(inputs[:,t:t+1,:])
				out = new_lstm(inputs[:,t:t+1,:])
			else:
				#_, h_state, cell_state = new_lstm(inputs[:,t:t+1,:], initial_state = [h_state, cell_state])
				out = new_lstm(inputs[:,t:t+1,:], initial_state = out[1:])#[h_state, cell_state])
			#h_states.append(h_state) 
			#cell_states.append(cell_state) 
			outs.append(out[0])
		#
		lambda_fn = Lambda(lambda vs:tf.stack(vs, axis = 0))
		outs = lambda_fn(outs)
		#cell_states = lambda_fn(cell_states)
		#h_states = tf.stack(h_states, axis = 0)
		#print (h_states)
		#h_states = np.moveaxis(h_states, [0,1], [1,0]) # shape = (batch_size, time_steps, num_units)
		#h_states = tf.transpose(h_states, perm = [1,0,2])
		#print ("after", h_states)
		#cell_states = np.asarray(cell_states, dtype =object) # shape = (time_steps, batch_size, num_units)
		#cell_states = np.moveaxis(cell_states, [0,1], [1,0]) # shape = (batch_size, time_steps, num_units)
		#cell_states = tf.stack(cell_states, axis = 0)
		#cell_states = tf.transpose(cell_states, [1,0,2])
		#return hstates, cell_states
		mdl = Model(inputs = inputs, outputs = outs)#[h_states, cell_states])
		mdl.summary()
		return mdl

