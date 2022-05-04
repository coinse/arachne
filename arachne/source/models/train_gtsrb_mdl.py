"""
train GTSRB model
"""
import os, sys
import argparse
import pickle
import numpy as np
import tensorflow as tf
import utils.data_util as data_util
import utils.gen_frame_graph as gen_frame_graph

def gen_and_train_model(train_X, train_y, test_X, test_y, destfile, 
	num_epoch = 5000, patience = 100, batch_size = 64):
	"""
	Implement a simple feed-forward CNN model with seven layers used in D. Cireşan et al., ijcnn'11"A Committee of Neural Networks for Traffic Sign Classification"
	The original approach distorts images on the fly with four additional image preprocessing, increasing the total number of inputs
	by five-folds. However, here, we only uses the original inputs since our target is not achieving the highest performance. 

	For the early stopping. we follow the strategy used in this paper: "only the original images are used for validation. 
	Training ends once the validation error is zero (usually after 10 to 50 epochs)."

	acheive arond 97% accuracy for the test data
	"""
	act = 'relu'
	
	# input layer
	input_shape = (3, 48, 48) #(48, 48, 3)
	inputs = tf.keras.Input(shape = input_shape)
	
	# 7x7 or 3x3, 100
	outputs = tf.keras.layers.Conv2D(100, (3,3), activation=act, padding = 'valid', data_format = 'channels_first')(inputs)
	outputs = tf.keras.layers.BatchNormalization(axis = 1)(outputs)
	outputs = tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')(outputs) 	

	# 4x4 or 4x4, 150
	outputs = tf.keras.layers.Conv2D(150, (4,4), activation=act, padding = 'valid', data_format = 'channels_first')(outputs)
	outputs = tf.keras.layers.BatchNormalization(axis = 1)(outputs)
	outputs = tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')(outputs)

	# 4x4 or 3x3, 250
	outputs = tf.keras.layers.Conv2D(250, (3,3), activation=act, padding = 'valid', data_format = 'channels_first')(outputs)
	outputs = tf.keras.layers.BatchNormalization(axis = 1)(outputs)
	outputs = tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')(outputs)

	# 300 or 200, fully
	outputs = tf.keras.layers.Flatten()(outputs)
	outputs = tf.keras.layers.Dense(200, activation = act)(outputs)
	outputs = tf.keras.layers.BatchNormalization()(outputs)

	# 43, fully
	outputs = tf.keras.layers.Dense(43, activation = 'softmax')(outputs)
	
	mdl = tf.keras.models.Model(inputs = inputs, outputs = outputs)
	
	## or keras.optimizers import SGD
	optimizer = tf.keras.optimizers.Adam(lr = 0.001)
	mdl.compile(loss='categorical_crossentropy', 
		optimizer = optimizer, 
		metrics = ['accuracy'])

	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			# Stop training when `val_acc` is no longer improving
			monitor = "val_acc",
			mode = 'max', 
			min_delta = 0, 
			# "no longer improving" being further defined as "for at least 100 epochs"
			patience = patience, 
			verbose = 1), 
		tf.keras.callbacks.ModelCheckpoint(
			destfile,
			monitor = 'val_acc', 
			mode = 'max', 
			verbose = 1, 
			save_best_only = True)
		]	

	# or batch_size = 128
	mdl.fit(train_X, train_y, 
		epochs = num_epoch, 
		batch_size = batch_size, 
		callbacks = callbacks, 
		verbose = 1,
		validation_data = (test_X, test_y))



def gen_and_train_simple_model(train_X, train_y, test_X, test_y, destfile, 
	num_epoch = 5000, patience = 100, batch_size = 64):
	"""
	a simpler version of ijcnn'11 model for RQ1 ~ RQ4. acheive arond 90% accuracy for the test data
	"""
	act = 'relu'
	
	# input layer
	input_shape = (3, 48, 48) #(48, 48, 3)
	inputs = tf.keras.Input(shape = input_shape)

	outputs = tf.keras.layers.Conv2D(16, (3,3), activation=act, padding = 'valid', data_format = 'channels_first')(inputs)
	outputs = tf.keras.layers.BatchNormalization(axis = 1)(outputs)
	outputs = tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')(outputs) 	

	outputs = tf.keras.layers.Flatten()(outputs)
	outputs = tf.keras.layers.Dense(512, activation = act)(outputs)
	outputs = tf.keras.layers.BatchNormalization()(outputs)

	outputs = tf.keras.layers.Dense(43, activation = 'softmax')(outputs)
	
	mdl = tf.keras.models.Model(inputs = inputs, outputs = outputs)
	
	## or keras.optimizers import SGD
	optimizer = tf.keras.optimizers.Adam(lr = 0.001)
	mdl.compile(loss='categorical_crossentropy', 
		optimizer = optimizer, 
		metrics = ['accuracy'])

	callbacks = [
		tf.keras.callbacks.EarlyStopping(
			# Stop training when `val_acc` is no longer improving
			monitor = "val_acc",
			mode = 'max', 
			min_delta = 0, 
			# "no longer improving" being further defined as "for at least 100 epochs"
			patience = patience, 
			verbose = 1), 
		tf.keras.callbacks.ModelCheckpoint(
			destfile,
			monitor = 'val_acc', 
			mode = 'max', 
			verbose = 1, 
			save_best_only = True)
		]	

	# or batch_size = 128
	mdl.fit(train_X, train_y, 
		epochs = num_epoch, 
		batch_size = batch_size, 
		callbacks = callbacks, 
		verbose = 1,
		validation_data = (test_X, test_y))


parser = argparse.ArgumentParser()
parser.add_argument("-datadir", type = str, default = 'data/GTSRB/')
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-mdlkey", type = str, default = 'gtsrb.cnn')
parser.add_argument("-w_hist", type= int, default = 0)
parser.add_argument("-simple", action = "store_true", help = "if given, train a simple model")

args = parser.parse_args()

train_data, test_data = data_util.load_data('GTSRB', args.datadir, with_hist = bool(args.w_hist))

indices = np.arange(len(train_data[0]))
np.random.shuffle(indices)

train_data[0] = train_data[0][indices]
train_data[1] = train_data[1][indices]

os.makedirs(args.dest, exist_ok = True)
destfile = os.path.join(args.dest, "gtsrb.model.{}.wh.{}.h5".format(args.mdlkey, args.w_hist))

if args.simple:
	gen_and_train_simple_model(train_data[0], data_util.format_label(train_data[1], 43), 
		test_data[0], data_util.format_label(test_data[1], 43), 
		destfile, 
		num_epoch = 5000, patience = 100, batch_size = 64)
else:
	gen_and_train_model(train_data[0], data_util.format_label(train_data[1], 43),
		test_data[0], data_util.format_label(test_data[1], 43),
		num_epoch = 5000, patience = 100, batch_size = 64)

saved_model = tf.keras.models.load_model(destfile)

# evaluate the model
_, train_acc = saved_model.evaluate(train_data[0], data_util.format_label(train_data[1], 43), verbose=0)
_, test_acc = saved_model.evaluate(test_data[0], data_util.format_label(test_data[1], 43), verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
