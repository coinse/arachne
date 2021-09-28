"""
train GTSRB model
"""
import os, sys
import argparse
import pickle
import numpy as np
import tensorflow as tf
import utils.data_util as data_util

def gen_and_train_model(train_X, train_y, test_X, test_y, destfile, 
	num_epoch = 5000, patience = 100, batch_size = 64):
	"""
	Implement a simple feed-forward CNN model with seven layers used in "A Committee of Neural Networks for Traffic Sign Classification"
	The original approach distors images on the fly with four additional image preprocessing, increasing the total number of inputs
	by five-folds. However, here, we only uses the original inputs since our target is not achieving the highest performance. 

	For the early stopping. we follow the strategy used in the paper: "only the original images are used for validation. 
	Training ends once the validation error is zero (usually after 10 to 50 epochs)." 
	"""
	# model = tf.keras.models.Sequential()
	# input_shape = (32, 32, 1) # grey-scale images of 32x32
	# model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', 
	#             activation='relu', input_shape=input_shape))
	# model.add(tf.keras.layers.BatchNormalization(axis=-1))      
	# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
	# model.add(tf.keras.layers.Dropout(0.2))

	# model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', 
	#                                 activation='relu'))
	# model.add(tf.keras.layers.BatchNormalization(axis=-1))

	# model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', 
	#                                 activation='relu'))
	# model.add(tf.keras.layers.BatchNormalization(axis=-1))
	# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	# model.add(tf.keras.layers.Dropout(0.2))
	# model.add(tf.keras.layers.Flatten())
	# model.add(tf.keras.layers.Dense(512, activation='relu'))
	# model.add(tf.keras.layers.BatchNormalization())

	# model.add(tf.keras.layers.Dropout(0.4))
	# model.add(tf.keras.layers.Dense(43, activation='softmax'))

	#mdl = tf.keras.models.Sequential()
	act = 'relu'
	
	# input layer
	input_shape = (3, 48, 48) #(48, 48, 3)
	#inputs = tf.keras.layers.InputLayer(input_shape = input_shape)
	inputs = tf.keras.Input(shape = input_shape)
	#mdl.add(Input)
	
	# 7x7 or 3x3, 100
	#mdl.add(tf.keras.layers.Conv2D(100, (3,3), activation=act, padding = 'valid', data_format = 'channels_first', input_shape = input_shape))
	#mdl.add(tf.keras.layers.BatchNormalization(axis = 1)) # 1 when channels_last, -1 and channels first then 1
	#mdl.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first'))
	#
	outputs = tf.keras.layers.Conv2D(100, (3,3), activation=act, padding = 'valid', data_format = 'channels_first')(inputs)
	outputs = tf.keras.layers.BatchNormalization(axis = 1)(outputs)
	outputs = tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')(outputs) 	

	#mdl.add(tf.keras.layers.Dropout(0.2))

	# 4x4 or 4x4, 150
	#mdl.add(tf.keras.layers.Conv2D(150, (4,4), activation=act, padding = 'valid', data_format = 'channels_first'))
	#mdl.add(tf.keras.layers.BatchNormalization(axis = 1))
	#mdl.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first'))
	#mdl.add(tf.keras.layers.Dropout(0.2))
	#
	outputs = tf.keras.layers.Conv2D(150, (4,4), activation=act, padding = 'valid', data_format = 'channels_first')(outputs)
	outputs = tf.keras.layers.BatchNormalization(axis = 1)(outputs)
	outputs = tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')(outputs)

	# 4x4 or 3x3, 250
	#mdl.add(tf.keras.layers.Conv2D(250, (3,3), activation=act, padding = 'valid', data_format = 'channels_first'))
	#mdl.add(tf.keras.layers.BatchNormalization(axis = 1))
	#mdl.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')) 
	#mdl.add(tf.keras.layers.Dropout(0.2))
	#
	outputs = tf.keras.layers.Conv2D(250, (3,3), activation=act, padding = 'valid', data_format = 'channels_first')(outputs)
	outputs = tf.keras.layers.BatchNormalization(axis = 1)(outputs)
	outputs = tf.keras.layers.MaxPooling2D(pool_size = (2,2), data_format = 'channels_first')(outputs)

	# 300 or 200, fully
	#mdl.add(tf.keras.layers.Flatten())
	#mdl.add(tf.keras.layers.Dense(200, activation = act))
	#mdl.add(tf.keras.layers.BatchNormalization())
	#mdl.add(tf.keras.layers.Dropout(0.5))
	#
	outputs = tf.keras.layers.Flatten()(outputs)
	outputs = tf.keras.layers.Dense(200, activation = act)(outputs)
	outputs = tf.keras.layers.BatchNormalization()(outputs)

	# 43, fully
	#mdl.add(tf.keras.layers.Dense(43, activation = 'softmax'))
	#
	outputs = tf.keras.layers.Dense(43, activation = 'softmax')(outputs)
	
	mdl = tf.keras.models.Model(inputs = inputs, outputs = outputs)
	
	## or keras.optimizers import SGD
	optimizer = tf.keras.optimizers.Adam(lr = 0.001)
	mdl.compile(loss='categorical_crossentropy', 
		optimizer = optimizer, 
		metrics = ['accuracy'])

	# callbacks = [
	# 	tf.keras.callbacks.EarlyStopping(
	# 		# Stop training when `val_loss` is no longer improving
	# 		monitor = "val_loss",
	# 		# "no longer improving" being defined as "no better than 1e-2 less"
	# 		min_delta=1e-2,
	# 		# "no longer improving" being further defined as "for at least 2 epochs"
	# 		patience=10, 
	# 		#restore_best_weights = True, # restore to the best weights during the patience 
	# 		verbose=1)]
		
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

	#return mdl



def gen_and_train_simple_model(train_X, train_y, test_X, test_y, destfile, 
	num_epoch = 5000, patience = 100, batch_size = 64):
	"""
	simple 
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

args = parser.parse_args()

train_data, test_data = data_util.load_data('GTSRB', args.datadir, with_hist = bool(args.w_hist))

sys.exit()
indices = np.arange(len(train_data[0]))
np.random.shuffle(indices)

train_data[0] = train_data[0][indices]
train_data[1] = train_data[1][indices]

print (train_data[0].shape)
os.makedirs(args.dest, exist_ok = True)
destfile = os.path.join(args.dest, "gtsrb.model.{}.wh.{}.h5".format(args.mdlkey, args.w_hist))

gen_and_train_simple_model(train_data[0], data_util.format_label(train_data[1], 43), 
	test_data[0], data_util.format_label(test_data[1], 43), 
	destfile, 
	num_epoch = 5000, patience = 100, batch_size = 64)

#mdl.save('test.h5')
#_, train_acc = mdl.evaluate(train_data[0], train_data[1], verbose=0)
#_, test_acc = mdl.evaluate(test_data[0], test_data[1], verbose=0)
#print('Returned: Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#print("Generate predictions for 3 samples")
#predictions = model.predict(x_test[:3])
#print("predictions shape:", predictions.shape)

saved_model = tf.keras.models.load_model(destfile)


# evaluate the model
_, train_acc = saved_model.evaluate(train_data[0], data_util.format_label(train_data[1], 43), verbose=0)
_, test_acc = saved_model.evaluate(test_data[0], data_util.format_label(test_data[1], 43), verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
