"""
train GTSRB model
"""
import os, sys
import argparse
import pickle
import numpy as np
import tensorflow as tf
import utils.data_util as data_util

def gen_and_train_model(train_X, train_y, test_X, test_y, 
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

	mdl = tf.keras.models.Sequential()
	input_shape = (48, 48, 3)

	# 7x7 or 3x3, 100
	mdl.add(tf.keras.layers.Conv2D(100, (3,3), 
		activation='tanh', padding = 'valid', input_shape = input_shape))
	mdl.add(tf.keras.layers.BatchNormalization(axis = 1))
	mdl.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
	#mdl.add(tf.keras.layers.Dropout(0.2))

	# 4x4 or 4x4, 150
	mdl.add(tf.keras.layers.Conv2D(150, (4,4), activation='tanh', padding = 'valid'))
	mdl.add(tf.keras.layers.BatchNormalization(axis = -1))
	mdl.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
	#mdl.add(tf.keras.layers.Dropout(0.2))

	# 4x4 or 3x3, 250
	mdl.add(tf.keras.layers.Conv2D(250, (3,3), activation='tanh', padding = 'valid'))
	mdl.add(tf.keras.layers.BatchNormalization(axis = -1))
	mdl.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2))) 
	#mdl.add(tf.keras.layers.Dropout(0.2))

	# 300 or 200, fully
	mdl.add(tf.keras.layers.Flatten())
	mdl.add(tf.keras.layers.Dense(200, activation = 'tanh'))
	mdl.add(tf.keras.layers.BatchNormalization())
	#mdl.add(tf.keras.layers.Dropout(0.5))

	# 43, fully
	mdl.add(tf.keras.layers.Dense(43, activation = 'softmax'))

	## or keras.optimizers import SGD
	optimizer = tf.keras.optimizers.Adam(lr = 0.001)
	mdl.compile(loss='categorical_crossentropy', 
		optimizer = optimizer, 
		metrics = ['accuracy', 'categorical_crossentropy'])

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
			min_delta = 0, 
			# "no longer improving" being further defined as "for at least 100 epochs"
			patience = patience, 
			restore_best_weights = True, # restore the weights to the best ones obtained during the patience 
			verbose = 1)]

	# or batch_size = 128
	mdl.fit(train_X, train_y, 
		epochs = num_epoch, 
		batch_size = batch_size, 
		callbacks = callbacks, 
		verbose = 2,
		validation_data = (test_X, test_y))

	return mdl

parser = argparse.ArgumentParser()
parser.add_argument("-datadir", type = str, default = 'data/GTSRB/')
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-mdlkey", type = str, default = 'gtsrb.cnn')

args = parser.parse_args()

train_data, test_data = data_util.load_data('GTSRB', args.datadir)

mdl = gen_and_train_model(train_data[0], train_data[1], test_data[0], test_data[1], 
	num_epoch = 5000, patience = 100, batch_size = 64)

results = mdl.evaluate(test_data[0], test_data[1], batch_size = 128)
print("test result", results)

#print("Generate predictions for 3 samples")
#predictions = model.predict(x_test[:3])
#print("predictions shape:", predictions.shape)

# save
destfile = os.path.join(args.dest, "gtsrb.model.{}.h5".format(args.mdlkey))
mdl.save('destfile')