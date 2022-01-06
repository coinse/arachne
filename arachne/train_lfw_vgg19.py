"""
train GTSRB model
"""
import os, sys
import argparse
import numpy as np
import tensorflow as tf
import utils.data_util as data_util
import tensorflow as tf 
from tensorflow.keras.models import load_model, Model

def get_vgg19_mdl_from_scratch(num_classes = 2, input_shape = (32,32,3)):
	import tensorflow.keras.layers as layers

	img_input = tf.keras.Input(shape = input_shape)
	# Block 1
	x = layers.Conv2D(
			64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = layers.Conv2D(
			64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = layers.Conv2D(
			128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = layers.Conv2D(
			128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	
	x = tf.keras.layers.Flatten(name='flatten')(x)
	x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
	x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
	x = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
	
	mdl = Model(img_input, x, name = 'vgg19_lfw')
	return mdl 


def train(mdl, train_X, train_y, test_X, test_y, destfile, 
	num_epoch = 5000, patience = 100, batch_size = 64):
	"""
	"""
	## or keras.optimizers import SGD
	optimizer = tf.keras.optimizers.Adam(lr = 2e-4)
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
			save_weights_only =True, 
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
parser.add_argument("-datadir", type = str, default = 'data/lfw_data/')
parser.add_argument("-dest", type = str, default = ".")

args = parser.parse_args()

which = 'lfw_vgg'
which_data = 'lfw'
path_to_female_names = 'data/lfw_np/female_names_lfw.txt'
train_data, test_data = data_util.load_data(which_data, args.datadir, 
	path_to_female_names = path_to_female_names)

indices = np.arange(len(train_data[0]))
np.random.shuffle(indices)
train_data[0] = train_data[0][indices]
train_data[1] = train_data[1][indices]

print (train_data[0].shape)
os.makedirs(args.dest, exist_ok = True)
destfile = os.path.join(args.dest, "LFW_gender_classifier.h5")

mdl = get_vgg19_mdl_from_scratch(num_classes = 2, input_shape = (32,32,3))
train(mdl, train_data[0], data_util.format_label(train_data[1],2), 
	test_data[0], data_util.format_label(test_data[1],2), destfile, 
	num_epoch = 5000, patience = 100, batch_size = 64)

saved_model = tf.keras.models.load_model(destfile)

# evaluate the model
_, train_acc = saved_model.evaluate(train_data[0], data_util.format_label(train_data[1], 2), verbose=0)
_, test_acc = saved_model.evaluate(test_data[0], data_util.format_label(test_data[1], 2), verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

checkpoint_path = os.path.join(args.dest, "cp.best.ckpt") 
mdl.load_weights(checkpoint_path)
best_mdl_destfile = os.path.join(args.dest, "LFW_gender_classifier_best.h5")
tf.keras.models.save_model(mdl, best_mdl_destfile)
