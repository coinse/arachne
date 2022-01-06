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


def get_vgg19_from_trained(num_classes = 2, input_shape = (32,32,3), dropout_rate = 0.5):
	"""
	"""
	import tensorflow.keras.layers as layers

	img_input = tf.keras.Input(shape = input_shape)
	vgg19_mdl_front = tf.keras.applications.vgg19.VGG19(
		include_top = False,
		weights = 'imagenet',
		input_shape = input_shape,
		classes = num_classes)
	
	out = vgg19_mdl_front(img_input)

	# classifier part
	out = layers.Flatten(name='flatten')(out)
	out = layers.Dense(4096, activation='relu', name='fc1')(out)
	if dropout_rate > 0: out = layers.Dropout(dropout_rate)
	out = layers.Dense(4096, activation='relu', name='fc2')(out)
	if dropout_rate > 0: out = layers.Dropout(dropout_rate)
	out = layers.Dense(num_classes, activation='softmax', name='predictions')(out)

	mdl = Model(inputs = img_input, outputs = out, name = 'vgg19_lfw')
	return mdl


def get_vgg19_mdl_from_scratch_v2(num_classes = 2, input_shape = (32,32,3), batch_norm = False, dropout_rate = 0.5, data_format = 'channels_last'):
	"""
	Channel first -> (NCHW)
	& batch normalization & valid padding & dropout (!!!)
	Convert the implementation of torchvision.models.vgg19 into the keras-based one

	data_format: either channels_last or channels_firt
	"""
	import tensorflow.keras.layers as layers

	img_input = tf.keras.Input(shape = input_shape)

	# Block 1
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block1_padding1')(img_input)
	x = layers.Conv2D(
			64, (3, 3), activation='relu', padding = 'same', name='block1_conv1', data_format = data_format)(img_input)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block1_padding2')(x)
	x = layers.Conv2D(
			64, (3, 3), activation='relu', padding = 'same', name='block1_conv2', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block2_padding1')(x)
	x = layers.Conv2D(
			128, (3, 3), activation='relu', padding = 'same', name='block2_conv1', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block2_padding2')(x)
	x = layers.Conv2D(
			128, (3, 3), activation='relu', padding = 'same', name='block2_conv2', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block3_padding1')(x)
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding = 'same', name='block3_conv1', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block3_padding2')(x)
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding = 'same', name='block3_conv2', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block3_padding3')(x)
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding = 'same', name='block3_conv3', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block3_padding4')(x)
	x = layers.Conv2D(
			256, (3, 3), activation='relu', padding = 'same', name='block3_conv4', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block4_padding1')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block4_conv1', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block4_padding2')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block4_conv2', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block4_padding3')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block4_conv3', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block4_padding4')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block4_conv4', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block5_padding1')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block5_conv1', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block5_padding2')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block5_conv2', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block5_padding3')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block5_conv3', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	#x = layers.ZeroPadding2D(padding=(1,1), data_format = data_format, name='block5_padding4')(x)
	x = layers.Conv2D(
			512, (3, 3), activation='relu', padding = 'same', name='block5_conv4', data_format = data_format)(x)
	if batch_norm: x = layers.BatchNormalization(axis = 1)(x)
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	
	### here we skip adaptive average pooling as don't need it ###

	x = layers.Flatten(name='flatten')(x)
	x = layers.Dense(4096, activation='relu', name='fc1')(x)
	if dropout_rate > 0: x = layers.Dropout(dropout_rate)
	x = layers.Dense(4096, activation='relu', name='fc2')(x)
	if dropout_rate > 0: x = layers.Dropout(dropout_rate)
	x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
	
	mdl = Model(img_input, x, name = 'vgg19_lfw')
	return mdl 


def train(mdl, train_X, train_y, test_X, test_y, destfile, 
	num_epoch = 500, patience = 100, batch_size = 64):
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
			#save_weights_only =True, 
			save_best_only = True)
		]	

	# or batch_size = 128
	mdl.fit(train_X, train_y, 
		epochs = num_epoch, 
		batch_size = batch_size, 
		callbacks = callbacks, 
		verbose = 1,
		validation_data = (test_X, test_y))


def get_weights_of_transfered_vgg19(source_vgg_mdl):
	"""
	first layer with weights -> vgg19 with cnn
	last 
	"""
	cnt = 0; weights = []
	for layer in source_vgg_mdl.layers:
		ws = layer.get_weights()
		if len(ws) > 0: # has weights
			cnt += 1
			if cnt == 1: # vgg part
				num_weights = len(ws)
				for idx in range(0,num_weights,2):
					curr_ws = ws[idx:idx+2]
					weights.append(curr_ws)
			else:
				weights.append(ws)		
	return weights


def update_weights_of_frame_model(frame_mdl, weights):
	"""
	"""
	cnt = 0
	for layer in frame_mdl.layers:
		if len(layer.get_weights()) > 0:
			layer.set_weights(weights[cnt])
			cnt += 1
	return frame_mdl


parser = argparse.ArgumentParser()
parser.add_argument("-datadir", type = str, default = 'data/lfw_data/')
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-version", type = int, default = 0,
	help = "0: copied from keras implementation\
			1: from the torch implementation\
			2: from keras implementation, but with vgg19 part as functional")

args = parser.parse_args()

data_format = 'channels_last'
dropout_rate = 0.
num_epoch = 1 #500
patience = 0 # 100

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

if args.version == 0:
	mdl = get_vgg19_mdl_from_scratch(num_classes = 2, input_shape = (32,32,3))
elif args.version == 1:
	mdl = get_vgg19_mdl_from_scratch_v2(num_classes = 2, input_shape = (32,32,3), batch_norm = False, dropout_rate = dropout_rate)
elif args.version == 2:
	mdl = get_vgg19_from_trained(num_classes = 2, input_shape = (32,32,3), dropout_rate = dropout_rate) # "dropout_rate == 0" means we won't use dropout
else:
	print ("version {} is not supported".format(args.version))
	assert False 

train(mdl, train_data[0], data_util.format_label(train_data[1], 2), 
	test_data[0], data_util.format_label(test_data[1],2), destfile, 
	num_epoch = num_epoch, patience = patience, batch_size = 64)

saved_model = tf.keras.models.load_model(destfile)

# evaluate the model
_, train_acc = saved_model.evaluate(train_data[0], data_util.format_label(train_data[1], 2), verbose=0)
_, test_acc = saved_model.evaluate(test_data[0], data_util.format_label(test_data[1], 2), verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#checkpoint_path = os.path.join(args.dest, "cp.best.ckpt") 
#mdl.load_weights(checkpoint_path)
new_mdl_destfile = os.path.join(args.dest, "LFW_gender_classifier_v2.h5")

if args.version == 2:
	frame_mdl = get_vgg19_mdl_from_scratch_v2(
		num_classes = 2, input_shape = (32,32,3), batch_norm = False, dropout_rate = dropout_rate, data_format = data_format)

	# get weights
	weights_from_transfered_learning = get_weights_of_transfered_vgg19(saved_model)
	filled_mdl = update_weights_of_frame_model(frame_mdl, weights_from_transfered_learning)
	tf.keras.models.save_model(filled_mdl, best_mdl_destfile)
#else:
#	tf.keras.models.save_model(mdl, best_mdl_destfile)
