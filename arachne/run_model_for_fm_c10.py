import os, sys
import argparse
from utils import data_util
from tensorflow.keras.models import load_model
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import json
import tensorflow as tf
from utils.run_utils import get_weights, run_model, record_predcs

HAS_RUN_ON_TEST = True


parser = argparse.ArgumentParser()
parser.add_argument("-model", type = str)
parser.add_argument("-datadir", type = str, default = "KangSeong/data")
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-w", type = str, default = None, help = "model.misclf-test.top.0.1.0.json")
parser.add_argument("-which", type = str, default = "simple_cm", help = "cnn1,cnn2,cnn3,simple_fm,simple_cm")
parser.add_argument("-val_index_file", type = str, default = None)
parser.add_argument("-which_data", type = str, default = 'cifar10', help = "cifar10, fashion_mnist")

args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)
HAS_RUN_ON_TEST = True if 'cnn' in args.which else False

train_data, test_data = data_util.load_data(args.which_data, args.datadir)
train_X,train_y = train_data
X,y = test_data
indices = np.arange(len(y))

num_label = 10

if HAS_RUN_ON_TEST:
	msg = "should give an index file of test data for those used for training"
	assert args.val_index_file is not None, msg
	index_file =  args.val_index_file
	
	import pandas as pd
	val_indices = pd.read_csv(index_file)['index'].values
	test_indices = [i for i in range(len(y)) if i not in val_indices]

	val_X = X[val_indices]; val_y = y[val_indices]
	X = X[test_indices]; y = y[test_indices]	
else:
	val_indices = None
	test_indices = None	

# get weights
with open(args.w) as f:
	weights = json.load(f)
wname = os.path.basename(args.w).replace("model", "pred")

loaded_model = load_model(args.model)
kernel_and_bias_pairs = get_weights(loaded_model)
initial_weights = kernel_and_bias_pairs
if weights is not None:
	initial_weights[-1][0] = np.float32(weights['weight'])


print ("=========================Valid============================")
if HAS_RUN_ON_TEST:
	pred_labels, corr_cnt, num, sess = run_model(args.which, val_X, val_y, num_label, args.model, initial_weights, sess = None)
	sess.close()
	print ("Val:\n\tOver {}, {} correctly predicted inputs ({}%)".format(num, corr_cnt, corr_cnt/num))

	filename = os.path.join(args.dest, wname + ".val.csv")
	record_predcs(val_y, pred_labels, filename, val_indices)


print ("=========================Test============================")
pred_labels, corr_cnt, num, sess = run_model(args.which, X, y, num_label, args.model, initial_weights, sess = None)
sess.close()
print ("Test:\n\tOver {}, {} correctly predicted inputs ({}%)".format(num, corr_cnt, corr_cnt/num))

filename = os.path.join(args.dest, wname + ".test.csv")
if test_indices	is None:
	test_indices = np.arange(len(y))
record_predcs(y, pred_labels, filename, test_indices)


print ("=========================Train============================")
train_pred_labels, corr_cnt, num, sess = run_model(args.which, train_X, train_y, num_label, args.model, kernel_and_bias_pairs, sess = None)
sess.close()
print ("Train:\n\tOver {}, {} correctly predicted inputs ({}%)".format(num, corr_cnt, corr_cnt/num))

filename = os.path.join(args.dest, wname + ".train.csv")
record_predcs(train_y, train_pred_labels, filename, np.arange(len(train_y)))

