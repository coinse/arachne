import os, sys
import argparse
from utils import data_util
from tensorflow.keras.models import load_model
import numpy as np
import json
from utils.run_utils import get_weights, run_model, record_predcs

which = 'lfw_vgg'

parser = argparse.ArgumentParser()
parser.add_argument("-model", type = str)
parser.add_argument("-datadir", type = str, default = "data/lfw")
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-w", type = str, default = None, help = "model.misclf-test.top.0.1.0.json")

args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)

train_data, test_data = data_util.load_data('lfw', args.datadir,
	path_to_female_names = "lfw_np/female_names_lfw.txt")
X,y = test_data
train_X,train_y = train_data

indices = np.arange(len(y))

with open(args.w) as f:
	weights = json.load(f)
wname = os.path.basename(args.w).replace("model", "pred")

import utils.torch_rel_util as torch_rel_util
kernel_and_bias_pairs = torch_rel_util.get_weights(args.model)

initial_weights = kernel_and_bias_pairs
if weights is not None:
	initial_weights[-1][0] = np.float32(weights['weight'])

num_label = 2
print ("=========================Test============================")
pred_labels, corr_cnt, num, sess = run_model(which, X[1::2], y[1::2], num_label, args.model, initial_weights, 
	sess = None, is_train = False, indices = np.arange(len(y))[1::2], use_raw = True) #kernel_and_bias_pairs)
sess.close()
print ("Test:\n\tOver {}, {} correctly predicted inputs ({}%)".format(num, corr_cnt, corr_cnt/num))
filename = os.path.join(args.dest, wname + ".test.csv")
record_predcs(y[1::2], pred_labels, filename, np.arange(len(y))[1::2])

print ("==========================Val===========================")
pred_labels, corr_cnt, num, sess = run_model(which, X[::2], y[::2], num_label, args.model, initial_weights, 
	sess = None, is_train = False, indices = np.arange(len(y))[::2], use_raw = True) #kernel_and_bias_pairs)
sess.close()
print ("Val:\n\tOver {}, {} correctly predicted inputs ({}%)".format(num, corr_cnt, corr_cnt/num))
filename = os.path.join(args.dest, wname + ".val.csv")
record_predcs(y[::2], pred_labels, filename, np.arange(len(y))[::2])


print ("\n=========================Train============================")
train_pred_labels, corr_cnt, num, sess = run_model(which, train_X, train_y, num_label, args.model, kernel_and_bias_pairs, sess = None, is_train = True)
sess.close()
print ("Train:\n\tOver {}, {} correctly predicted inputs ({}%)".format(num, corr_cnt, corr_cnt/num))

filename = os.path.join(args.dest, wname + ".train.csv")
record_predcs(train_y, train_pred_labels, filename, np.arange(len(train_y)))

