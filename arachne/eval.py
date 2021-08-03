import argparse
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import utils.data_util as data_util

def compute_acc(pred, y):
	pred_prob = np.argmax(pred, axis = 1)
	return np.sum(pred_prob == y)/len(y)

def get_top_10_freq(pred, y):
	pred_prob = np.argmax(pred, axis = 1)
	indices_to_wrong = np.where(pred_prob != y)[0]
	pairs = set([(pred_prob[i], y[i]) for i in indices_to_wrong])

	cnts = {}
	for i in indices_to_wrong:
		k = (pred_prob[i], y[i])
		if k not in cnts.keys():
			cnts[k] = 0
		cnts[k] += 1
	
	cnt_vs = list(cnts.items())
	top_10_freq_pairs = sorted(cnt_vs, reverse = True, key = lambda v:v[1])[:10]
	return top_10_freq_pairs

def cnt_error_num(pred, y, k):
	pred_prob = np.argmax(pred, axis = 1)
	#indices_to_wrong = np.where(pred_prob != y)[0]
	pairs = list(zip(list(pred_prob), list(y)))
	#print ("p", pairs[9], k[0])
	cnt = pairs.count(k[0])
	return cnt
			
	
	
parser = argparse.ArgumentParser()
parser.add_argument("-datadir", type = str, default = None)
parser.add_argument("-model_path", type = str)
parser.add_argument("-model_w", type = str, default = None)
parser.add_argument("-which_data", type = str)

args = parser.parse_args()

train_data, test_data = data_util.load_data(args.which_data, args.datadir)

mdl = load_model(args.model_path)
if args.model_w is not None:
	with open(args.model_w, 'rb') as f:
		new_weights = pickle.load(f)
else:
	new_weights = {}

pred_train = mdl.predict(train_data[0])
if len(pred_train.shape) == 3:
	pred_train = pred_train.reshape(pred_train.shape[0], pred_train.shape[-1])
pred_test = mdl.predict(test_data[0])
if len(pred_test.shape) == 3:
	pred_test = pred_test.reshape(pred_test.shape[0], pred_test.shape[-1])
top_10_freq = get_top_10_freq(pred_test, test_data[1])

print ("For the original model")
print ("\tTrain: {}".format(compute_acc(pred_train, train_data[1])))
print ('\tTest: {}'.format(compute_acc(pred_test, test_data[1])))
for k in top_10_freq:
	print ("\t{} -> {}: {}".format(k[0][1], k[0][0], cnt_error_num(pred_test, test_data[1], k)))

# set new 

for idx_to_tl, new_w in new_weights.items():
	mdl.layers[idx_to_tl].set_weights([new_w, mdl.layers[idx_to_tl].get_weights()[1]])

print ("For a new model")
pred_train = mdl.predict(train_data[0])
if len(pred_train.shape) == 3:
	pred_train = pred_train.reshape(pred_train.shape[0], pred_train.shape[-1])
pred_test = mdl.predict(test_data[0])
if len(pred_test.shape) == 3:
	pred_test = pred_test.reshape(pred_test.shape[0], pred_test.shape[-1])
print ("\tTrain: {}".format(compute_acc(pred_train, train_data[1])))
print ('\tTest: {}'.format(compute_acc(pred_test, test_data[1])))
for k in top_10_freq:
	print ("\t{} -> {}: {}".format(k[0][1], k[0][0], cnt_error_num(pred_test, test_data[1], k)))
