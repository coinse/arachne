import os
import argparse
from utils import data_util
from tensorflow.keras.models import load_model
import numpy as np
from utils.run_utils import run_model, gen_and_run_model
from main_rq1 import combine_init_aft_predcs

def compare(init_pred_df, aft_pred_df):
	"""
	"""
	num_patched = len(init_pred_df.loc[(init_pred_df.flag == False) & (aft_pred_df.flag == True)])
	num_broken = len(init_pred_df.loc[(init_pred_df.flag == True) & (aft_pred_df.flag == False)])
	init_acc = np.sum(init_pred_df.true == init_pred_df.pred)/len(init_pred_df)

	aft_acc = np.sum(aft_pred_df.true == aft_pred_df.pred)/len(aft_pred_df)
	print ("\tpatched: {}, broken: {}: {} -> {}".format(num_patched, num_broken, init_acc, aft_acc))

	
parser = argparse.ArgumentParser()
parser.add_argument("-init_mdl", "--path_to_init_model", type = str)
parser.add_argument("-patch", "--path_to_patch", type = str)
parser.add_argument("-datadir", type = str, default = "KangSeong/data")
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-which", type = str, default = "simple_cm", help = "cnn1,cnn2,cnn3,simple_fm,simple_cm")
parser.add_argument("-val_index_file", type = str, default = None)
parser.add_argument("-which_data", type = str, default = 'cifar10', help = "cifar10, fashion_mnist")
parser.add_argument("-on_test", type = int, default = 0)
parser.add_argument("-num_label", type = int, default = 10)
parser.add_argument("-batch_size", type = int, default = None)

args = parser.parse_args()

os.makedirs(args.dest, exist_ok = True)
HAS_RUN_ON_TEST = bool(args.on_test)

train_data, test_data = data_util.load_data(args.which_data, args.datadir, with_hist = False)
train_X,train_y = train_data
X,y = test_data
print ('Training: {}, Test: {}'.format(len(train_y), len(y)))
indices = np.arange(len(y))

num_label = args.num_label

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
pred_name = os.path.basename(args.path_to_patch).replace("model", "pred")[:-4]

init_model = load_model(args.path_to_init_model, compile = False)
init_pred_df_test = run_model(init_model, X, y, args.which_data)
init_pred_df_train = run_model(init_model, train_X, train_y, args.which_data)

input_reshape = args.which_data	== 'fashion_mnist'
need_act = args.which_data == 'GTSRB'

print ("=========================Valid============================")
if HAS_RUN_ON_TEST:
	init_pred_df_val = run_model(init_model, val_X, val_y, args.which_data)
	aft_pred_df_val = gen_and_run_model(init_model, args.path_to_patch, val_X, val_y, num_label,
	        input_reshape = input_reshape, need_act = need_act, batch_size = args.batch_size)

	combined_df = combine_init_aft_predcs(init_pred_df_val, aft_pred_df_val)
	filename = os.path.join(args.dest, pred_name + ".val.pkl")
	combined_df.to_pickle(filename)

	print ('For validation:')
	compare(init_pred_df_val, aft_pred_df_val)

print ("=========================Test============================")
aft_pred_df_test = gen_and_run_model(init_model, args.path_to_patch, X, y, num_label,
	input_reshape = input_reshape, need_act = need_act, batch_size = args.batch_size)
combined_df = combine_init_aft_predcs(init_pred_df_test, aft_pred_df_test)
filename = os.path.join(args.dest, pred_name + ".test.pkl")
combined_df.to_pickle(filename)

print ('For test:')
compare(init_pred_df_test, aft_pred_df_test)


print ("=========================Train============================")
aft_pred_df_train = gen_and_run_model(init_model, args.path_to_patch, train_X, train_y, num_label,
        input_reshape = input_reshape, need_act = need_act, batch_size = args.batch_size)
combined_df = combine_init_aft_predcs(init_pred_df_train, aft_pred_df_train)
filename = os.path.join(args.dest, pred_name + ".train.pkl")
combined_df.to_pickle(filename)

print ('For train:')
compare(init_pred_df_train, aft_pred_df_train)



