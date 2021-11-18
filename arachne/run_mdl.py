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


def get_data_for_evaluation(**kwargs):
	"""
	for this, we will look at h
	"""
	index_file = kwargs['file']
	assert index_file.endswith(".csv"), index_file
	rq = kwargs['rq']
	data_X = kwargs['X']; data_y = kwargs['y']
	# for rq2 (reproduce the random selection of misclassified inputs)
	seed = kwargs['seed']
	#which_data = kwargs['which_data']

	if rq == 2: # sam
		# index_file = prediction file
		# here, we need an indices to the targets that we tried to correct
		# used data -> all correct inputs in the test data and 10% of the misclassified inputs
		indices_to_targeted = data_util.get_misclf_for_rq2(
			index_file, percent = 0.1, seed = seed)
		
		# used data = test data (we don't have a pure eval data)
		used_data = (data_X, data_y)
		return (indices_to_targeted, used_data)
	elif rq == 3:
		top_n = kwargs['n']
		outs = data_util.gen_data_for_rq3(index_file, top_n, idx = 0) 
		assert len(outs) == 4, index_file
		misclf_key, misclf_indices, new_data_indices, new_test_indices = outs
		used_X = data_X[new_data_indices]; eval_X = data_X[new_test_indices]
		used_misclf_X = data_X[misclf_indices]
		used_y = data_y[new_data_indices]; eval_y = data_y[new_test_indices]
		used_misclf_y = data_y[misclf_indices]
		
		used_data = (used_X, used_y)
		eval_data = (eval_X, eval_y)
		used_misclf_data = (used_misclf_X, used_misclf_y)
		
		print ("RQ3: processed {}".format(misclf_key))
		return (used_data, eval_data, used_misclf_data)
	else:
		pass 
	pass
	

parser = argparse.ArgumentParser()
parser.add_argument("-init_mdl", "--path_to_init_model", type = str)
parser.add_argument("-patch", "--path_to_patch", type = str)
parser.add_argument("-datadir", type = str, default = "KangSeong/data")
parser.add_argument("-dest", type = str, default = None)
parser.add_argument("-which", type = str, default = "simple_cm", help = "cnn1,cnn2,cnn3,simple_fm,simple_cm")
parser.add_argument("-index_file", type = str, default = None)
parser.add_argument("-which_data", type = str, default = 'cifar10', help = "cifar10, fashion_mnist")
#parser.add_argument("-on_test", type = int, default = 0)
parser.add_argument("-num_label", type = int, default = 10)
parser.add_argument("-batch_size", type = int, default = None)
parser.add_argument("-rq", type = int, default = 0, help = "should be one of: 2, 3, 4, 5 (can be extended)")
parser.add_argument("-top_n", type = int, default = 1, help = "required for rq3")
parser.add_argument("-seed", type = int, help = "required for rq2")
parser.add_argument("-on_both", action='store_true')
args = parser.parse_args()

if args.dest is None:
	dest = os.path.join(os.path.dirname(args.path_to_patch), 'pred')
else:
	dest = args.dest
os.makedirs(dest, exist_ok = True)
#HAS_RUN_ON_TEST = bool(args.on_test)

train_data, test_data = data_util.load_data(args.which_data, args.datadir, with_hist = False)
train_X,train_y = train_data
X,y = test_data
print ('Training: {}, Test: {}'.format(len(train_y), len(y)))
indices = np.arange(len(y))
num_label = args.num_label

if args.rq == 2:
	params = {'X':X, 'y':y, 'rq':args.rq, 'file':args.index_file, 'seed':args.seed}
	if not args.on_both:
		(indices_to_targeted, used_data) = get_data_for_evaluation(
			X=X, y=y, rq=args.rq, file=args.index_file, seed=args.seed)#, which_data = args.which_data)
		used_X, used_y = used_data
	else:
		(indices_to_targeted, used_data) = get_data_for_evaluation(
			X=train_X, y=train_y, rq=args.rq, file=args.index_file, seed=args.seed)
		used_X, used_y = train_X, train_y
		eval_X, eval_y = X,y
elif args.rq == 3:
	#params = {'X':X, 'y':y, 'rq':args.rq, 'file':args.index_file, 'n':args.top_n}
	#(used_data, eval_data, used_misclf_data) = get_data_for_evaluation(params)
	(used_data, eval_data, used_misclf_data) = get_data_for_evaluation(
		X=X, y=y, rq=args.rq, file=args.index_file, n=args.top_n) #, which_data = args.which_data)
	used_X, used_y = used_data
	eval_X, eval_y = eval_data 
else:
	pass

#print (used_X.shape, used_y.shape, eval_X.shape, eval_y.shape)

#sys.exit()
#if HAS_RUN_ON_TEST:
	#msg = "should give an index file of test data for those used for training"
	#assert args.index_file is not None, msg
	#index_file =  args.index_file
	#import pandas as pd
	#val_indices = pd.read_csv(index_file)['index'].values
	#test_indices = [i for i in range(len(y)) if i not in val_indices]
	#val_X = X[val_indices]; val_y = y[val_indices]
	#X = X[test_indices]; y = y[test_indices]	
#else:
	#val_indices = None
	#test_indices = None	

# get weights
pred_name = os.path.basename(args.path_to_patch).replace("model", "pred")[:-4]

init_model = load_model(args.path_to_init_model, compile = False)
#init_pred_df_test = run_model(init_model, X, y, args.which_data)
#init_pred_df_train = run_model(init_model, train_X, train_y, args.which_data)
init_pred_df_used = run_model(init_model, used_X, used_y, args.which_data)
if args.on_both or args.rq in [3]:
	init_pred_df_eval = run_model(init_model, eval_X, eval_y, args.which_data)

####
input_reshape = args.which_data	== 'fashion_mnist'
need_act = args.which_data == 'GTSRB'
#print ("=========================Valid============================")
#if HAS_RUN_ON_TEST:
	#init_pred_df_val = run_model(init_model, val_X, val_y, args.which_data)
	#aft_pred_df_val = gen_and_run_model(init_model, args.path_to_patch, val_X, val_y, num_label,
	        #input_reshape = input_reshape, need_act = need_act, batch_size = args.batch_size)
	#combined_df = combine_init_aft_predcs(init_pred_df_val, aft_pred_df_val)
	#filename = os.path.join(args.dest, pred_name + ".val.pkl")
	#combined_df.to_pickle(filename)
	#print ('For validation:')
	#compare(init_pred_df_val, aft_pred_df_val)
if args.on_both or args.rq in [3]: # for rq2, this doesn't apply 
	print ("=========================For evaluation============================")
	aft_pred_df_eval = gen_and_run_model(init_model, args.path_to_patch, eval_X, eval_y, num_label,
		input_reshape = input_reshape, need_act = need_act, batch_size = args.batch_size)
	combined_df = combine_init_aft_predcs(init_pred_df_eval, aft_pred_df_eval)
	filename = os.path.join(dest, pred_name + ".eval.pkl")
	combined_df.to_pickle(filename)

	print ('For test:')
	compare(init_pred_df_eval, aft_pred_df_eval)

print ("=========================Used for patching============================")
init_model.summary()
aft_pred_df_used = gen_and_run_model(init_model, args.path_to_patch, used_X, used_y, num_label,
    input_reshape = input_reshape, need_act = need_act, batch_size = args.batch_size)
combined_df = combine_init_aft_predcs(init_pred_df_used, aft_pred_df_used)
filename = os.path.join(dest, pred_name + ".train.pkl")
combined_df.to_pickle(filename)

print ('For train:')
compare(init_pred_df_used, aft_pred_df_used)

if args.rq == 2:
	target_df = combined_df.iloc[indices_to_targeted]
	num_corrected = np.sum((target_df.true == target_df.new_pred).values)
	total_targeted = len(indices_to_targeted)
	print ("Out of {}, {} are corrected: {}%".format(
		total_targeted, num_corrected, np.round(100*num_corrected/total_targeted, decimals = 2)))



