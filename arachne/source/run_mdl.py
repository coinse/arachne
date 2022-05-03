import os
import argparse
from utils import data_util
from tensorflow.keras.models import load_model
import numpy as np
from utils.run_utils import run_model, gen_and_run_model
from utils.eval_util import combine_init_aft_predcs

def compare(init_pred_df, aft_pred_df):
	"""
	"""
	num_patched = len(init_pred_df.loc[
		(init_pred_df.flag == False) & (aft_pred_df.flag == True)])
	rr = num_patched/np.sum(init_pred_df.true != init_pred_df.pred)
	num_broken = len(init_pred_df.loc[
		(init_pred_df.flag == True) & (aft_pred_df.flag == False)])
	br = num_broken/np.sum(init_pred_df.true == init_pred_df.pred)
	#init_acc = np.sum(init_pred_df.true == init_pred_df.pred)/len(init_pred_df)
	#aft_acc = np.sum(aft_pred_df.true == aft_pred_df.pred)/len(aft_pred_df)
	#print ("\tpatched: {}, broken: {} & acc: {} -> {}".format(
	#	num_patched, num_broken, 
	#	np.round(init_acc, decimals =4), np.round(aft_acc, decimals = 4)))
	print ("\tpatched: {} (overall RR: {:.4f}), broken: {} (BR: {:.4f})".format(
		num_patched, np.round(rr, decimals=4), 
		num_broken, np.round(br, decimals=4)))
	
def get_data_for_evaluation(**kwargs):
	"""
	"""
	index_file = kwargs['file']
	assert index_file.endswith(".csv"), index_file
	rq = kwargs['rq']
	data_X = kwargs['X']; data_y = kwargs['y']

	if rq == 2:
		# used data: 
		# indices_to_targeted: the targeted random 10% miclassifications
		seed = kwargs['seed']
		indices_to_targeted = data_util.get_misclf_for_rq2(
			index_file, percent = 0.1, seed = seed)
		
		used_data = (data_X, data_y)
		return (indices_to_targeted, used_data)
	else: # all the remaining rqs: 3~7
		top_n = kwargs['n']
		# idx = 0 -> 0 is used for patch generation
		outs = data_util.get_balanced_dataset(index_file, top_n, idx = 0)
		misclf_key, misclf_indices, new_data_indices, new_test_indices = outs	
		print ("Processing: {},{}".format(misclf_key[0], misclf_key[1]))

		used_X = data_X[new_data_indices]; eval_X = data_X[new_test_indices]
		used_misclf_X = data_X[misclf_indices]
		used_y = data_y[new_data_indices]; eval_y = data_y[new_test_indices]
		used_misclf_y = data_y[misclf_indices]
		
		used_data = (used_X, used_y)
		eval_data = (eval_X, eval_y)
		used_misclf_data = (used_misclf_X, used_misclf_y)
		
		print ("RQ{}: processed {}".format(rq, misclf_key))
		return (used_data, eval_data, used_misclf_data)


parser = argparse.ArgumentParser()
parser.add_argument("-init_mdl", "--path_to_init_model", type = str)
parser.add_argument("-patch", "--path_to_patch", type = str)
parser.add_argument("-datadir", type = str, default = None)
parser.add_argument("-dest", type = str, default = None)
parser.add_argument("-index_file", type = str, default = None)
parser.add_argument("-which_data", type = str, default = 'cifar10', 
	help = "cifar10, fashion_mnist, GTSRB, us_airline, fm_for_rq5")
parser.add_argument("-num_label", type = int, default = 10)
parser.add_argument("-batch_size", type = int, default = None)
parser.add_argument("-rq", type = int, default = 0, help = "research question number:2~7")
parser.add_argument("-top_n", type = int, default = 0, help = "required for rq3")
parser.add_argument("-female_lst_file", action = 'store',
	default = None, help = 'data/lfw_np/female_names_lfw.txt', type = str)
args = parser.parse_args()

if not os.path.exists(args.path_to_patch): 
	# pattern is given instead of a concrete path
	import glob
	afiles = glob.glob(args.path_to_patch)
	assert len(afiles) == 1, afiles 
	path_to_patch = afiles[0]
else:
	path_to_patch = args.path_to_patch

if args.dest is None:
	dest = os.path.join(
		os.path.dirname(path_to_patch), 'pred')
else:
	dest = args.dest
os.makedirs(dest, exist_ok = True)

# data specific configuration 
num_label = args.num_label
if args.which_data == 'us_airline':
	has_lstm_layer = True
else:
    has_lstm_layer = False
#input_reshape = args.which_data	== 'fashion_mnist'
need_act = args.which_data == 'GTSRB'
# 

if args.which_data != 'fm_for_rq5':
	train_data, test_data = data_util.load_data(
		args.which_data, 
		args.datadir, 
		path_to_female_names = args.female_lst_file)
	train_X,train_y = train_data
	X,y = test_data
	print ('Training: {}, Test: {}'.format(len(train_y), len(y)))
else:
	# retrive the half of the test data, denoted as validation data
	data = data_util.load_rq5_fm_test_val(args.datadir, which_type = "both") 

# get data for the prediction
if args.rq == 2:
	seed = int(os.path.basename(path_to_patch).split(".")[-2])
	params = {'X':X, 'y':y, 'rq':args.rq, 'file':args.index_file, 'seed':seed}
	(indices_to_targeted, used_data) = get_data_for_evaluation(
		X=X, y=y, rq=args.rq, 
		file=args.index_file, seed=seed) 
	used_X, used_y = used_data
else: 
	if args.which_data != 'fm_for_rq5':
		(used_data, eval_data, used_misclf_data) = get_data_for_evaluation(
			X=X, y=y, rq=args.rq, 
			file=args.index_file, n=args.top_n)
		used_X, used_y = used_data
		eval_X, eval_y = eval_data
		print ("used", used_X.shape, used_y.shape)	
		print ("eval", eval_X.shape, eval_y.shape)
	else:
		used_X, used_y = data['val']
		eval_X, eval_y = data['test']
		print ("used", used_X.shape, used_y.shape)
		print ("eval", eval_X.shape, eval_y.shape)

# for the prediction file name
pred_name = os.path.basename(path_to_patch).replace("model", "pred")[:-4]

init_model = load_model(args.path_to_init_model, compile = False)
init_pred_df_used = run_model(init_model, used_X, used_y)

if args.rq != 2:
	init_pred_df_eval = run_model(init_model, eval_X, eval_y)

if args.rq != 2:
	print ("===========================For evaluation=============================")
	aft_pred_df_eval = gen_and_run_model(
		init_model, path_to_patch, 
		eval_X, eval_y, num_label, 
		has_lstm_layer = has_lstm_layer, 
		need_act = need_act, 
		batch_size = args.batch_size)

	combined_df = combine_init_aft_predcs(init_pred_df_eval, aft_pred_df_eval)
	filename = os.path.join(dest, pred_name + ".eval.pkl")
	print ("results written to {}".format(filename))
	combined_df.to_pickle(filename)
	print ('For Evaluation:')
	compare(init_pred_df_eval, aft_pred_df_eval)

print ("=========================Used for patching============================")
aft_pred_df_used = gen_and_run_model(
	init_model, path_to_patch, 
	used_X, used_y, num_label, 
	has_lstm_layer = has_lstm_layer, 
	need_act = need_act, 
	batch_size = args.batch_size)

combined_df = combine_init_aft_predcs(init_pred_df_used, aft_pred_df_used)
filename = os.path.join(dest, pred_name + ".train.pkl")
print ("results written into {}".format(filename))
combined_df.to_pickle(filename)
print ('For Patch (Validation set):')

compare(init_pred_df_used, aft_pred_df_used)
if args.rq == 2: # additionaly analysis for RQ2: for the selected 10%
	target_df = combined_df.iloc[indices_to_targeted]
	num_corrected = np.sum((target_df.true == target_df.new_pred).values)
	total_targeted = len(indices_to_targeted)
	print ("Out of {}, {} are corrected: {}% (RR)".format(
		total_targeted, num_corrected, 
		np.round(
			100*num_corrected/total_targeted, decimals = 2)))