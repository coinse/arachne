"""
"""
import pandas as pd
import os, sys
import numpy as np

# tie-breaker for RQ1
TIE = 'avg' #'MAX'

def read_and_add_flag(filename):
	"""
	"""
	df = pd.read_csv(filename)
	indices = df.index.values
	df['flag'] = df.true == df.pred
	return df


def compute_acc(init_pred_df, aft_pred_df, classes):
	"""
	"""
	from sklearn.metrics import accuracy_score
	# per class
	acc_per_class = {}
	for class_label in classes:
		to_target = init_pred_df.true.values == class_label
		to_target_aft = aft_pred_df.true.values == class_label
		assert all(to_target == to_target_aft), class_label

		labels = init_pred_df.loc[to_target].true.values
		init_pred_for_class = init_pred_df.loc[to_target].pred.values

		aft_pred_for_class = aft_pred_df.loc[to_target].pred.values

		init_acc_for_class = accuracy_score(labels, init_pred_for_class)
		aft_acc_for_class = accuracy_score(labels, aft_pred_for_class)

		init_acc_for_class = np.sum([l==p for l,p in zip(init_pred_for_class,labels)])/np.sum(to_target)
		aft_acc_for_class = np.sum([l==p for l,p in zip(aft_pred_for_class,labels)])/np.sum(to_target)

		acc_per_class[class_label] = {'init':init_acc_for_class, 'aft':aft_acc_for_class}

	return acc_per_class


def combine_init_aft_predcs(init_pred_df, aft_pred_df):
	"""
	"""
	# combine
	combined_df = pd.DataFrame(data = {
		'true':init_pred_df.true.values, 
		'pred':init_pred_df.pred.values,
		'new_pred':aft_pred_df.pred.values,
		'init_flag':init_pred_df.flag})

	return combined_df


def classify_changes(result_df):
	"""
	"""
	classes = set(result_df.true)
	clf_changes = {}

	changed = result_df.loc[result_df.pred != result_df.new_pred]

	for class_label in classes:
		t_df = result_df.loc[result_df.true == class_label]
		cnt_patched = np.sum((t_df.true != t_df.pred) & (t_df.pred != t_df.new_pred) & (t_df.new_pred == t_df.true))
		cnt_broken = np.sum((t_df.true == t_df.pred) & (t_df.pred != t_df.new_pred) & (t_df.new_pred != t_df.true))

		clf_changes[class_label] = {'patched':cnt_patched, 'broken':cnt_broken}

	return clf_changes
	

def get_weight(path_to_model):
	"""
	"""
	import utils.apricot_rel_util as apricot_rel_util

	kernel_and_bias_pairs = apricot_rel_util.get_weights(path_to_model, start_idx = 0)
	return kernel_and_bias_pairs[-1][0]

def get_gts(init_weight, aft_weight):
	"""
	"""
	gts = np.where(init_weight != aft_weight)
	return gts


def compute_roc_auc(ranks, gts):
	"""
	"""
	from sklearn.metrics import auc,roc_curve 
	from sklearn.preprocessing import MinMaxScaler
	
	scaler = MinMaxScaler()	

	uniq_ranks = sorted(list(set(ranks.reshape(-1,))))

	ranks_of_gts = ranks[gts]
	num_gts = len(ranks_of_gts)

	recall_per_ranks = []
	for i,uniq_rank in enumerate(uniq_ranks):
		cnt_loc = np.sum(ranks_of_gts <= uniq_rank)
		recall_per_ranks.append(cnt_loc/num_gts)

	xs = [int(r) for r in uniq_ranks]
	auc_score = auc(xs, recall_per_ranks)
	
	return auc_score, list(zip(xs, recall_per_ranks))


def eval_loc_acc(loc_file, gts, weight_shape, loc_method = 'localiser'):
	"""
	"""
	import json

	row_indices = lambda _front: np.asarray(_front)[:,0]	
	col_indices = lambda _front: np.asarray(_front)[:,1]

	if loc_method == 'localiser':
		with open(loc_file) as f:
			locs = json.load(f)

		fronts = locs['fronts']
		ranks = -np.ones(weight_shape)

		cnt = 1
		for i, front in enumerate(fronts):
			if TIE == 'avg':
				ranks[row_indices(front),col_indices(front)] = cnt + (len(front)-1)/2
			else:
				ranks[row_indices(front),col_indices(front)] = cnt + len(front)	
			cnt += len(front) # update	
			
		assert all((ranks >= 0).tolist()), np.where(ranks < 0)
		# compute auc
		auc_score, pairs = compute_roc_auc(ranks, gts)
	else:
		import pandas as pd
		from scipy.stats import rankdata

		locs = pd.read_json(loc_file)
		grads = locs['grads']
		indices = locs['weights']

		if TIE == 'avg':
			tie_broken_ranks = rankdata(-grads, method = 'average')
		else:
			tie_broken_ranks = rankdata(-grads, method = 'max')

		ranks = -np.ones(weight_shape)
		for i, a_rank in zip(indices, tie_broken_ranks):
			ridx, cidx = i
			ranks[ridx, cidx] = a_rank

		assert all((ranks >= 0).tolist()), np.where(ranks < 0)
		auc_score, pairs = compute_roc_auc(ranks, gts)

	return auc_score, pairs


def gen_random_ranks(weight_shape):
	"""
	"""
	num_weights = weight_shape[0] * weight_shape[1]
	random_ranks = np.arange(num_weights)
	np.random.shuffle(random_ranks)

	random_ranks = random_ranks.reshape(weight_shape)
	return random_ranks


def eval_loc_random_baseline(gts, weight_shape):
	"""
	"""
	random_ranks = gen_random_ranks(weight_shape)
	auc_score, pairs = compute_roc_auc(random_ranks, gts)
	
	return auc_score, pairs


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-init_pred_file", type = str)
	parser.add_argument("-aft_pred_file", type = str)
	parser.add_argument("-dest", type = str, default = ".")
	parser.add_argument("-num_label", type = int, default = 10)
	parser.add_argument("-misclf_type", type = str, default = None, help = "5,3")
	parser.add_argument("-rq", type = int, default = 2, help = "1 (loc) vs (2,3,4)")
	parser.add_argument("-init_model", type = str, default = None, help = "initial model")
	parser.add_argument("-aft_model", type = str, default = None, help = "a model with injected faults")
	parser.add_argument("-loc_file", type = str, default = "FM.init_pred.loc.0.json")
	parser.add_argument("-gl_loc_file", type = str)
	parser.add_argument("-loc_method", type = str, default = "localiser", help = "localiser, gradient_loss, random")
	parser.add_argument("-key", type = str)

	args = parser.parse_args()	

	if args.rq == 1:
		which_data = os.path.basename(args.loc_file).split(".")[0]
		iter_idx = int(os.path.basename(args.loc_file).split(".")[-2])

		np.random.seed(iter_idx)

		init_weight = get_weight(args.init_model)
		aft_weight = get_weight(args.aft_model)

		gts = get_gts(init_weight, aft_weight)
		weight_shape = init_weight.shape
		assert weight_shape == aft_weight.shape, "{} vs {}".format(weight_shape, aft_weight.shape)

		auc_score, loc_pairs = eval_loc_acc(args.loc_file, gts, weight_shape, loc_method = 'localiser')
		#print ("AUC score for {} faulty neural weights: {}".format(len(gts[0]), auc_score))
		auc_score_of_random, rd_pairs = eval_loc_random_baseline(gts, weight_shape)
		#print ("AUC score for {} faulty neural weights: {}".format(len(gts[0]), auc_score))
		auc_score_of_gl, gl_pairs = eval_loc_acc(args.gl_loc_file, gts, weight_shape, loc_method = 'gradient_loss')

		print ("AUC score for {} faulty neural weights: {} vs {} (gl) vs {} (random)".format(
			len(gts[0]), auc_score, auc_score_of_gl, auc_score_of_random))
		
		destfile = os.path.join(args.dest, "{}.{}.{}.auc.csv".format(which_data, iter_idx, args.key))
		with open(destfile, 'w') as f:
			f.write("idx,loc,gl,random\n")
			f.write("{},{},{},{}\n".format(iter_idx, auc_score, auc_score_of_gl, auc_score_of_random))

		### for visualisation
		pair_dest = os.path.join(args.dest, 'pairs')
		os.makedirs(pair_dest, exist_ok = True)
		pair_destfile = os.path.join(pair_dest, "{}.{}.{}.pairs.json".format(which_data, iter_idx, args.key))
		in_df_form = {'loc':loc_pairs, 'gl':gl_pairs, 'random':rd_pairs}
		import json
		with open(pair_destfile, 'w') as f:
			f.write(json.dumps(in_df_form))
	else:
		init_pred_df = read_and_add_flag(args.init_pred_file)
		aft_pred_df = read_and_add_flag(args.aft_pred_file)
		classes = np.arange(args.num_label)	

		# acc per calss
		acc_per_class = compute_acc(init_pred_df, aft_pred_df, classes)
		for class_label in classes:
			num_class = np.sum(init_pred_df.true == class_label)
			print ("For class {}({}), {} => {}".format(
				class_label, 
				num_class, 
				acc_per_class[class_label]['init'], 
				acc_per_class[class_label]['aft']))	

		# 
		if args.misclf_type is not None:
			combined_df = combine_init_aft_predcs(init_pred_df, aft_pred_df)
			##
			t_df = combined_df.loc[combined_df.true == 6]
			cnt_init = np.sum(t_df.true == t_df.pred)
			cnt_aft = np.sum(t_df.true == t_df.new_pred)
			clf_changes = classify_changes(combined_df)	

			true_class, pred_class = args.misclf_type.split(",")
			true_class = int(true_class)
			pred_class = int(pred_class)

			print ("===========================================================")
			print ("For {},{}, {}: ({},{}), {}: ({},{})".format(true_class, pred_class,
				true_class, clf_changes[true_class]['patched'], clf_changes[true_class]['broken'],
				pred_class, clf_changes[pred_class]['patched'], clf_changes[pred_class]['broken']))	

