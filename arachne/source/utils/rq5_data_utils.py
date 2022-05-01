import numpy as np

def get_misclf_indices_balanced(df, idx = 0):
	"""
	"""
	misclf_types = list(set([tuple(pair) for pair in df[["true","pred"]].values]))
	ret_misclfds = {}
	for misclf_type in misclf_types:
		misclf_type = tuple(misclf_type)
		true_label, pred_label = misclf_type
		indices_to_misclf = df.loc[
			(df.true == true_label) & (df.pred == pred_label)].index.values
	
		if len(indices_to_misclf) >= 2:
			indices_1, indices_2 = np.array_split(indices_to_misclf, 2)
			ret_misclfds[misclf_type] = indices_1 if idx == 0 else indices_2
		else: # a single input
			ret_misclfds[misclf_type] = indices_to_misclf
	
	return ret_misclfds


def sort_keys_by_cnt(misclfds):
	"""
	"""
	cnts = []
	for misclf_key in misclfds:
		cnts.append([misclf_key, len(misclfds[misclf_key])])
	sorted_keys = [v[0] for v in sorted(cnts, key = lambda v:v[1], reverse = True)]
	return sorted_keys


def get_balanced_dataset(pred_file, top_n, idx = 0):
	"""
	generate the training and test dataset for rq3 ~ rq6 
	idx = 0 or 1 -> to which half, 0 = front half, 1 = latter half
	"""
	import pandas as pd
	
	idx = idx if idx == 0 else 1 # only 0 or 1
	target_idx = idx; eval_idx = np.abs(1 - target_idx)
	
	df = pd.read_csv(pred_file, index_col = 'index')
	misclf_df = df.loc[df.true != df.pred]
	misclfds_idx_target = get_misclf_indices_balanced(misclf_df, idx = target_idx)
	sorted_keys = sort_keys_by_cnt(misclfds_idx_target) # for patch generation
	misclfds_idx_eval = get_misclf_indices_balanced(misclf_df, idx = eval_idx)

	indices_to_corr = df.loc[df.true == df.pred].sort_values(by=['true']).index.values
	indices_to_corr_target = [_idx for i,_idx in enumerate(indices_to_corr) if i % 2 == target_idx]
	indices_to_corr_eval = [_idx for i,_idx in enumerate(indices_to_corr) if i % 2 == eval_idx]

	np.random.seed(0)
	if top_n < len(sorted_keys):
		misclf_key = sorted_keys[top_n]
		misclf_indices = misclfds_idx_target[misclf_key]

		new_data_indices = []; new_test_indices = []
		for sorted_k in sorted_keys: # this means that all incorrect ones are include in new_data
			new_data_indices.extend(misclfds_idx_target[sorted_k])
			new_test_indices.extend(misclfds_idx_eval[sorted_k])
			
		new_data_indices += indices_to_corr_target
		new_test_indices += indices_to_corr_eval

		np.random.shuffle(new_data_indices)
		np.random.shuffle(new_test_indices)	

		return (misclf_key, misclf_indices, new_data_indices, new_test_indices)
	else:
		return len(sorted_keys)
