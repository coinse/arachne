import os, sys
import numpy as np
from numpy.core.fromnumeric import argmax, argsort
import pandas as pd
import pickle
import tqdm

def get_weight_and_cost(loc_which, seed, loc_file, target_weights, gts, method = 'max'):
	"""
	"""
	from scipy.stats import rankdata
	
	with open(loc_file, 'rb') as f:
		locs = pickle.load(f)

	pairs = {}	
	if loc_which == 'localiser':
		new_locs = []
		for loc in tqdm.tqdm(locs):
			indices, cost = loc
			idx_to_l, local_idx = indices
			local_idx = np.unravel_index(local_idx, target_weights[idx_to_l][0].shape)
			new_locs.append([[idx_to_l, local_idx], cost])
		
		costs = [vs[1] for vs in new_locs]
		indices = [vs[0] for vs in new_locs]
	
		ret_lst, ret_front_lst = compute_pareto(np.asarray(costs), np.asarray(indices), gts)
		#for i,r in enumerate(ret_lst):
		#	#rint (indices[i])
		#	pairs[r] = i+1 # 1 ~ num
		###
		arank = 0
		for i,rs in enumerate(ret_front_lst):
			arank += len(rs)
			for r in rs:
				pairs[r[0]] = (arank,r[1])
		###
	elif loc_which == 'gradient_loss':
		costs = [-vs[-1] for vs in locs]
		ranks = rankdata(costs, method = method)
		indices = [vs[0] for vs in locs]
		
		founds = np.asarray([False] * len(gts))
		for i,r in enumerate(ranks):
			pairs[tuple(indices[i])] = [r, costs[i]]

			to_look_indices = np.where(founds == False)[0]
			for idx in to_look_indices:
				founds[idx] = gts[idx] == tuple(indices[i])
			
			if all(founds): break
	else:
		np.random.seed(seed)
		nindices = np.arange(len(locs))
		np.random.shuffle(nindices)
		
		for i,aloc in enumerate(nindices):
			pairs[tuple(locs[aloc])] = [i+1, -1.] # 1 ~ num
	
	return pairs

def compute_pareto(costs, curr_nodes_to_lookat, gts):
	ret_lst = []
	ret_front_lst = []
	
	print ("Number of target to localise: {}".format(len(curr_nodes_to_lookat))) 
	num_total = len(costs)
	import time
	t1 = time.time()
	founds = np.asarray([False] * len(gts))
	while len(curr_nodes_to_lookat) > 0:
		t1 = time.time()
		_costs = costs.copy()
		is_efficient = np.arange(costs.shape[0])
		next_point_index = 0 # Next index in the is_efficient array to search for

		while next_point_index < len(_costs):
			nondominated_point_mask = np.any(_costs > _costs[next_point_index], axis=1)
			nondominated_point_mask[next_point_index] = True
			is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
			_costs = _costs[nondominated_point_mask]
			next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
		
		#current_ret = [tuple(v) for v in curr_nodes_to_lookat[is_efficient]]
		current_ret = [[tuple(v),c] for v,c in zip(curr_nodes_to_lookat[is_efficient], costs[is_efficient])]
		ret_lst.extend(current_ret)
		ret_front_lst.append(current_ret)

		to_look_indices = np.where(founds == False)[0]
		for idx in to_look_indices:
			founds[idx] = gts[idx] in [v[0] for v in current_ret]
		
		if all(founds):
			break 
		else:
			# remove selected items (non-dominated ones)
			curr_nodes_to_lookat = np.delete(curr_nodes_to_lookat, is_efficient, 0)
			costs = np.delete(costs, is_efficient, 0)
	
			t2 = time.time()
			print ("For computing pareto front", t2 - t1)
			print ("\tremain: {} out of {}: {} ({})".format(len(costs), num_total, num_total - len(costs), len(current_ret)))
			#if len(ret_lst):
			#	break		
			#sys.exit()
	return ret_lst, ret_front_lst

if __name__ == "__main__":
	import argparse
	from run_localise import get_target_weights
	from main_rq1 import return_target_mdl_and_gt_path

	parser = argparse.ArgumentParser()
	parser.add_argument("-which_data", type = str, help = "fashion_mnist, cifar10, GTSRB")
	parser.add_argument("-loc_which", default = 'localiser', type = str, help = "localiser, gradient_loss, random")
	parser.add_argument("-loc_dir", type = str)
	parser.add_argument("-fid_file", type = str)

	args = parser.parse_args()

	loc_loc_file = "c_loc/loc.all_cost.loc.{}.1.pkl"
	loc_grad_file = "grad/loc.all_cost.loc.{}.1.grad.pkl"
	loc_random_file = "random/loc.all_cost.loc.{}.1.random.pkl"

	if args.loc_which == 'localiser':
		loc_file = loc_loc_file
	elif args.loc_which == 'gradient_loss':
		loc_file = loc_grad_file
	else:
		loc_file = loc_random_file

	c10_mdl_path = "data/models/cifar_simple_90p.h5"
	fm_mdl_path = "data/models/fmnist_simple.h5"
	gtsrb_mdl_path = "data/models/GTSRB_NEW/simple/gtsrb.model.0.wh.0.h5"

	if args.which_data == 'cifar10':
		target_weights = get_target_weights(None, c10_mdl_path, indices_to_target = None, target_all = True)
	elif args.which_data == 'fashion_mnist':
		target_weights = get_target_weights(None, fm_mdl_path, indices_to_target = None, target_all = True)
	else:
		target_weights = get_target_weights(None, gtsrb_mdl_path, indices_to_target = None, target_all = True)

	
	dest = os.path.join(args.loc_dir, "pairs/{}".format(args.loc_which))
	os.makedirs(dest, exist_ok = True)
	comp = lambda a,b: a == b
	for seed in tqdm.tqdm(range(35)): #40)): # 40 for cifar10 and 31 for fm, 35 for GTSRB
		## get gt
		_, gt_file = return_target_mdl_and_gt_path(args.fid_file, seed, args.which_data)
		print (gt_file)
		gt_df = pd.read_pickle(gt_file)
		gts_layer = gt_df.layer.values
		gts_weight = list(map(tuple, gt_df.w_idx.values))
		gts = list(zip(gts_layer, gts_weight)) # a list of [layer, index to a weight (np.ndarray)]
		##

		curr_loc_file = os.path.join(args.loc_dir, loc_file.format(seed))
		pairs = get_weight_and_cost(args.loc_which, seed, curr_loc_file, target_weights, gts, method = 'max')
		df = pd.DataFrame(list(pairs.items()))
		pairfile = os.path.join(dest, "{}.pairs.csv".format(seed))
		print (pairfile)	
		df.to_csv(pairfile, sep = ";", header = False, index = False)
		print (df)	
		print ("For {}".format(seed))
		ranks = []
		for gt in gts:
			#output = df.loc[list(map(comp, df[0].values, [gt]*len(df[0])))].index.values
			output = df.loc[list(map(comp, df[0].values, [gt]*len(df[0])))][1].values
			if len(output) > 0:
				rank = output[0]	
				ranks.append(rank)
				print ("GT ({}) {}: {}".format(seed, gt, rank))
			else:
				print ("GT ({}) {}: -".format(seed, gt))
	


