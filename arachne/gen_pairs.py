import os, sys
import numpy as np
from numpy.core.fromnumeric import argmax, argsort
import pandas as pd
import pickle

def get_weight_and_cost(loc_which, seed, loc_file, target_weights, method = 'max'):
	"""
	"""
	from scipy.stats import rankdata
	
	with open(loc_file, 'rb') as f:
		locs = pickle.load(f)
	
	if loc_which == 'localiser':
		new_locs = []
		for loc in locs:
			indices, cost = loc
			idx_to_l, local_idx = indices
			local_idx = np.unravel_index(local_idx, target_weights[idx_to_l][0].shape)
			new_locs.append([[idx_to_l, local_idx], cost])
		
		costs = [vs[1] for vs in new_locs]
		ranks = rankdata(costs, method = method)
		indices = [vs[0] for vs in new_locs]
		
		#print (len(ranks), len(indices))
		ret_lst = compute_pareto(np.asarray(costs), indices)
		
		pairs = {}
		for i,r in enumerate(ret_lst):
			#rint (indices[i])
			pairs[r] = i+1 # 1 ~ num
			
	elif loc_which == 'gradient_loss':
		costs = [-vs[-1] for vs in locs]
		ranks = rankdata(costs, method = method)
		indices = [vs[0] for vs in locs]
		
		pairs = {}
		for i,r in enumerate(ranks):
			pairs[tuple(indices[i])] = r
	else:
		np.random.seed(seed)
		nindices = np.arange(len(locs))
		np.random.shuffle(nindices)
		
		pairs = {}
		for i,aloc in enumerate(nindices):
			pairs[tuple(locs[aloc])] = i+1 # 1 ~ num
	
	return pairs

def compute_pareto(costs, curr_nodes_to_lookat):
	ret_lst = []
	ret_front_lst = []
	 
	while len(curr_nodes_to_lookat) > 0:
		_costs = costs.copy()
		is_efficient = np.arange(costs.shape[0])
		next_point_index = 0 # Next index in the is_efficient array to search for

		while next_point_index < len(_costs):
			nondominated_point_mask = np.any(_costs > _costs[next_point_index], axis=1)
			nondominated_point_mask[next_point_index] = True
			is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
			_costs = _costs[nondominated_point_mask]
			next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
			
		current_ret = [tuple(v) for v in np.asarray(curr_nodes_to_lookat)[is_efficient]]
		ret_lst.extend(current_ret)
			
		ret_front_lst.append(current_ret)
		# remove selected items (non-dominated ones)
		curr_nodes_to_lookat = np.delete(curr_nodes_to_lookat, is_efficient, 0)
		costs = np.delete(costs, is_efficient, 0)
	
	return ret_lst

import argparse
from run_localise import get_target_weights
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-which_data", type = str, help = "fashion_mnist, cifar10, GTSRB")
parser.add_argument("-loc_which", default = 'localiser', type = str, help = "localiser, gradient_loss, random")
parser.add_argument("-loc_dir", type = str)

args = parser.parse_args()


loc_loc_file = "loc.all_cost.loc.{}.1.pkl"
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
gtsrb_mdl_path = "data/models/GTSRB/gtsrb.model.0.wh.0.h5"

if args.which_data == 'cifar10':
	target_weights = get_target_weights(None, c10_mdl_path, indices_to_target = None, target_all = True)
elif args.which_data == 'fashion_mnist':
	target_weights = get_target_weights(None, fm_mdl_path, indices_to_target = None, target_all = True)
else:
	target_weights = get_target_weights(None, gtsrb_mdl_path, indices_to_target = None, target_all = True)

	
dest = os.path.join(args.loc_dir, "pairs/{}".format(args.loc_which))
os.makedirs(dest, exist_ok = True)

for seed in tqdm.tqdm(range(30)):
	curr_loc_file = os.path.join(args.loc_dir, loc_file.format(seed))
	pairs = get_weight_and_cost(args.loc_which, seed, curr_loc_file, target_weights, method = 'max')
	df = pd.DataFrame(list(pairs.items()))
	pairfile = os.path.join(dest, "{}.pairs.csv".format(seed))
	
	df.to_csv(pairfile, sep = ";", header = False, index = False)