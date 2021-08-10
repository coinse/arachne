import os, sys
import utils.data_util as data_util
import numpy as np

def generate_base_mdl(mdl_path, X, indices_to_target = None, target_all = True):
	from tensorflow.keras.models import load_model, Model 
	from gen_frame_graph import build_k_frame_model
	from run_localise import get_target_weights

	mdl = load_model(mdl_path)
	target_weights = get_target_weights(mdl, mdl_path, 
		indices_to_target = indices_to_target, target_all = target_all)
	
	print (target_weights.keys())
	k_fn_mdl, _, _  = build_k_frame_model(mdl, X, list(sorted(target_weights.keys())))
	return k_fn_mdl, target_weights


def random_sample_weights(target_ws, indices_to_target, num_sample = 1):
	"""
	"""
	indices = []
	for idx in indices_to_target:
		target_w = target_ws[idx][0]
		curr_indices = list(np.ndindex(target_w.shape))
		curr_indices = list(zip([idx]*len(curr_indices), curr_indices))
		
		indices.extend(curr_indices)

	sel_indices = np.random.choice(np.arange(len(indices)), num_sample, replace = False)
	selected_neural_weights = [indices[i] for i in sel_indices]
	return selected_neural_weights


def inject_faults_for_keras_mdl(deltas_as_lst, mdl):
	"""
	"""
	
	for idx_to_tl, delta in deltas_as_lst:
		_, org_bias = mdl.layers[idx_to_tl].get_weights()
		mdl.layers[idx_to_tl].set_weights([delta, org_bias])
	return mdl


def tweak_weights(k_fn_mdl, target_weights, ys, selected_neural_weights, by_v = 0.1):
	"""
	"""
	import tensorflow as tf
	import time

	# initial prediction
	indices_to_tls = sorted(list(target_weights.keys()))
	init_predictions, _ = k_fn_mdl([target_weights[idx][0] for idx in indices_to_tls] + [ys])
	init_corr_predictions = np.argmax(init_predictions, axis = 1)
	init_corr_predictions = init_corr_predictions == np.argmax(ys, axis = 1)

	indices_to_sel_w_tls = np.asarray([vs[0] for vs in selected_neural_weights])
	indices_to_uniq_sel_w_tls = np.unique(indices_to_sel_w_tls)
	indices_to_uniq_sel_w_tls.sort()
	indices_to_sel_ws = np.asarray([vs[1] for vs in selected_neural_weights]) 

	num_inputs = len(ys)
	prev_corr_predictons = init_corr_predictions
	num_init_corr = np.sum(prev_corr_predictons)
	num_prev_corr = num_init_corr
	by = by_v # starting from here
	print ("By: {}".format(by))
	chg_limit = 0.

	which_direction_arr = np.ones(len(selected_neural_weights))
	print ("Number of selected neural weights", len(selected_neural_weights))
	which_direction_arr[np.where(which_direction_arr > 0.5)[0]] = -1.
	which_direction = {tuple(vs):d for vs,d in zip(selected_neural_weights, which_direction_arr)}

	org_weights = {idx_to_tl:np.copy(target_weights[idx_to_tl][0]) for idx_to_tl in indices_to_tls}
	t1 = time.time()
	timeout = 60 * 5
	while True:
		t2 = time.time()
		if t2 - t1 > timeout:
			print ("Time out: {}".format(t2 - t1))
			return None, None, None

		deltas_as_lst = []
		deltas_of_snws = {"layer":[], "w_idx":[], "init_v":[], "new_v":[]}
		# update
		for idx_to_tl in indices_to_tls:
			init_weight, _ = target_weights[idx_to_tl]
			if idx_to_tl not in indices_to_uniq_sel_w_tls:
				deltas_as_lst.append(init_weight)
			else:
				w_stdev = np.std(init_weight)
				w_mean = np.mean(init_weight)
				local_indices_to_sel_nws = list(zip(*np.where(indices_to_sel_w_tls == idx_to_tl))) 
				curr_indices_to_sel_nws = [indices_to_sel_ws[i] for i in local_indices_to_sel_nws]
				delta = by * w_stdev * np.random.rand(*init_weight.shape) 
				for idx in curr_indices_to_sel_nws:
					deltas_of_snws['init_v'].append(org_weights[idx_to_tl][tuple(idx)])
					#print ("++",idx_to_tl, idx, init_weight[tuple(idx)], delta[tuple(idx)], which_direction[(idx_to_tl,tuple(idx))], org_weights[idx_to_tl][tuple(idx)])
					init_weight[tuple(idx)] += which_direction[(idx_to_tl,tuple(idx))] * delta[tuple(idx)]
					#which_dir = -1. if np.random.rand(1)[0] > 0.5 else -1.
					#init_weight[tuple(idx)] = org_weights[idx_to_tl][tuple(idx)] + delta[tuple(idx)]*which_dir
					#print ("++", init_weight[tuple(idx)], delta[tuple(idx)]*which_direction[(idx_to_tl,tuple(idx))])
					deltas_of_snws['layer'].append(idx_to_tl)
					deltas_of_snws['w_idx'].append(idx)
					deltas_of_snws['new_v'].append(init_weight[tuple(idx)])

				deltas_as_lst.append(init_weight)

		aft_predictions, _ = k_fn_mdl(deltas_as_lst + [ys])
		aft_corr_predictions = np.argmax(aft_predictions, axis = 1)
		aft_corr_predictions = aft_corr_predictions == np.argmax(ys, axis = 1)

		# check whehter the accuracy decreases
		num_aft_corr = np.sum(aft_corr_predictions)
		#print ("--", num_init_corr - num_aft_corr, (num_init_corr - num_aft_corr)/num_inputs, (num_init_corr - num_aft_corr)/num_inputs > chg_limit)
		if num_init_corr - num_aft_corr > 0: #num_inputs * chg_limit:
			print ("Accuracy has been decreased: {} -> {}".format(num_prev_corr/num_inputs, num_aft_corr/num_inputs))
			num_broken = np.sum((prev_corr_predictons == 1) & (aft_corr_predictions == 0))
			num_patched = np.sum((prev_corr_predictons == 0) & (aft_corr_predictions == 1))
			print ("\tNumber of broken: {}, number of patched: {}".format(num_broken, num_patched))
			return list(zip(indices_to_tls, deltas_as_lst)), deltas_of_snws, num_aft_corr
		else:
			if num_init_corr < num_aft_corr: # fix 
				print ("Has been improved instead: {} -> {}".format(num_init_corr/num_inputs, num_aft_corr/num_inputs))
				# set to init weight
				for idx_to_tl in indices_to_tls:
					target_weights[idx_to_tl][0] = np.copy(org_weights[idx_to_tl])
					
				for vs in selected_neural_weights:
					which_direction[tuple(vs)] *= -1
			
				num_prev_corr = num_init_corr
		
			else: # num_prev == num_aft_corr (nothing has been changed)
				print ("here", num_prev_corr - num_aft_corr, num_init_corr - num_aft_corr, by)
				if num_prev_corr > num_aft_corr:
					for vs in selected_neural_weights:
						which_direction[tuple(vs)] *= -1	
				num_prev_corr = num_aft_corr
				by += by_v/2
				if by > 3:
					print ("Out of the initial distribution: {}".format(by))
					if by > 4.5:
						for idx_to_tl in indices_to_tls:
							target_weights[idx_to_tl][0] = np.copy(org_weights[idx_to_tl])
					
						# reverse
						which_direction = {tuple(vs):-1*d for vs,d in zip(selected_neural_weights, which_direction_arr)}
		
						num_prev_corr = num_init_corr
						by = by_v*2
						print ("Increase by and start again", by)



if __name__ == "__main__":
	import argparse
	import pandas as pd

	parser = argparse.ArgumentParser()
	parser.add_argument("-datadir", type = str)
	parser.add_argument("-dest", type = str)
	parser.add_argument("-which_data", type = str, help = "fashion_mnist, cifar10, GTSRB")
	parser.add_argument("-model_path", type = str)
	parser.add_argument("-target_all", type = int, default = 1)
	parser.add_argument("-seed", type = int, default = 0)
	parser.add_argument("-num_label", type = int, default = 10)
	parser.add_argument("-by_v", type = float, default = 0.1)
	parser.add_argument("-num_sample", type = int, default = 1)

	args = parser.parse_args()

	np.random.seed(args.seed)

	num_label = args.num_label

	train_data, test_data = data_util.load_data(args.which_data, args.datadir)

	k_fn_mdl, target_weights = generate_base_mdl(args.model_path, train_data[0], 
		indices_to_target = None, target_all = bool(args.target_all))

	num_sample = args.num_sample
	indices_to_target_layers = list(target_weights.keys())
	selected_neural_weights = random_sample_weights(target_weights, indices_to_target_layers, num_sample = num_sample)

	from collections import Iterable
	if not isinstance(train_data[1][0], Iterable):
		new_ys = data_util.format_label(train_data[1], num_label)
	else:
		new_ys = train_data[1]

	deltas_as_lst, deltas_of_snws, num_aft_corr = tweak_weights(
		k_fn_mdl, target_weights, new_ys, selected_neural_weights, by_v = args.by_v)
	
	print ("Changed Accuracy: {}".format(num_aft_corr/len(train_data[1])))
	deltas_of_snws = pd.DataFrame.from_dict(deltas_of_snws)
	print (deltas_of_snws)

	dest = os.path.join(args.dest, "{}/{}".format(args.which_data, num_sample))
	os.makedirs(dest, exist_ok = True)
	destfile = os.path.join(dest, "faulty_nws.{}.pkl".format(args.seed))
	print ("Saved to {}".format(destfile))
	
	deltas_of_snws.to_pickle(destfile)
	
	# model save
	from tensorflow.keras.models import load_model
	mdl = load_model(args.model_path, compile = False)
	new_mdl = inject_faults_for_keras_mdl(deltas_as_lst, mdl)
	
	mdl_key = os.path.basename(args.model_path)[:-3]
	mdl_destfile = os.path.join(dest, "{}_seed{}.h5".format(mdl_key, args.seed))

	print ("Saved to {}".format(mdl_destfile))
	new_mdl.save(mdl_destfile)
	


