import os, sys
import utils.data_util as data_util
import numpy as np
from run_localise import compute_gradient_to_output

BATCH_SIZE = 5096

def return_chunks(num):
	num_split = int(np.round(num/BATCH_SIZE))
	if num_split == 0:
		num_split = 1
	chunks = np.array_split(np.arange(num), num_split)
	return chunks


def generate_base_mdl(mdl_path, X, indices_to_target = None, target_all = True, batch_size = None, act_func = None):
	from tensorflow.keras.models import load_model, Model 
	from gen_frame_graph import build_k_frame_model
	from run_localise import get_target_weights

	mdl = load_model(mdl_path, compile = False)
	target_weights = get_target_weights(mdl, mdl_path, 
		indices_to_target = indices_to_target, target_all = target_all)
	
	if batch_size is None:
		k_fn_mdl, _, _  = build_k_frame_model(mdl, X, list(sorted(target_weights.keys())))
		k_fn_mdl_lst = [k_fn_mdl]
	else:
		num = len(X)
		chunks = return_chunks(num)
		k_fn_mdl_lst = []
		for chunk in chunks:
			k_fn_mdl, _, _  = build_k_frame_model(mdl, X[chunk], list(sorted(target_weights.keys())), act_func = act_func)
			k_fn_mdl_lst.append(k_fn_mdl)
	return k_fn_mdl_lst, target_weights


def get_indices_to_grad_above_mean(path_to_keras_mdl, idx_to_tl, X):
	"""
	return the index to neural weights whose gradient is non-zero
	"""
	grad = np.abs(compute_gradient_to_output(path_to_keras_mdl, idx_to_tl, X, on_weight = True, wo_reset = True, by_batch = True))
	mean_grad_v = np.mean(grad.flatten())
	indices_to_above_mean = list(zip(*np.where(grad > mean_grad_v)))
	return indices_to_above_mean	


def random_sample_weights(path_to_keras_mdl, X, indices_to_target, num_sample = 1):
	"""
	"""
	cand_layers = indices_to_target
	if len(cand_layers) >= num_sample:
		sel_layers = np.random.choice(cand_layers, num_sample, replace = False)
	else:
		sel_layers = np.random.choice(cand_layers, num_sample, replace = True)

	selected_neural_weights = []
	for idx in sel_layers:
		indices_to_cand_ws = get_indices_to_grad_above_mean(path_to_keras_mdl, idx, X)
		#target_w = target_ws[idx][0]
		#curr_indices = list(np.ndindex(target_w.shape))
		sel_idx = indices_to_cand_ws[np.random.choice(np.arange(len(indices_to_cand_ws)), 1, replace = False)[0]]
		selected_neural_weights.append((idx, sel_idx))

	return selected_neural_weights


def inject_faults_for_keras_mdl(deltas_as_lst, mdl):
	"""
	"""
	
	for idx_to_tl, delta in deltas_as_lst:
		_, org_bias = mdl.layers[idx_to_tl].get_weights()
		mdl.layers[idx_to_tl].set_weights([delta, org_bias])
	return mdl


def is_in_bound(bound_lr, v):
	bound_l,bound_r = bound_lr
	return (bound_l <= v) and (bound_r >= v)


def compute_predictions(k_fn_mdl_lst, ys, target_weights, indices_to_tls):
	"""
	"""
	from functools import reduce
	if len(k_fn_mdl_lst) == 1:
		k_fn_mdl = k_fn_mdl_lst[0]
		pred_probas, _ = k_fn_mdl([target_weights[idx][0] for idx in indices_to_tls] + [ys])
	else:
		num = len(ys)
		chunks = return_chunks(num)
		pred_probas = None
		for k_fn_mdl, chunk in zip(k_fn_mdl_lst, chunks):
			a_pred_probas, _ = k_fn_mdl([target_weights[idx][0] for idx in indices_to_tls] + [ys[chunk]])
			if pred_probas is None:
				pred_probas = a_pred_probas
			else:
				pred_probas = np.append(pred_probas, a_pred_probas, axis = 0)
				#print ("aftr", pred_probas.shape)
	predictions = np.argmax(pred_probas, axis = 1)
	#print ("Final", predictions.shape)
	return predictions


def tweak_weights_v2(k_fn_mdl_lst, target_weights, ys, selected_neural_weights, by_v = 0.1, is_rd = False, test_mdl = None, test_ys = None):
	"""
	the criterion would be the changes
	"""
	import tensorflow as tf
	import time

	# initial prediction
	indices_to_sel_ws = np.asarray([vs[1] for vs in selected_neural_weights]) 
	indices_to_tls = sorted(list(target_weights.keys()))
	indices_to_sel_w_tls = np.asarray([vs[0] for vs in selected_neural_weights])
	indices_to_uniq_sel_w_tls = np.unique(indices_to_sel_w_tls)
	indices_to_uniq_sel_w_tls.sort()

	num_inputs = len(ys)
	init_predictions = compute_predictions(k_fn_mdl_lst, ys, target_weights, indices_to_tls)
	#print ("Init predcitions")
	#print (np.sum(init_predictions == np.argmax(ys,1))/len(ys))
	prev_predictions = np.copy(init_predictions)

	by = {(vs[0],tuple(vs[1])):by_v for vs in selected_neural_weights} # starting from here
	print ("By: {}".format(by))
	chg_limit = 0.001 #1 #0.001 #0005
	print ("\t{} number of inputs should be changed".format(num_inputs * chg_limit))
	### for testing
	#if test_mdl is not None: # to check the generalisability of the changes (can be removed later)
	#	test_init_pred_probas, _ = k_fn_mdl_test([target_weights[idx][0] for idx in indices_to_tls] + [test_ys])
	#	test_init_predictions = np.argmax(test_init_pred_probas, axis = 1)
	###

	# set direction -> -1 (decrese), +1 (increase)
	which_direction_arr = np.ones(len(selected_neural_weights))
	print ("Number of selected neural weights", len(selected_neural_weights))
	which_direction_arr[np.where(which_direction_arr > 0.5)[0]] = -1.
	which_direction = {(vs[0],tuple(vs[1])):d for vs,d in zip(selected_neural_weights, which_direction_arr)}

	org_weights = {idx_to_tl:np.copy(target_weights[idx_to_tl][0]) for idx_to_tl in indices_to_tls}

	bound_lr_vs = {}
	for idx_to_tl in indices_to_uniq_sel_w_tls:
		w = target_weights[idx_to_tl][0]
		std_v = np.std(w)
		mean_v = np.mean(w)
		bound_l = np.max([mean_v - 3 * std_v, np.min(w)]) #np.quantile(w, 0.25)])
		bound_r = np.min([mean_v + 3 * std_v, np.max(w)])  #np.quantile(w, 0.75)])
		bound_lr_vs[idx_to_tl] = [bound_l, bound_r]

	print ("Boundary", bound_lr_vs)
	t1 = time.time()
	timeout = 60 * 5 #10
	num_prev_chgd = 0
	is_out_of_bound = 0 # to count the consecutive out-of-bound cases 
	while True:
		t2 = time.time()
		if t2 - t1 > timeout:
			print ("Time out: {}".format(t2 - t1))
			return None, None, None

		deltas_as_lst = []
		deltas_of_snws = {"layer":[], "w_idx":[], "init_v":[], "new_v":[]} # store current result
		# update
		for idx_to_tl in indices_to_tls:
			init_weight, _ = target_weights[idx_to_tl]
			if idx_to_tl not in indices_to_uniq_sel_w_tls:
				deltas_as_lst.append(init_weight)
			else:
				w_stdev = np.std(org_weights[idx_to_tl])

				local_indices_to_sel_nws = list(zip(*np.where(indices_to_sel_w_tls == idx_to_tl))) 
				curr_indices_to_sel_nws = [indices_to_sel_ws[i] for i in local_indices_to_sel_nws]
				#delta = by[idx_to_tl] * w_stdev * np.random.rand(*init_weight.shape)
				delta = w_stdev * np.random.rand(*init_weight.shape)  

				for idx in curr_indices_to_sel_nws:
					k = (idx_to_tl, tuple(idx)) # key to a neural weight
					#
					deltas_of_snws['init_v'].append(org_weights[idx_to_tl][tuple(idx)])
					#print ("++",idx_to_tl, idx, init_weight[tuple(idx)], delta[tuple(idx)], which_direction[(idx_to_tl,tuple(idx))], org_weights[idx_to_tl][tuple(idx)])	
					if not is_rd:
						init_weight[tuple(idx)] += which_direction[k] * (delta[tuple(idx)] * by[k])
						## check whether a new value exceeeds the bound
						if not is_in_bound(bound_lr_vs[idx_to_tl], init_weight[tuple(idx)]):
							is_out_of_bound += 1
							#print ("out of bound: ", bound_lr_vs[idx_to_tl], init_weight[tuple(idx)])
							# go back to the previous value
							init_weight[tuple(idx)] -= which_direction[k] * (delta[tuple(idx)] * by[k])
							# reset the step size
							by[k] = by[k]/2 # decrease the step size
							which_direction[k] *= -1 # change the direction
						else:
							is_out_of_bound = 0 
					else:
						which_dir = -1. if np.random.rand(1)[0] > 0.5 else 1.
						#init_weight[tuple(idx)] = org_weights[idx_to_tl][tuple(idx)] + delta[tuple(idx)]*which_dir
						init_weight[tuple(idx)] = delta[tuple(idx)] * which_dir

					deltas_of_snws['layer'].append(idx_to_tl)
					deltas_of_snws['w_idx'].append(idx)
					deltas_of_snws['new_v'].append(init_weight[tuple(idx)])

				deltas_as_lst.append(init_weight)

		aft_predictions = compute_predictions(k_fn_mdl_lst, ys, target_weights, indices_to_tls)
		
		# compute the number of changed inputs
		num_chgd = np.sum(aft_predictions != prev_predictions)
		# for the test dataset
		#if test_mdl is not None:
		#	test_aft_pred_probas, _ = k_fn_mdl_test(deltas_as_lst + [test_ys])
		#	test_aft_predictions = np.argmax(test_aft_pred_probas, axis = 1)
		#	test_num_chgd = np.sum(test_init_predictions != test_aft_predictions) 		
		#	print ("The number of changes in the test data set: {} ({}/{}), {}%".format(test_num_chgd, test_num_chgd, len(test_ys), 100*test_num_chgd/len(test_ys)))

		if num_chgd >= num_inputs * chg_limit:
			print ('Success!')
			prev_corr_predictions = prev_predictions == np.argmax(ys, axis = 1)
			aft_corr_predictions = aft_predictions == np.argmax(ys, axis =1)
			num_broken = np.sum((prev_corr_predictions == 1) & (aft_corr_predictions == 0))
			num_patched = np.sum((prev_corr_predictions == 0) & (aft_corr_predictions == 1))
			print ("Total number of changes: {}\n\tNumber of broken: {}, number of patched: {}".format(num_chgd, num_broken, num_patched))
			num_aft_corr = np.sum(aft_corr_predictions)
			
			return list(zip(indices_to_tls, deltas_as_lst)), deltas_of_snws, num_aft_corr
		elif is_rd:
			print ("here", num_chgd, deltas_of_snws['init_v'],deltas_of_snws['new_v'])
			continue
		else:
			if is_out_of_bound > 10 * len(selected_neural_weights): # out of bound for more than 10 consecutive runs
				is_out_of_bound = 0
				# set to init weight
				for idx_to_tl in indices_to_tls:
					target_weights[idx_to_tl][0] = np.copy(org_weights[idx_to_tl])
				for k in by.keys():
					by[k] = by_v
			else: # num_prev == num_aft_corr (nothing has been changed)
				print ("here: {} -> {} ({})".format(num_prev_chgd, num_chgd, num_chgd - num_prev_chgd))
				if num_prev_chgd > num_chgd: # has been improved from the "previous" result (but, still below the initial results)
					print ("has been improved")
					for vs in selected_neural_weights:
						print ("Before", which_direction[(vs[0], tuple(vs[1]))])
						which_direction[(vs[0], tuple(vs[1]))] *= -1
						print ("After", which_direction[(vs[0], tuple(vs[1]))])
				num_prev_chgd = num_chgd
				for k in by.keys():	
					by[k] += by_v/10# by[k]/2
					print ("By", k, by[k])
					#if False:
					if by[k] > 3:
						print ("Out of the initial distribution: {}".format(by[k]))
						for idx_to_tl in indices_to_tls:
							target_weights[idx_to_tl][0] = np.copy(org_weights[idx_to_tl])
						# reverse
						which_direction = {tuple(vs):-1*d for vs,d in zip(selected_neural_weights, which_direction_arr)}
						#
						num_prev_chgd = 0
						by[k] = by_v*2
						print ("Increase by and start again", by[k])


if __name__ == "__main__":
	import argparse
	import pandas as pd
	import tensorflow as tf

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
	parser.add_argument("-rd", type = int, default = 0)
	parser.add_argument("-on_test", action = "store_true")

	args = parser.parse_args()

	np.random.seed(args.seed)

	num_label = args.num_label

	train_data, test_data = data_util.load_data(args.which_data, args.datadir, is_input_2d = args.which_data == 'fashion_mnist', with_hist = False)
	target_data = train_data if not args.on_test else test_data

	k_fn_mdl_lst, target_weights = generate_base_mdl(args.model_path, target_data[0],
		indices_to_target = None, target_all = bool(args.target_all), 
		batch_size = BATCH_SIZE if args.which_data == 'GTSRB' else None,
		act_func = tf.nn.relu if args.which_data == 'GTSRB' else None)

	num_sample = args.num_sample
	indices_to_target_layers = list(target_weights.keys())
	print ("Indices", indices_to_target_layers)

	if args.which_data == 'fashion_mnist':
		selected_neural_weights = random_sample_weights(args.model_path, target_data[0].reshape(target_data[0].shape[0],1,target_data[0].shape[-1]), indices_to_target_layers, num_sample = num_sample)
	else:
		selected_neural_weights = random_sample_weights(args.model_path, target_data[0], indices_to_target_layers, num_sample = num_sample)

	print ("Selected Neural Weights", selected_neural_weights)
	from collections import Iterable
	if not isinstance(target_data[1][0], Iterable):
		new_ys = data_util.format_label(target_data[1], num_label)
		new_ys_test = data_util.format_label(test_data[1], num_label)
	else:
		new_ys = target_data[1]

	deltas_as_lst, deltas_of_snws, num_aft_corr = tweak_weights_v2(
		k_fn_mdl_lst, target_weights, new_ys, selected_neural_weights, by_v = args.by_v, is_rd = bool(args.rd))
	
	if num_aft_corr is not None:
		print ("Changed Accuracy: {}".format(num_aft_corr/len(target_data[1])))
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

