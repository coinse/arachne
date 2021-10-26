"""
A module for data related functions
"""
import numpy as np
import random

def divide_into_val_and_test(X, y, num_label = 10, is_val = True, half_n = 500):
	"""
	divide test data into two folds
	"""
	new_indices = []
	
	if half_n <= 0: 
		cnts = {}
		for ay in y:
			if ay not in cnts.keys():
				cnts[ay] = 0
			cnts[ay] += 1
		
		assert len(cnts.keys()) == num_label, "{} vs {}".format(len(cnts.keys()), num_label)
	else:
		cnts = None

	for class_idx in range(num_label):
		class_img_idxs = np.where(np.asarray(y) == class_idx)[0]
		if half_n > 0: 
			if is_val:
				new_indices.extend(list(class_img_idxs[:half_n]))
			else:
				new_indices.extend(list(class_img_idxs[half_n:]))
		else:
			if is_val:
				new_indices.extend(list(class_img_idxs[:int(cnts[class_idx]/2)]))
			else:
				new_indices.extend(list(class_img_idxs[int(cnts[class_idx]/2):]))

	new_X = X[new_indices]
	new_y = y[new_indices]

	return new_X, new_y, new_indices


def read_tensor_name(tensor_name_file):
	"""
	"""
	import os
	assert os.path.exists(tensor_name_file), "%s does not exist" % (tensor_name_file)
	tensor_names = {}
	with open(tensor_name_file) as f:
		lines = [line.strip() for line in f.readlines()]
		for line in lines:
			terms = line.split(",")
			assert len(terms) >= 2, "%s should contain at least two terms" % (line)
			tensor_key = terms[0]
			tensor_names[tensor_key] = terms[1] if len(terms) == 2 else terms[1:]

	return tensor_names


def get_lfw_data(is_train = True):
	"""
	"""
	import pickle
	if is_train:
		datafile = "lfw_np/LFW_train_info.pkl"
	else:
		datafile = "lfw_np/LFW_test_info.pkl"
	
	with open(datafile, 'rb') as f:
		data = pickle.load(f)
	
	names = sorted(list(data.keys()))
	at_arr = np.asarray([data[n]['at'] for n in names])
	true_label_arr = np.asarray([data[n]['true'] for n in names]) 
	pred_label_arr = np.asarray([data[n]['pred'] for n in names])

	return {'name':names, 'at':at_arr, 'true':true_label_arr, 'pred':pred_label_arr}


def load_data(which, path_to_data, 
	is_input_2d = False, 
	path_to_female_names = None, 
	with_hist = True):
	import tensorflow as tf
	import os
		
	assert which in ['mnist','cifar10','cifar100','fashion_mnist', 'GTSRB', 'lfw'], which

	if which == 'lfw':
		from utils.data_loader import get_LFW_loader

		names_in_train = sorted(get_lfw_data(is_train = True)['name'])
		names_in_test = sorted(get_lfw_data(is_train = False)['name'])

		trainloader = get_LFW_loader(image_path = path_to_data,
			split='train', batch_size=1, path_to_female_names = path_to_female_names)
		testloader = get_LFW_loader(image_path = path_to_data,
			split='test', batch_size=1, path_to_female_names = path_to_female_names)

		train_vs = {}
		for image_names, images, image_labels in trainloader:
			train_vs[image_names[0]] = [images.numpy()[0], image_labels.item()]
		#
		train_data = [[],[]]
		sorted_train_vs = [train_vs[n] for n in names_in_train]
		train_data[0] = [vs[0] for vs in sorted_train_vs]
		train_data[1] = [vs[1] for vs in sorted_train_vs]
		
		test_data = [[],[]]
		test_vs = {}
		for image_names, images, image_labels in testloader:
			test_vs[image_names[0]] = [images.numpy()[0], image_labels.item()]
		#
		sorted_test_vs = [test_vs[n] for n in names_in_test]
		test_data[0] = [vs[0] for vs in sorted_test_vs]
		test_data[1] = [vs[1] for vs in sorted_test_vs]
	elif which != 'GTSRB':
		import torch
		import torchvision
		import torchvision.transforms as transforms

		if which == 'fashion_mnist':
			trainset = torchvision.datasets.FashionMNIST(root=path_to_data, train=True,
				download=True, transform=transforms.ToTensor())
			testset = torchvision.datasets.FashionMNIST(root=path_to_data, train=False,
				download=True, transform=transforms.ToTensor())
		else:
			trainset = torchvision.datasets.CIFAR10(root=os.path.join(path_to_data, "train"), train=True,
				download=True, transform=transforms.ToTensor())
			testset = torchvision.datasets.CIFAR10(root=os.path.join(path_to_data, "test"), train=False,
				download=True, transform=transforms.ToTensor())

		trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
		testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

		train_data = [[],[]]
		
		for data in trainloader:
			images, labels = data
			if which == 'fashion_mnist':
				if is_input_2d: # for RQ1s
					train_data[0].append(images.numpy()[0].reshape(-1,))
				else:
					train_data[0].append(images.numpy()[0].reshape(1,-1))
			else:
				train_data[0].append(images.numpy()[0])
			train_data[1].append(labels.item())

		train_data[0] = np.asarray(train_data[0])
		train_data[1] = np.asarray(train_data[1])

		test_data = [[],[]]
		for data in testloader:
			images, labels = data
			if which == 'fashion_mnist': 
				if is_input_2d: # for RQ1
					test_data[0].append(images.numpy()[0].reshape(-1,))
				else:
					test_data[0].append(images.numpy()[0].reshape(1,-1))
			else:
				test_data[0].append(images.numpy()[0])
			test_data[1].append(labels.item())

		test_data[0] = np.asarray(test_data[0])
		test_data[1] = np.asarray(test_data[1])
	else: # GTSRB
		#which, path_to_data,
		import pickle
		# train
		if with_hist:
			path_to_data = os.path.join(path_to_data, 'hist')

		with open(os.path.join(path_to_data, "train_data.pkl"), 'rb') as f:
			train_data_dict = pickle.load(f)

		#train_data = [np.moveaxis(train_data_dict['data'], [1], [-1]), train_data_dict['label']]
		train_data = [train_data_dict['data'], train_data_dict['label']]
		# test
		with open(os.path.join(path_to_data, "test_data.pkl"), 'rb') as f:
			test_data_dict = pickle.load(f)

		#test_data = [np.moveaxis(test_data_dict['data'], [1], [-1]), test_data_dict['label']]
		test_data = [test_data_dict['data'], test_data_dict['label']]

	return (train_data, test_data)

	
def format_label(labels, num_label):
	"""
	format label which has a integer label to flag type
	e.g., [3, 5] -> [[0,0,1,0,0],[0,0,0,0,1]]
	"""
	num_data = len(labels)
	from collections.abc import Iterable

	new_labels = np.zeros([num_data, num_label])
	for i, v in enumerate(labels):
		if isinstance(v, Iterable):
			new_labels[i,v[0]] = 1
		else:		
			new_labels[i,v] = 1

	return new_labels


def get_indexed_data(indices, datas):
	"""
	return an array of data with indioces
	"""
	new_data = [datas[i] for i in indices]
	return np.asarray(new_data)


def split_into_wrong_and_correct(correct_predictions):
	"""
	Spilt wrong and correct classification result
	Ret (dict):
		ret = {'wrong':indices_to_wrong(list), 'correct':indices_to_correct(list)}
	"""
	indices = {'wrong':[], 'correct':[]}
	
	indices_to_wrong, = np.where(correct_predictions == False)
	indices_to_correct, = np.where(correct_predictions == True)
	indices['correct'] = list(indices_to_correct)
	indices['wrong'] = list(indices_to_wrong)

	return indices


def split_into_chgd_and_unchgd():
	"""
	"""
	pass


def get_misclf_indices(misclf_indices_file, target_indices = None, use_all = True):
	"""
	"""
	import pandas as pd 

	misclfds = {}
	df = pd.read_csv(misclf_indices_file)
	for i,t,p in zip(df['index'], df['true'], df['pred']):
		if target_indices is not None:
			if i in target_indices:
				key = (t,p)
				if key not in misclfds.keys():
					misclfds[key] = []
				misclfds[key].append(target_indices.index(i))
		else:
			if use_all or i % 2 == 0:
				key = (t,p)
				if key not in misclfds.keys():
					misclfds[key] = []
				misclfds[key].append(i)

	return misclfds


def get_misclf_indices_balanced(df, idx = 0):
	"""
	"""
	misclf_types = np.unique(df[["true","pred"]].values, axis = 1)
	ret_misclfds = {}
	
	for misclf_type in misclf_types:
		misclf_type = tuple(misclf_type)
		true_label, pred_label = misclf_type
		indices_to_misclf = df.loc[
			(df.true == true_label) & (df.pred == pred_label)].index.values
		
		#np.random.shuffle(indices_to_misclf)
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


def gen_data_for_rq3(misclf_indices_file, top_n, idx = 0):
	"""
	"""
	import pandas as pd
	
	idx = idx if idx == 0 else 1 # only 0 or 1
	target_idx = idx; eval_idx = np.abs(1 - target_idx)
	
	df = pd.read_csv(misclf_indices_file, index_col = 'index')
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
		for sorted_k in sorted_keys:
			new_data_indices.extend(misclfds_idx_target[sorted_k])
			new_test_indices.extend(misclfds_idx_eval[sorted_k])
			
		new_data_indices += indices_to_corr_target
		new_test_indices += indices_to_corr_eval

		np.random.shuffle(new_data_indices)
		np.random.shuffle(new_test_indices)	

		return (misclf_key, misclf_indices, new_data_indices, new_test_indices)
	else:
		return len(sorted_keys)




