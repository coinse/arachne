"""
Record the initial predictions and misclassification (for RQ1,2,3,4,5)
"""
import os, sys
import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import numpy as np

is_input_2d = False 

parser = argparse.ArgumentParser()
parser.add_argument("-model", type = str)
parser.add_argument("-datadir", type = str)
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-which_data", type = str, default = "cifar10", help = "cifar10, FM")
parser.add_argument("-is_train", type = int, default = 1)

args = parser.parse_args()

import torch
import torchvision
import torchvision.transforms as transforms

if args.which_data != 'GTSRB':
	if args.which_data == 'cifar10':
		if bool(args.is_train):
			dataset = torchvision.datasets.CIFAR10(root=args.datadir, train=True,
				download=True, transform=transforms.ToTensor())
		else: # test
			dataset = torchvision.datasets.CIFAR10(root=args.datadir, train=False,
				download=True, transform=transforms.ToTensor())
	elif args.which_data == 'fashion_mnist':
		if bool(args.is_train):
			dataset = torchvision.datasets.FashionMNIST(root=args.datadir, train=True,
				download=True, transform=transforms.ToTensor())
		else: # test
			dataset = torchvision.datasets.FashionMNIST(root=args.datadir, train=False,
				download=True, transform=transforms.ToTensor())

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
	X = []; y = []
	for data in dataloader:
		images, labels = data
		if args.which_data == 'cifar10':
			X.append(images.numpy()[0])
		else:
			if is_input_2d:
				X.append(images.numpy()[0].reshape(-1,)) # since (1,x,x,x)
			else:
				X.append(images.numpy()[0].reshape(1,-1))
		y.append(labels.item())

	X = np.asarray(X)
	y = np.asarray(y)
else: # gtsrb
	import pickle
	if bool(args.is_train):
		with open(os.path.join(args.datadir, "train_data.pkl"), 'rb') as f:
			data = pickle.load(f)
	else:
		with open(os.path.join(args.datadir, "test_data.pkl"), 'rb') as f:
			data = pickle.load(f)

	X = data['data']
	y = data['label']		

loaded_model = load_model(args.model)
loaded_model.summary()

if args.which_data in ['cifar10', 'GTSRB']: # and also GTSRB
	predicteds = loaded_model.predict(X)
else:
	if is_input_2d:
		predicteds = loaded_model.predict(X)
	else:
		predicteds = loaded_model.predict(X).reshape(-1, 10)

pred_labels = np.argmax(predicteds, axis = 1)
os.makedirs(args.dest, exist_ok = True)

init_preds = [['index', 'true', 'pred']]
misclfs = [['index','true','pred']]
cnt = 0
for i, (true_label, pred_label) in enumerate(zip(y, pred_labels)):
	if pred_label != true_label:
		misclfs.append([i,true_label,pred_label])
	init_preds.append([i,true_label,pred_label])
	if true_label == pred_label:
		cnt += 1
	
import csv
filename = os.path.join(args.dest, "{}.misclf.indices.csv".format(args.which_data))
with open(filename, 'w') as f:
	csvWriter = csv.writer(f)
	for row in misclfs:
		csvWriter.writerow(row)

# all initial preditions	
filename = os.path.join(args.dest, "{}.init_pred.indices.csv".format(args.which_data))
with open(filename, 'w') as f:
	csvWriter = csv.writer(f)
	for row in init_preds:
		csvWriter.writerow(row)

