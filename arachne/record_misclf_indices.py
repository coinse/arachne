"""
Record the initial predictions and misclassification (for RQ1,2,3,4,5)
"""
import os, sys
import argparse
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import numpy as np
import utils.data_util as data_util

is_input_2d = False 

parser = argparse.ArgumentParser()
parser.add_argument("-model", type = str)
parser.add_argument("-datadir", type = str)
parser.add_argument("-dest", type = str, default = ".")
parser.add_argument("-which_data", type = str, default = "cifar10", help = "cifar10, FM")
parser.add_argument("-is_train", type = int, default = 1)
parser.add_argument("-w_hist", type = int, default = 0)

args = parser.parse_args()

import torch
import torchvision
import torchvision.transforms as transforms

if args.which_data in ['fashion_mnist', 'cifar10']: #!= 'GTSRB':
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
elif args.which_data in ['GTSRB', 'imbd']: # gtsrb
	train_data, test_data = data_util.load_data(args.which_data, args.datadir, with_hist = bool(args.w_hist))
	if bool(args.is_train):
		X,y = train_data
	else:
		X,y = test_data

else:# will be deleted later
	train_data, test_data = data_util.load_data(args.which_data, args.datadir)
	if bool(args.is_train):
		X,y = train_data
	else:
		X,y = test_data

	from sklearn.preprocessing import MinMaxScaler
	import pandas as pd
	datafile = "data/lstm/airline-passengers.csv"
	dataframe = pd.read_csv(datafile, usecols=[1], engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')

	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(dataset)	


loaded_model = load_model(args.model)
loaded_model.summary()

if args.which_data in ['cifar10', 'GTSRB', 'imdb', 'simple_lstm']: # and also GTSRB
	predicteds = loaded_model.predict(X)
else:
	if is_input_2d:
		predicteds = loaded_model.predict(X)
	else:
		predicteds = loaded_model.predict(X).reshape(-1, 10)

print ("predicted shape", predicteds.shape)
if args.which_data != 'simple_lstm':
	if predicteds.shape[-1] > 1:
		pred_labels = np.argmax(predicteds, axis = 1)
	else:
		pred_labels = np.round(predicteds).flatten()
else: # might be deleted later
	pred_labels = scaler.inverse_transform(predicteds)	
	y = scaler.inverse_transform(y.reshape(-1,1))
	y = np.int32(np.round(y.reshape(-1,)))
	pred_labels = np.int32(np.round(pred_labels.reshape(-1,)))

os.makedirs(args.dest, exist_ok = True)

init_preds = [['index', 'true', 'pred']]
misclfs = [['index','true','pred']]
cnt = 0
for i, (true_label, pred_label) in enumerate(zip(y, pred_labels)):
	if args.which_data != 'simple_lstm':
		if pred_label != true_label:
			misclfs.append([i,true_label,pred_label])
	else:
		if np.abs(pred_label - true_label) > 30:
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

