#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dss
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image
import pandas

class CelebAFolder(data.Dataset):
    """CelebA folder"""
    def __init__(self, root, split='train', transform=None, max_per_class=-1):
        """Initializes image paths and preprocessing module."""
        img_dir = os.path.join(root, split)
        self.img_names = os.listdir(img_dir)
        self.image_paths = list(map(lambda x: os.path.join(img_dir, x), self.img_names))
        
        celeba_attrs = pandas.read_csv('list_attr_celeba.txt', sep='\s+', header=1)
        self.img2gender = dict(celeba_attrs[['File_Name', 'Male']].to_dict('split')['data'])
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        image_label = 0 if self.img2gender[os.path.basename(image_path)] < 0 else 1
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_label
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)

def get_CelebA_loader(image_path, split, batch_size, num_workers = 2):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))
    ])
    
    dataset = CelebAFolder(image_path, split, transform=transform)
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    return data_loader

class LFWFolder(data.Dataset):
    """LFW folder"""
    def __init__(self, root, split='train', transform=None, max_per_class=-1, path_to_female_names = "./female_names_lfw.txt" ):
        """Initializes image paths and preprocessing module."""
        img_dir = os.path.join(root, split)
        self.image_names = os.listdir(img_dir)
        self.image_paths = list(map(lambda x: os.path.join(img_dir, x), self.image_names))
        
        with open(path_to_female_names, 'r') as f:
            self.female_names = set(map(lambda x: x.strip(), f.readlines()))
        
        self.transform = transform
        
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_name = self.image_names[index]
        image_path = self.image_paths[index]
        image_label = 0 if image_name in self.female_names else 1
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image_name, image, image_label
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.image_paths)

def get_LFW_loader(image_path, split, batch_size, num_workers = 0, path_to_female_names = "./female_names_lfw.txt"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                             std=(0.229, 0.224, 0.225))
    ])
    
    dataset = LFWFolder(image_path, split, transform=transform, path_to_female_names = path_to_female_names)
    data_loader = data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = False, #True, # b/c data index is used later
        num_workers = num_workers
    )
    return data_loader
