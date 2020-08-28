#!/usr/bin/python

'''Loads data as numpy data form'''

import torch
import torchvision.datasets as dss
from torchvision import transforms
from torch.utils import data
import os
import numpy as np
from PIL import Image

class MNISTCustomWrapper(dss.MNIST): # should change class... when dataset changes
    """Custom Wrapper to constrict dataset size for MNIST-like datasets"""
    def __init__(self, **kwargs):
        self.num_restrict = kwargs['num_restrict']
        del kwargs['num_restrict']
        super(MNISTCustomWrapper, self).__init__(**kwargs)
        print(f'dataset start size: {len(self.targets)/10} per class')
        max_num_by_class = 5000 # len(self.targets)//10
        self.new_data = []
        self.new_targets = []
        for class_idx in range(10):
            class_img_idxs = (self.targets == class_idx)
            select_img_idxs = np.random.choice(max_num_by_class, size=self.num_restrict, replace=False)
            class_imgs = self.data[class_img_idxs]
            select_imgs = class_imgs[select_img_idxs]
            self.new_data.append(select_imgs)
            self.new_targets.append(class_idx*torch.ones(select_img_idxs.shape))
        self.new_data = torch.cat(self.new_data, dim=0)
        self.new_targets = torch.cat(self.new_targets, dim=0)
        print(f'dataset final size: {len(self.new_targets)/10} per class')
        
    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'processed')
        
    def __getitem__(self, index):
        img, target = self.new_data[index], int(self.new_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.new_data)
    
class CIFARCustomWrapper(dss.CIFAR10): # should change class... when dataset changes
    """Custom Wrapper to constrict dataset size for MNIST-like datasets"""
    def __init__(self, **kwargs):
        self.num_restrict = kwargs['num_restrict']
        del kwargs['num_restrict']
        super(CIFARCustomWrapper, self).__init__(**kwargs)
        print(f'dataset start size: {len(self.targets)/10} per class')
        max_num_by_class = 5000 # len(self.targets)//10
        self.new_data = []
        self.new_targets = []
        for class_idx in range(10):
            class_img_idxs = (np.array(self.targets) == class_idx)
            select_img_idxs = np.random.choice(max_num_by_class, size=self.num_restrict, replace=False)
            class_imgs = self.data[class_img_idxs]
            select_imgs = class_imgs[select_img_idxs]
            self.new_data.append(select_imgs)
            self.new_targets.append(class_idx*torch.ones(select_img_idxs.shape))
        self.new_data = np.vstack(self.new_data)
        self.new_targets = torch.cat(self.new_targets, dim=0)
        print(f'dataset final size: {len(self.new_targets)/10} per class')
    
        
    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'processed')
        
    def __getitem__(self, index):
        img, target = self.new_data[index], int(self.new_targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.new_data)
    
class CIFARVT(dss.CIFAR10): # should change class... when dataset changes
    """Custom Wrapper to constrict dataset size for MNIST-like datasets"""
    def __init__(self, **kwargs):
        self.val = kwargs['val']
        del kwargs['val']
        super(CIFARVT, self).__init__(**kwargs)
        print(f'dataset start size: {len(self.targets)/10} per class')
        self.new_data = []
        self.new_targets = []
        for class_idx in range(10):
            class_img_idxs = (np.array(self.targets) == class_idx)
            class_imgs = self.data[class_img_idxs]
            if self.val:
                select_imgs = class_imgs[:500]
            else:
                select_imgs = class_imgs[500:]
            self.new_data.append(select_imgs)
            self.new_targets.append(class_idx*torch.ones(500))
        self.new_data = np.vstack(self.new_data)
        self.new_targets = torch.cat(self.new_targets, dim=0)
        print(f'dataset final size: {len(self.new_targets)/10} per class')
        
    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__bases__[0].__name__, 'processed')
        
    def __getitem__(self, index):
        img, target = self.new_data[index], int(self.new_targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        """Returns the total number of image files."""
        return len(self.new_data)

