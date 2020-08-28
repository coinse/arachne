import torch
import torch.nn as nn
import torch.nn.functional as F

'''Simple classifier for SA'''

class EasyClassifier(nn.Module):
    def __init__(self, img_size=28**2, h_dim=200):
        super(EasyClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(img_size, h_dim),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.layer3 = nn.Linear(h_dim, 10)
    
    def up2lyr1(self, x):
        return self.layer1(x)
    
    def up2lyr2(self, x):
        return self.layer2(self.layer1(x))
    
    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

class MnistClassifier(nn.Module):
    '''Same architecture as in original SA paper'''
    def __init__(self, img_size=32):
        super(MnistClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(3136, 512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 10), # regularization only on linear
            ),
        )
    
    def at_by_layer(self, x, layer_idx = 0):
        current_idx = 0
        out = x
        
        if layer_idx == -1:
            out = out.view(-1, 28*28)
            return out # special case handling
        
        for lyr in self.conv_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                at = torch.mean(out, dim=3)
                at = torch.mean(at, dim=2)
                return at
            else:
                current_idx += 1
        out = out.view(-1, 64*7*7)
        if current_idx == layer_idx:
            return out
        else:
            current_idx += 1
        for lyr in self.dense_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                return out
            else:
                current_idx += 1
        # should never reach here
        raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
    
    def layer_num(self):
        return 9
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 64*7*7)
        out = self.dense_layers(out)
        return out
    
class CifarClassifier(nn.Module):
    '''Same architecture as in original SA paper'''
    def __init__(self, img_size=32):
        super(CifarClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(2048, 1024),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(1024, 512), # regularization only on linear
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 10), # regularization only on linear
            ),
        )
    
    def at_by_layer(self, x, layer_idx = 0):
        current_idx = 0
        out = x
        for lyr in self.conv_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                at = torch.mean(out, dim=3)
                at = torch.mean(at, dim=2)
                return at
            else:
                current_idx += 1
        out = out.view(-1, 128*4*4)
        if current_idx == layer_idx:
            return out
        else:
            current_idx += 1
        for lyr in self.dense_layers:
            out = lyr(out)
            if current_idx == layer_idx:
                return out
            else:
                current_idx += 1
        # should never reach here
        raise ValueError('layer_idx value %d failed match with layers' % layer_idx)
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 128*4*4)
        out = self.dense_layers(out)
        return out
    
class ApricotCNN1(nn.Module):
    '''Same architecture as in original SA paper'''
    def __init__(self, img_size=32):
        super(ApricotCNN1, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Sequential(
#                 nn.Dropout(),
                nn.Linear(2048*4, 256),
                nn.ReLU()
            ),
            nn.Sequential(
#                 nn.Dropout(),
                nn.Linear(256, 256), # (dropout not in paper)
                nn.ReLU()
            ),
            nn.Sequential(
#                 nn.Dropout(),
                nn.Linear(256, 10),
            ),
        )
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 512*4*4)
        out = self.dense_layers(out)
        return out
    
class ApricotCNN2(nn.Module):
    '''Same architecture as in original SA paper'''
    def __init__(self, img_size=32):
        super(ApricotCNN2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Sequential(
#                 nn.Dropout(),
                nn.Linear(128*4*4, 10), # regularization only on linear
            ),
        )
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 128*4*4)
        out = self.dense_layers(out)
        return out
    
class ApricotCNN3(nn.Module):
    '''Same architecture as in original SA paper'''
    def __init__(self, img_size=32):
        super(ApricotCNN3, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 96, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(96, 96, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(96, 96, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(96, 192, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(192, 192, 3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(192, 192, 3, padding=1),
                nn.ReLU()
            ),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(192, 10, 3, padding=1),
                nn.ReLU()
            ),
        )
        self.softmax = nn.Softmax(dim=1)
        self.dense_layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 10), # regularization only on linear
            ),
        )
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = torch.mean(torch.mean(out, dim=3), dim=2)
#         out = self.softmax(out)
        out = self.dense_layers(out)
        return out