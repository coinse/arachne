import torch
import torch.nn as nn
import torch.nn.functional as F
   
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
                nn.Linear(128*4*4, 10),
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
                nn.Linear(10, 10),
            ),
        )
    
    def forward(self, x):
        out = self.conv_layers(x)
        out = torch.mean(torch.mean(out, dim=3), dim=2)
#         out = self.softmax(out)
        out = self.dense_layers(out)
        return out
