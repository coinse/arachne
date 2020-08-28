import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from model.model import ApricotCNN3
from model.data_loader import CIFARCustomWrapper

# create sample directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
img_size = 32
h_dim = 800
num_epochs = 60
batch_size = 128
learning_rate = 2e-4
darc1_lambda = 1e-3
num_restrict = 1000
rDLM_num = 20

test_dataset = torchvision.datasets.CIFAR10(root='./',
                                     train=False,
#                                      split='test',
                                     transform=transforms.ToTensor(),
                                     download=False)

# Data loader

test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)

for rdlm_idx in range(rDLM_num):
    model = ApricotCNN3(img_size = img_size)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    train_dataset = CIFARCustomWrapper(root='./',
                                         train=True,
                                         transform=transforms.ToTensor(),
                                         num_restrict = num_restrict,
                                         download=False)
    
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size, 
                                                    shuffle=True)

    print(f'---Training rDLM #{rdlm_idx}---')
    for epoch in range(num_epochs):
        for i, (x, x_class) in enumerate(train_data_loader):
            # Forward pass
            x = x.cuda()
            class_logits = model(x)

            # Backprop and optimize
            class_loss = loss_fn(class_logits, x_class.cuda())
#             darc1_loss = torch.max(torch.sum(torch.abs(class_logits), dim=0))
#             darc1_loss = (darc1_lambda / x.size(0)) * darc1_loss
            loss = class_loss #+ darc1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], CE Loss: {:.4f}"
                       .format(epoch+1, num_epochs, i+1, len(train_data_loader), loss.item()))

    with torch.no_grad():
        total = 0.
        correct = 0.
        model.eval()
        for tx, tx_class in test_data_loader:
            tx = tx.cuda()
            tclass_logits = model(tx)
            _, mostprob_result = torch.max(tclass_logits, dim=1)
            total += tx.size(0)
            correct += torch.sum(mostprob_result == tx_class.cuda())
        print("rDLM #%d acc: %d/%d correct (%.2f %%)" % (
            rdlm_idx, correct, total, (100.*correct.item())/total
        ))
        model.train()
        
#     torch.save(model.state_dict(), f'weights/CIFAR10_classifier_CNN3_iDLM.pth')
    torch.save(model.state_dict(), f'rDLM_weights/CIFAR10_classifier_CNN3_rDLM{rdlm_idx}.pth')