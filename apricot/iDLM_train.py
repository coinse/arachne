import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from model.model import ApricotCNN2
from model.data_loader import CIFARCustomWrapper

# create sample directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
img_size = 32
h_dim = 800
num_epochs = 5000
batch_size = 128
learning_rate = 1e-3
darc1_lambda = 1e-3
num_restrict = 5000

test_dataset = torchvision.datasets.CIFAR10(root='./',
                                     train=False,
#                                      split='test',
                                     transform=transforms.ToTensor(),
                                     download=False)

# Data loader

test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)

model = ApricotCNN2(img_size = img_size)
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

print(f'---Training iDLM---')
max_acc = 0
max_time = 0

for epoch in range(num_epochs):
    cumm_corr = 0
    cumm_eval = 0
    for i, (x, x_class) in enumerate(train_data_loader):
        # Forward pass
        x = x.cuda()
        class_logits = model(x)
        loss = loss_fn(class_logits, x_class.cuda())

        _, pred_class = torch.max(class_logits, dim=1)
        cumm_corr += torch.sum(pred_class == x_class.cuda())
        cumm_eval += x_class.size(0)

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
        model.train()
        print('train: %d/%d correct (%.2f%%)' % (cumm_corr, cumm_eval, 100*float(cumm_corr)/cumm_eval))
        test_acc = 100*float(correct)/total
        print("test: %d/%d correct (%.2f%%) | prev max %.2f%% @ %d epoch" % (correct, total, test_acc, max_acc, max_time))

    if test_acc > max_acc:
        max_acc = test_acc
        max_time = epoch
        torch.save(model.state_dict(), f'weights/CIFAR10_classifier_CNN2_iDLM_full.pth')
    if epoch-max_time > 100:
        break
