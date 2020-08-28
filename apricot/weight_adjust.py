import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from model.model import ApricotCNN3
from model.data_loader import CIFARVT
import tqdm
import time
import argparse

# create sample directory if not exists
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    
# Hyper-parameters

parser = argparse.ArgumentParser()
parser.add_argument('-err_src', type=int, default=-1)
parser.add_argument('-err_dst', type=int, default=-1)
args = parser.parse_args()

img_size = 32
h_dim = 800
train_batch_size = 20
test_batch_size = 256
retrain_epoch_num = 5
learning_rate = 2e-4
ticker = 'CIFAR10_classifier_CNN3_iDLM_full'
idlm_path = './weights/%s.pth' % ticker
rdlm_dir = './rDLM_weights/'
rdlm_query_str = 'CNN3'
device = 'cuda'
if args.err_src != -1 and args.err_dst != -1:
    targeted = True
    target_err_type = (args.err_src, args.err_dst)
else:
    targeted = False

# MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='./',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
val_dataset = CIFARVT(root='./',
                                     train=False,
                                     val=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
test_dataset = CIFARVT(root='./',
                                     train=False,
                                     val=False,
                                     transform=transforms.ToTensor(),
                                     download=False)

# Data loader
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=train_batch_size, 
                                                shuffle=True)
val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=test_batch_size, 
                                               shuffle=False)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=test_batch_size, 
                                               shuffle=False)

# neural net initialization
iDLM = ApricotCNN3(img_size = img_size)
iDLM.load_state_dict(torch.load(idlm_path))
iDLM.to(device)
iDLM.eval()
optimizer = torch.optim.Adam(iDLM.parameters(), lr=learning_rate)
rDLMs = []
for filename in os.listdir(rdlm_dir):
    if rdlm_query_str not in filename: continue
    rDLM = ApricotCNN3(img_size = img_size)
    rDLM.load_state_dict(torch.load(rdlm_dir + filename))
    rDLM.to(device)
    rDLM.eval()
    rDLMs.append(rDLM)
loss_fn = nn.CrossEntropyLoss()

# utility function(s)
def setWeights(model, weight_list):
    for weight, (name, v) in zip(weight_list, model.named_parameters()):
        attrs = name.split('.')
        obj = model
        for attr in attrs:
            obj = getattr(obj, attr)
        obj.data = weight

def AdjustWeights(baseWeights, corrDiff, incorrDiff, a, b,
                  strategy='both-org', lr=1e-3):
    # haha, should I implement this as it is in the paper? or not? 
    # Honestly I just can't stand this shit. These... how can this be accepted?
    if 'org' in strategy:
        sign = 1
    else:
        sign = -1
    
    p_corr, p_incorr = a/(a+b), b/(a+b)
    
    if 'both' in strategy:
        return [b_w + sign*lr*(p_corr*cD - p_incorr*iD)
                for b_w, cD, iD in zip(baseWeights, corrDiff, incorrDiff)]
    elif 'corr' in strategy:
        return [b_w + sign*lr*p_corr*cD
                for b_w, cD in zip(baseWeights, corrDiff)]
    elif 'incorr' in strategy:
        return [b_w - sign*lr*p_incorr*iD
                for b_w, iD in zip(baseWeights, incorrDiff)]
    else:
        raise ValueError(f'Unrecognized strategy {strategy}')

def EvaluateModel(model, loader=val_data_loader):
    with torch.no_grad():
        total = 0.
        correct = 0.
        model.eval()
        for tx, tx_class in loader:
            tx = tx.to(device)
            tclass_logits = model(tx)
            _, mostprob_result = torch.max(tclass_logits, dim=1)
            total += tx.size(0)
            correct += torch.sum(mostprob_result == tx_class.to(device))
    return correct.item()/total

def TrainingModel(model, epoch_num=retrain_epoch_num):
    # I mean this is really crazy. 
    # Train your model every batch? wut?
    model.train()
    for epoch in tqdm.trange(epoch_num):
        for i, (x, x_class) in enumerate(train_data_loader):
            # Forward pass
            x = x.to(device)
            class_logits = model(x)

            # Backprop and optimize
            class_loss = loss_fn(class_logits, x_class.to(device))
            loss = class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    

# --- actual process --- #
bestAcc = EvaluateModel(iDLM)
with torch.no_grad():
    best_weights = list(map(lambda x: x.data, iDLM.parameters()))

t_acc = EvaluateModel(iDLM, loader=test_data_loader)
print(f'org best acc (val): {bestAcc} | (tst): {t_acc}')
last_improvement = 0

s = time.clock()
print(f'Start time: {s}')

for b_idx, (x, x_class) in enumerate(train_data_loader):
    # Forward pass
    with torch.no_grad():
        x, x_class = x.to(device), x_class.to(device)
        yOrigin = torch.argmax(iDLM(x), dim=1)
        ySubList = [torch.argmax(rdlm(x), dim=1) for rdlm in rDLMs]

        for i_idx in range(train_batch_size):
            img_class = x_class[i_idx]
            if yOrigin[i_idx] == img_class:
                continue
            elif targeted and not (img_class, yOrigin[i_idx]) == target_err_type:
                continue
            else:
                correctSubModels = [rdlm for r_idx, rdlm in enumerate(rDLMs)
                                    if ySubList[r_idx][i_idx] == img_class]
                incorrectSubModels = [rdlm for r_idx, rdlm in enumerate(rDLMs)
                                      if ySubList[r_idx][i_idx] != img_class]
                if len(correctSubModels) == 0 or len(incorrectSubModels) == 0:
                    continue # slightly different from paper
                
                correctWeightSum = [sum(t) for t in zip(*[m.parameters() for m in correctSubModels])]
                correctWeights = [e/len(correctSubModels) for e in correctWeightSum]
                incorrWeightSum = [sum(t) for t in zip(*[m.parameters() for m in incorrectSubModels])]
                incorrWeights = [e/len(incorrectSubModels) for e in incorrWeightSum]
                baseWeights = list(map(lambda x: x.data, iDLM.parameters()))
                corrDiff = [b_w - c_w for b_w, c_w in zip(baseWeights, correctWeights)]
                incorrDiff = [b_w - i_w for b_w, i_w in zip(baseWeights, incorrWeights)]
                baseWeights = AdjustWeights(
                    baseWeights, corrDiff, incorrDiff,
                    len(correctSubModels), len(incorrectSubModels),
                    strategy='both-org', lr=1e-3
                )
                setWeights(iDLM, baseWeights)
    
    currAcc = EvaluateModel(iDLM)
    print(f'batch {b_idx} done, last improvement {b_idx-last_improvement} batches ago')
    print(f'New accuracy prior training: {currAcc}')
    if bestAcc < currAcc:
        with torch.no_grad():
            best_weights = list(map(lambda x: x.data, iDLM.parameters()))
        
        t_acc = EvaluateModel(iDLM, test_data_loader)
        print(f'new best acc (val): {currAcc} | (tst): {t_acc}')
        bestAcc = currAcc
        last_improvement = b_idx
    else:
        if bestAcc != currAcc:
            with torch.no_grad():
                setWeights(iDLM, best_weights)
        if last_improvement + 100 < b_idx:
            print('No improvement for too long, terminating')
            break
    TrainingModel(iDLM)
    currAcc = EvaluateModel(iDLM)
    print(f'New accuracy post training: {currAcc}')

e = time.clock()
print(f'End time: {e}')
print(f'Total execution time: {e-s}')
    
setWeights(iDLM, best_weights)
torch.save(iDLM.state_dict(), f'weights/%s_fixed_tgt%d%d.pth' % (ticker, target_err_type[0], target_err_type[1]))
