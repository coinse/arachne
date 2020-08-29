## Arachne

### Preparation

Before reproducing the experiements of Arachne, we first need to prepare the target models, either by 
generating new models based on the description about ... or unzip the ```models.zip``` under ```data```.
This document explains how to reproduce the six experiments in Arachne.

```
unzip models.zip data/
```


### Execution

To run each experiment for each RQ, 

```

```


To apply a patch to the model and use this patched model to predict data, ```run run_mdl.sh``` as below:

For RQs 2 to 4, ( RQ = {rq2,rq3,rq4}, which_data = {fashion_mnist, cifar10})
```
./run_mdl.sh RQ path_to_patch datadir dest which_data
```
e.g.,
```
./run_mdl.sh rq3 results/rq3/cm/model.misclf-t1-3-5.0.json data/cm/ results/rq3
```

For RQ5, \
( which = {cnn1,cnn2,cnn3}, \
  path_to_validx_file = a path to a file that stores the indices to test data used for the repair; 
  this file is saved under the same directory with a patch (```model*.json```) as default ( e.g., ```results/rq5/cnn1/cnn1.indices_to_val.csv```)

```
./run_mdl.sh rq5 path_to_patch datadir dest cifar10 which path_to_validx_file
```
e.g., 
```./run_mdl.sh rq5 results/rq5/cnn1/model.misclf-t1-3-5.0.json data/cm/ results/rq5/cnn1/pred cifar10 cnn1 results/rq5/cnn1/cnn1.indices_to_val.csv```

For RQ6,
```
./run_mdl.sh rq6 path_to_patch datadir dest lfw 
```
e.g., \
```./run_mdl.sh rq6 results/rq6/model.misclf-t1-0-1.0.json data/lfw/lfw_data/ results/rq6/pred lfw```


