## Arachne

### Preparation

Before reproducing the experiements of Arachne, we first need to prepare the target models, either by 
generating new models based on the description about ... or unzip the ```models.zip``` under ```data```.
This document explains how to reproduce the six experiments in Arachne.

```
unzip models.zip data/
```

### Arguments

This a list of arguments commonly used in the experiement scripts.
* *which*: denote a type of model.
  * *cnn1*: CNN1 model used in Apricot
  * *cnn2*: CNN2 model used in Apricot
  * *cnn3*: CNN3 model used in Apricot
  
* *which_data*: denote a dataset type
  * *fashion_mnist*: FashionMNITS 
  * *cifar10*: CIFAR-10
  * *lfw*: LFW (facial image dataset)
  
* *datadir*: a path to the data directory. The dataset will be downloaded automatically if it does not exist under the given directory. 

* *dest*: a path to the directory to save generated patches.
* *pkey*


### Execution

To run an experiment for each RQ, 
Belwo

For RQ2, 
```
./rq2.sh datadir which_data localisation_method pkey dest
```
``localisation_method`` can be *localiser* (our method *BL*), *gradient_loss*, and *random*.

e.g.,
```
./rq2.sh data/fm fashion_mnist localiser t1 results/rq2/loc
```

For RQ3,
```
./rq3.sh which_data datadir pkey temp/rq3/
```
e.g.,
```
./rq3.sh cifar10 data/cm t1 temp/rq3/
```


For RQ4,
```
./rq4.sh which_data datadir pkey dest
```
e.g., 
```
./rq4.sh cifar10 data/cm/ t1 results/rq4/cm
```

For RQ5, 
```
./rq5.sh which datadir pkey resultdir true_label predicted_label
```
Here, *true*

One can repair *3->5* misbehaviour of CNN1 by running the script below:
e.g.,
```
./rq5.sh cnn1 data/cm t1 results/rq5/cnn1 3 5
```

For RQ6,
```
./rq6.sh datadir pkey dest
```

e.g., 
```
./rq6.sh data/lfw/lfw_data/ t1 temp
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


