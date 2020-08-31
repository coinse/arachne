## Arachne

### Preparation

This document explains how to reproduce the six experiments performed to evaluate Arachne.

Models used in the experiments of Arachne can be downloaded through this link: https://www.dropbox.com/s/89sb9m6lf4ia2bh/models.tar.gz?raw=1
After downloading the zipped file, decompress them to *data* directory. 
*data* directory is where all data, including both model and image data are stored. 
```
tar -zxvf models.tar.gz -C  data
```
To prepare the dataset used in the experiments, download LFW dataset through this link (https://www.dropbox.com/s/zrgjb9ramgnc46i/lfw_np.tar.gz?raw=1) and unzip them under *data* directory. 
FashionMNIST and CIFAR-10 datasets are automatically download in the scripts if they do not exist under a specified data directory.  

### Arguments

This a list of arguments commonly used in the experimental scripts.

* *localisation_method*: denote a localisation method to use
  * *localiser*: our bidirectional localisation method (BL)
  * *gradient_loss*: gradient loss-based selection
  * *random* : random selection

* *which_data*: denote a dataset type
  * *fashion_mnist*: FashionMNIST
  * *cifar10*: CIFAR-10
  * *lfw*: LFW (facial image dataset)

* *which*: denote a type of model to repair
  * *cnn1*: CNN1 model used in Apricot
  * *cnn2*: CNN2 model used in Apricot
  * *cnn3*: CNN3 model used in Apricot
  * *simple_fm*: a simple DNN model trained on FashionMNIST dataset
  * *simple_cm*: a simple DNN model trained on CIFAR-10 dataset
  * *lfw_vgg*: a VGG16 model trained on LFW dataset

- *simple_fm* and *simple_fm* refer to models used for the experiments in RQs 1 to 4, and *lfw_vgg* denotes a model used for the experiment in RQ6. 
These models are automatically given in the scripts.

* *datadir*: a path to the data directory. The default value is *data*. 
	For CIFAR-10 and FashionMNIST, The dataset will be downloaded automatically if it does not exist under the given directory. 
	For LFW, this argument refers to a directory to which *lfw_np.tar.gz* was unzipped.  

* *dest*: a path to the directory where Arachne will store generated patches.
* *pkey*: a specific patch key to identify a generated patch. e.g., if `pkey=t1`, `model.t1.json`


### Experiment (repair)

To run an experiment for each RQ, 

For RQ1, 
```
./rq1.sh data/fm fashion_mnist $dest localisation_method
```

e.g., to localise the neural weights of a model trained on FashionMNIST dataset with our BL method, 
```
./rq1.sh data/fm fashion_mnist results/rq1 localiser
```

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
Here, *true_label* indicates the ground truth label of a input and *pred_label* refers to the predicted label of the input.
Thus, "*true_label = 3* and *pred_label = 5*" means Arachne will target to repair all misbehaviour that wrongly predict *3* to *5*.
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
./rq6.sh data/lfw/lfw_data/ t1 results/rq6
```

### Evaluation 

To apply the resulting patch to the target model and use this patched model to predict, ```run run_mdl.sh``` as below:

For RQs 2 to 4, (RQ = {rq2,rq3,rq4}, which_data = {fashion_mnist, cifar10})
```
./run_mdl.sh RQ path_to_patch datadir dest which_data
```
e.g., to evaluate a patch generated to repair *3->5* misbehaviour of a model trained on CIFAR-10 dataset in RQ3, 
```
./run_mdl.sh rq3 results/rq3/cm/model.misclf-t1-3-5.json data/cm/ results/rq3/pred cifar10
```

For RQ5, 
  * *which* = {*cnn1*, *cnn2*, *cnn3*} 
  * *path_to_validx_file* = a path to a file that stores the indices to test data used for the repair.\
    This index file is saved in the same directory where the patch is saved as default, \
    e.g., \
      for `results/rq5/cnn1/model.misclf-pkey-true_label-pred_label.json`, *path_to_validx_file* = `results/rq5/cnn1/cnn1.indices_to_val.csv`

```
./run_mdl.sh rq5 path_to_patch datadir dest which_data which path_to_validx_file
```
e.g., to evaluate a patch generated to repair *3->5* misbehaviour of CNN1 model, 
```./run_mdl.sh rq5 results/rq5/cnn1/model.misclf-t1-3-5.0.json data/cm/ results/rq5/cnn1/pred cifar10 cnn1 results/rq5/cnn1/cnn1.indices_to_val.csv```

For RQ6,
```
./run_mdl.sh rq6 path_to_patch datadir dest lfw 
```
e.g., \
```./run_mdl.sh rq6 results/rq6/model.misclf-t1-0-1.json data/lfw/lfw_data/ results/rq6/pred lfw```


### Results

#### Patch files

Either `model.pkey.json` (for a patch generated from multiple types of misbehaviour) \
 or `model.misclf-pkey-true_label-pred_label.json` (for a patch generated from a specific type of misbehaviour)
This json file stores a new value for the weight variable in the last layer of a model. (dictionary with key *weight*)

#### Prediction results

Either `pred.pkey.dtype.csv` (for a patch generated from multiple types of misbehaviour) \
 or `pred.misclf-pkey-true_label-pred_label.dtype.csv` (for a patch generated from a specific type of misbehaviour).
*dtype* can be *train*, *test*, and *val*(for RQ5,RQ6). 

The prediction result file is a csv file with a header row `[index,true,pred]`. 
Here, *index* refers to the index to an individual inputs in the dataset, and *true* and *pred* refer to a ground-truth and a predicted label of the input. 


