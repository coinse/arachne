## Arachne

### Preparation

This document is to replicate the experiments conducted to evaluate Arachne. 

Models used in the experiments of Arachne can be downloaded through this link: https://www.dropbox.com/s/p5n91wtc7b4s6b8/models.tar.gz?dl=0 <br />
After downloading the zipped file, please decompress them to *final_data* directory where all the models as well as image <br />
and text datasets used in the experiements will be stored. <br />

``` tar -zxvf models.tar.gz -C  data ```

The datasets used in the experiments can be download through this link: https://www.dropbox.com/s/i7ttm3iqyur2usk/data.tar.gz?dl=0 <br />
The downloaded *data.tar* contains the GTSBR, LFW, and Twitter US Airline Sentiment datasets preprocessed for the experiements, <br />
and the corresponding index files that contain predictions results along with the indices to each datapoint. <br />
Please unzip this file under *final_data* directory.  FashionMNIST and CIFAR-10 datasets will be automatically downloaded when they are first accessed. <br />


### Experiements 

To run an experiment for each RQ, 

#### RQ1

__usage__ <br />
```
./rq1.sh $datadir $loc_method $which_data $dest
```

`$loc_method` denotes a localisation stratgey to use to locate neural weights. `$loc_method` can be among one of those below: <br />

  * *localiser*: bidirectional localisation method (BL) (ours)
  * *gradient_loss*: gradient loss-based selection
  * *random*: random selection

`$which_data` denotes a datatype; it can be among: 'cifar10', 'fashion_mnist', 'GTSRB'.<br />
The output of the localisation will be saved under `$dest/$loc_method`.

__example__ <br />
To localise the neural weights of a model trained on FashionMNIST dataset (stored under final_data/data) with our BL method, 

```
./rq1.sh final_data/data localiser fashion_mnist results/rq1
```

#### RQ2

__usage__ <br />
```
./rq2.sh $datadir $loc_method $which_data $dest
```

`$loc_method` can be *localiser* (our method *BL*), *gradient_loss*, and *random*. <br />
The generated patch will be saved under *$dest*, and the localised neural weights will be under *$dest/loc* <br />
`$which_data`: the same with RQ1. <br />

__example__ <br />
To repair a model trained on FashionMNIST dataset while using BL for the neural weight localisation,  

```
./rq2.sh final_data/data localiser fashion_mnist results/rq2
```


*From RQ3 to RQ7, by default, Arachne uses BL for the weight localisation.*

#### RQ3 
__usage__ <br />
```
./rq3.sh $datadir $which_data $dest $top_n
```

To repair the top N'th frequent faults, please set `$top_n=N-1`. <br />
`$which_data`: the same with RQ1. <br />

__example__ <br />
To repair the most frequent faults in a FashionMNIST model,  

```
./rq3.sh final_data/data fashion_mnist $dest 0
```

#### RQ4 
__usage__ <br />
```
./rq4.sh $datadir $which_data $dest $patch_aggr 
```

`$patch_aggr` denotes the hyperparameter $alpha$ that balances between correcting given misbehaviour of a model <br />
and retaining initially correct behaviour. The greater it is, the more focus Arachne puts in the correction. <br />
By default, it will correct the most frequent type of errors. <br />


__example__ <br />
To repair the FashionMNIST model with the dafault value of $alpha$ (=10), 

```
./rq4.sh final_data/data fashion_mnist $dest 10
```

#### RQ5

__usage__ <br />
```
./rq5.sh $datadir $which $dest
```

`$which` denotes the model type. it can be among: *cnn1*, *cnn2*, *cnn3*, *GTSRB*, *fm* (fashion_mnist). <br />
By default, it will correct the most frequent type of errors. <br />

__example__ <br />
To repair *cnn1* model, 

```
./rq5.sh final_data cnn1 $dest
```

#### RQ6

__usage__ <br />

```
./rq6.sh $datadir $dest 
```

This will repair a gender classification model trained on the LFW dataset for the most frequent type of errors (female to male) <br />

__example__ <br />
```
./rq6.sh final_data/data $dest
```

#### RQ7

__usage__ <br />
```
./rq7.sh $datadir $dest 
```
This will repair a text sentiment analysis model trained on the Twitter US Airline Sentiment dataset <br />
for the most frequent type of errors (neutral to negative). <br />

__example__ <br />
```
./rq7.sh final_data/data $dest
```


### Evaluation  

To apply the resulting patch to a target model and use this patched model to predict, please run ```run run_mdl.sh``` as below: <br />

__usage__ <br />
```
./run_mdl.sh $rq $datadir $which_data $path_to_patch $top_n $which
```

`$rq` denotes the research question number, which can be between 1 to 7. <br />
`$which_data` denotes the type of data and can be among: *cifar10*, *fashion_mnist*, *GTSRB*, *fm_for_rq5*, *lfw*, *us_airline*. <br />
For RQs 5 to 7, this value will be automatically determined; thus, passing "" should be enough. <br />
`$path_to_patch` is the path to a patch under evaluation. <br />
`$top_n` denotes the top N'th frequent faults. This should be the same *top_n* used to generate `$path_to_patch`. <br />
`$which` is a model type. This is only required for RQ5; thus, for the other RQs, it can be skipped. <br />

__example__ <br />
To evaluate a patch generated to repair the most frequent errors of *cnn1* (*3->5*) (RQ5), 

```
./run_mdl.sh 5 final_data/data cifar10 $dest/model.misclf-rq5.0.0-3-5.pkl 0 cnn1 
```

### Results

#### Patch files

Either`model.misclf-rq#.$seed-$true_label-$pred_label.pkl` (for a patch generated from a specific type of misbehaviour (RQs 3 to 7)) <br />
or `model.rq2.$seed.pkl` (for a patch generated from multiple types of misbehaviour (RQ2)). <br />
This *pkl* file contains a new value for the weight variable from which at least one neural weight has been selected. <br />

#### Prediction results

The prediction files (i.e., the output of running `run_mdl.sh`) are stored under *pred*, a directory under the same directory <br />
where the patches were saved (`$dest`). A prediction file contains a dataframe of the combined results of intial prediction <br />
and the prediction after applying the patch. It has four columns: *true*, *pred*, *new_pred*, *init_flag*. <br />
*true* is the true label. *pred* and *new_pred* indicate a prediction result before and after the patch. *init_flag* is whether an intial prediction <br />
is correct (True) or not (False). <br />


The prediction result file is a csv file with a header row `[index,true,pred]`. <br />
Here, *index* refers to the index to an individual inputs in the dataset, and *true* and *pred* refer to a ground-truth and <br /> 
a predicted label of the input. <br />


