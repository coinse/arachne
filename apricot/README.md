## Apricot

### Preparation

We use the DNN repair technique [Apricot](https://2019.ase-conferences.org/details/ase-2019-papers/79/Apricot-A-Weight-Adaptation-Approach-to-Fixing-Deep-Learning-Models) as our baseline.

To run Apricot, one must first prepare the iDLM and rDLMs. 
To do so, one should decide on which CNN to use and edit the `iDLM_train.py` and `rDLM_train.py` files accordingly.
Then run:

```
mkdir weights rDLM_weights
python iDLM_train.py
python rDLM_train.py
```

Other hyperparameters can be changed by editing the python scripts. 

### Execution

Once the iDLM and rDLMs are prepared, one can run the `weight_adjust.py` script to improve the network using Apricot. 
One may provide the `-err_src` and `-err_dst` arguments to run our modified targeted version of Apricot.
If these arguments are not provided, the untargeted (original) version of Apricot will run. 

```
python weight_adjust.py -err_src 3 -err_dst 5
```