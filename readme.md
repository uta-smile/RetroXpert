## Reference implementation of our NeurIPS2020 paper [RetroXpert: Decompose Retrosynthesis Prediction Like A Chemist](https://arxiv.org/pdf/2011.02893.pdf) 

The final results may slightly different from those reported in the paper. We are continuously improving the implementation after the acceptance. 
For example, directed edges are adopted in the implementation, and previous implementation requires both directed edges are correctly predicted to be disconnected for each bond disconnection.
We later change the implementation slightly to combine the predictions for each bond. This improves the reaction center prediction accuracy slightly, but can not improve the final accuracy.

## Conda environment
We recommend to new a Conda environment to run the code. We use Pytorch 1.3, DGL 0.4.2, OpenNMT-py 1.0.0, and Rdkit 2019.03.4.0. It should be okay to use the latest Rdkit version with some slight changes accordingly. Please refer to the *requirements.txt* file for detailed packages.

Please refer to the OpenNMT-py/README.md for how to install the openmnt package. 

## Step-1: Product bond disconnection prediction

In the main directory (*~/RetroXpert/*)

1. Run data preprocessing, this will preprocess the USPTO-50K dataset to prepare required labels and DGL graphs.
```
python preprocessing.py
```
2. Extract semi-template patterns.
```
# extract semi-tempaltes for training data
python extract_semi_template_pattern.py --extract_pattern

# find semi-template patterns for all data
python extract_semi_template_pattern.py
```
3. Start to train EGAT model with reaction category
```
python train.py --typed
```

4. Evaluate the EGAT model
```
# evaluate on test data
python train.py --typed --test_only  --load

# evaluate on training data
python train.py --typed --test_on_train --load

```

## Postprocessing step-1 results before step-2 

1. Gegerate formatted dataset for OpenNMT.

```
python prepare_data.py --typed
```


3. Generate synthons for test data according to the bond disconnection prediction in step-1, to do 2-step direct prediction:
```
python prepare_test_prediction.py  --typed
```


4. Generate step-1 train error data to augment training data in step-2:
```
python prepare_train_error_aug.py  --typed
```


## Ready to train step-2 models


1. In the **script** directory (*~/RetroXpert/OpenNMT-py/script/USPTO50K-aug-typed/*), first preprocess the data.
   To speed up the training process, we may duplicate the training data so that less dataset loads are required.
   Please check the *~/RetroXpert/OpenNMT-py/data/USPTO50K-aug-typed/copy_files.py* for details.

```
bash preprocss.sh
```

2. Start to train the model:
```
bash train.sh
```

3. Processing the checkpoints in the checkpoint directory ( *~/RetroXpert/OpenNMT-pyexperiments/checkpoints/USPTO50K-aug-typed/* ) and run the script:
```
bash average_models.sh
```


4. Translate
```
bash translate.sh
```

5. Score the prediction
```
bash score.sh
```


