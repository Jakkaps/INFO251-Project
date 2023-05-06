# Aircraft Project

For a full report of all the work done in the project, please refer to the submitted project report.

## Pre-requisites

- `pip`
- `python 3.x`. Also make sure the `python3` command points to this installation.
- `wget` to download artifacts. These can be generated yourself, however this will require considerable time and computational resources.

## Set Up

Run

```bash
$ chmod +x setup.sh && ./setup.sh --download
```

This creates a virtual environment and installs all required packages.

In addition, to save on time for the reviewer, it also downloads all required model weights and other artifacts. As a result, all analysis can be run without first training a model. If you don't want to download artifacts, simply run the command without the downoad flag.

## Usage

The project consists of 4 parts:
- `train`: Runs a full training run using the NTS model.
  - In order to train the model, simply run the notebook `train.ipynb`. This will save the weights of the best epoch. Hyperparameters can be configured by editing the file `hyperparameters.json`
- `test`: Runs a suite of tests of the NTS model on the test set and displays how the model performed in accordance to different performance metrics.
  - For testing the model on the test set, run the cells in the notebook `test.ipynb`. *NOTE*: the model have to be trained first as the test notebook will use the weights output from the `train.ipynb` notebook.
- `tuning`: The hyperparameters of the model was tuned using the notebook `bayesian_sweep.ipynb` as a template, which allowed for different hyperparameter-configurations to be evaluated through **wandb (weights and biases)**
- `data_augmentations`: Runs analysis on how the different data augmentations affect performance of the NTS model. Results are shown in popups.
  - To run the data_augmentation script, use
    ```bash
    $ ./data_augmentations.sh
    ```
