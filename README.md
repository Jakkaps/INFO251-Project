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

The project contains 3 parts, each with it's own script to run

- `train.sh` - Runs a full training run using the NTS model.
- `test.sh` - Runs a suite of tests of the NTS model on the test set.
- `analysis.sh` - Runs a more thourough error analysis on the trained NTS model. Refer to the submitted project report for a full exploration of this analysis.
