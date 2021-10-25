# BEMPS

This is the code repository for the NeurIPS (2021) paper: Diversity Enhanced Active Learning with StrictlyProper Scoring Rules. 
We provide non-batch and batch mode active learning approaches named BEMPS (Bayesian estimate of mean proper scores) based on the ELR framework.


Please cite our paper by using the bibex below:
```cite

```


### Installation
1. Create virtual environment with Python 3.7 +
2. Run following commands:

```
git clone https://github.com/davidtw999/BEMPS
cd BEMPS
pip install -r requirements.txt
```


### Folder information

+ src: source code 
+ scripts: scripts for running experiments 
+ data: folder for datasets 
+ models: saved models from running experiments

### Run active learning experiment

To run the default experiment on SST-5 dataset, type the command below:

`$ bash scripts/run_experiment.sh`

The model will be saved in 'models' directory.
Results will be saved in 'results.txt'.

You can modify the variables in scripts/run_experiment.sh for other active learning methods.

#### Notes:

Download the other datasets from online resources before run the script.


