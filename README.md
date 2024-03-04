# Non-Parametric Estimation in Information Theory

## 1. Introduction

This is a repository for our paper on: "Evaluating Density- and Nearest Neighbor-based Methods to Accurately Estimate Information-Theoretic Quantities from Multi-Dimensional Sample Data".

The projects is organizes as follows:
```
├── analysis_results\
│   ├── plots\
├── data_evaluation\
│   ├── data\
│   ├── notebooks\
│   ├── results\
│   ├── utils\
│   ├── (...) scripts
├── data_generation\
├── README.md
└── .gitignore
```

## 2. Installation

Code was written in `Python 3.11.5` but should be compatible with later and earlier versions of Python down to `Python 3.6`.  Check the `requirements.txt` file for any dependency issues.

Usage is recommended by cloning the repository to a local directory and setting up the required environment using `venv` and `pip`:

```shell
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt
```

## 3. Generating Data

Initially data is generated and stored in the `data_evaluation/data` directory using the script in the `data_generation/` directory. The data for the experiments is stored as an HDF5 database.

From the root directory:
```python
    python data_generation/data_generation.py
```
**Note**: as the `data.hdf5` file is  ~123 GB, it is recommended to be locally generated. This process takes about ~12 hrs in an Intel Xeon E5-26280 v2 but shouldn't vary too much in any modern CPU. 

## 4. Conducting an Evaluation

The scripts in the directory `data_evaluation/` are used to read the data and perform the experiments. Results are stored in the `results/` directory.

Again, from the root directory:
```python
    python data_evaluation/eval_bin_entropy.py
```

All of the names of the scripts have the format `eval_{estimator}_{quantity}.py`. In total, 12 scripts must be run, tree for each estimator: binning, KDE, numerical integration of KDE and *k*-NN.

The `notebooks/` directory serves as an archive of the development of the workflow to test each estimator. The contents of each notebook are generally the same as the code in the scripts. Log files describe the history of the project.

## 5. Visualizing Results

The `analysis_results` directory contains a notebook to create the plots used in the paper, as well as a script to read the log files and calculate the time per iteration of the different experiments.

The plots are generated using the results from the `data_evaluation/results` directory. Results are read from `.hdf5` files.

## 5. Contact

All results produced using the [UNITE Toolbox](https://github.com/manuel-alvarez-chaves/unite_toolbox).

Contact: [manuel.alvarez-chaves@simtech.uni-stuttgart.de](manuel.alvarez-chaves@simtech.uni-stuttgart.de)
