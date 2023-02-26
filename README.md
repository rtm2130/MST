# MST

This is a Python code base for training "Market Segmentation Trees" (MSTs) (formerly known as "Model Trees for Personalization" (MTPs)). MSTs provide a general framework for jointly performing market segmentation and response modeling. The folder `/scripts` contains the code relative to the case study on Swiss Metro data. Currently, this repo only supports Python 2.7 and does not support Python 3.

Link to paper: [https://arxiv.org/abs/1906.01174](https://arxiv.org/abs/1906.01174)

"mst.py" contains the Python class "MST" used for training MSTs. This class supports the following methods:
* `__init__()`: initializes the MST
* `fit()`: trains the MST on data: contexts X, decisions P (labeled as A in this code base), responses Y
* `traverse()`: prints out the learned MST
* `prune()`: prunes the tree on a held-out validation set to prevent overfitting
* `predict()`: predict response distribution given new contexts X and decisions P

Users can build their own response models for the MST class by filling in the template "leaf_model_template.py".

Two examples of MSTs are provided here:
(1) "Isotonic Regression Model Trees" (IRTs): Here, the response models are isotonic regression models. The "irt_exmaple.py" file provides an example of running IRTs on a synthetic dataset.
(2) "Choice Model Trees" (CMTs): Here, the response models are MNL choice models. The "cmt_exmaple.py" file provides an example of running CMTs on a synthetic dataset. 

For significantly faster CMT fitting, users can combine this repo with the MNL fitting code found at https://github.com/rtm2130/CMT-R. Please see the CMT-R repo for further instructions. Note that the CMT-R repo is under a GPLv2 license, which is more restrictive on terms of use than this repo's MIT license. The case study on the Swiss Metro data from our paper was run using the code from the CMT-R repo.

## Package Installation

Here we provide guidance on installing the MST package dependencies excluding the files located in the CMT-R repo (https://github.com/rtm2130/CMT-R). For the complete installation instructions including the CMT-R files, see the README.md file of the CMT-R repo.

### Prerequisites

First, clone the MST repo. This can be done through opening a command prompt / terminal and typing: `git clone https://github.com/rtm2130/MST.git` (if this command does not work then install git).

Install the conda command-line tool. This can be accomplished through installing miniforge, miniconda, or anaconda. We advise users to consult the license terms of use for these tools because as of 2023-02-26 miniconda and anaconda are not free for commercial use.

Open a command prompt / terminal and execute the following steps:
1. Update conda: `conda update -n base -c defaults conda`
2. Install the conda-forge channel into conda: `conda config --add channels conda-forge`

### Installing Package Dependencies

In this step, we will be creating a new conda virtual environment called `mstenv` which will contain Python 2.7.15 and the package dependencies. Open a command prompt / terminal and execute the steps below.

1. Build a new MST virtual environment which will be named mstenv with the recommended Python version: `conda create --name mstenv python=2.7.15`
2. Activate the newly-created MST virtual environment: `conda activate mstenv`. All subsequent steps should be followed within the activated virtual environment. 
3. Install the scikit-learn and joblib packages. Execute the following: `conda install scikit-learn`, `conda install -c anaconda joblib`
4. Install tensorflow ensuring compatibility with python 2.7. The following worked for us: `pip install --upgrade tensorflow`
5. Deactivate the environment: `conda deactivate`. Going forward, users should activate their MST virtual environment prior to working with the code in this repo via `conda activate mstenv`.

## Running the Package Demos / Testing Installation

To test the package installation or demo the package, users can take the following steps:
1. Open command prompt / terminal and navigate into the MST directory
2. Activate the MST virtual environment: `conda activate mstenv`
3. We will first demo MST's implementation of Choice Model Trees. Open mst.py. At the top of the file under "Import proper leaf model here:" , ensure that only one leaf model is being imported which should read `from leaf_model_mnl import *`. In command prompt / terminal, execute command `python cmt_example.py` which will run the MST on a synthetic choice modeling dataset. At the end of execution, the test set error will be outputted which should be under 0.05.
5. We will next demo MST's implementation of Isotonic Regression Trees. Open mst.py. At the top of the file under "Import proper leaf model here:" , ensure that only one leaf model is being imported which should read `from leaf_model_isoreg import *`. In command prompt / terminal, execute command `python irt_example.py` which will run the MST on a synthetic ad auction dataset. At the end of execution, the test set error will be outputted which should be under 0.05.

## Running MSTs on the Swiss Metro dataset
To run MSTs on the Swiss Metro dataset used by our paper, please take the following steps:
1. Copy the files leaf_model_mnl_tensorflow.py and mst.py from this repo to the /scripts/src directory
2. Copy the files newmnlogit.R, leaf_model_mnl.py, and leaf_model_mnl_rmnlogit.py from the https://github.com/rtm2130/CMT-R repo to the /scripts/src directory
3. Create and activate a virtual environment using the environment_mst.yml file from the https://github.com/rtm2130/CMT-R repo
4. Open the /scripts/src/newmnlogit.R file and at the top of the file, change `ro.r.source("newmnlogit.R")` to `ro.r.source("src/newmnlogit.R")`
5. In /scripts/src/leaf_model_mnl_rmnlogit.py, within the implementation of the `error(self,A,Y)` function, change `log_probas = -np.log(Ypred[(np.arange(Y.shape[0]),Y)])` to `log_probas = -np.log(np.maximum(Ypred[(np.arange(Y.shape[0]),Y)],0.01))`
6. Open mst.py and ensure that at the top of the file the right leaf model is being imported (`from leaf_model_mnl import *`)
