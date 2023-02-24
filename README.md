# MST

This is a Python code base for training "Market Segmentation Trees" (MSTs) (formerly known as "Model Trees for Personalization" (MTPs)). MSTs provide a general framework for jointly performing market segmentation and response modeling. The folder `/scripts` contains the code relative to the case study on Swiss Metro data.

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

For faster CMT fitting, users can combine this repo with the MNL fitting code found at https://github.com/rtm2130/CMT-R. Please see the CMT-R repo for further instructions. Note that the CMT-R repo is under a GPLv2 license, which is more restrictive on terms of use than this repo's MIT license.

For guidance on how to create a virtual environment containing all of the above package dependencies, see file "create_virtualenv_tutorial.txt" in the repo. For the precise package versions used in testing the MST code, see the environment_mst.yml file. Note that the code base was tested in Python=2.7.17. In particular, it is not guaranteed that the code will work correctly in Python 3.

To run MSTs on the Swiss Metro dataset used by our paper, please take the following steps:
1. Copy the files leaf_model_mnl_tensorflow.py and mst.py from this repo to the /scripts/src directory
2. Copy the files newmnlogit.R, leaf_model_mnl.py, and leaf_model_mnl_rmnlogit.py from the https://github.com/rtm2130/CMT-R repo to the /scripts/src directory
3. Create and activate a virtual environment using the environment_mst.yml file from the https://github.com/rtm2130/CMT-R repo
4. Open the /scripts/src/newmnlogit.R file and at the top of the file, change `ro.r.source("newmnlogit.R")` to `ro.r.source("src/newmnlogit.R")`
5. In /scripts/src/leaf_model_mnl_rmnlogit.py, within the implementation of the `error(self,A,Y)` function, change `log_probas = -np.log(Ypred[(np.arange(Y.shape[0]),Y)])` to `log_probas = -np.log(np.maximum(Ypred[(np.arange(Y.shape[0]),Y)],0.01))`
6. Open mst.py and ensure that at the top of the file the right leaf model is being imported (`from leaf_model_mnl import *`)
