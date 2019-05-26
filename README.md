# MTP

This is a Python code base for training "Model Trees for Personalization" (MTPs). MTPs provide a general framework for jointly performing market segmentation and response modeling. 

"mtp.py" contains the Python class "MTP" used for training MTPs. This class supports the following methods:
* \_\_init\_\_(): initializes the MTP
* fit(): trains the MTP on data: contexts X, decisions P (labeled as A in this code base), responses Y
* traverse(): prints out the learned MTP
* prune(): prunes the tree on a held-out validation set to prevent overfitting
* predict(): predict response distribution given new contexts X and decisions P

Users can build their own response models for the MTP class by filling in the template "leaf_model_template.py".

Two examples of MTPs are provided here:
(1) "Isotonic Regression Model Trees" (IRMTs): Here, the response models are isotonic regression models. The "irmt_exmaple.py" file provides an example of running IRMTs on a synthetic dataset.
(2) "Choice Model Trees" (CMTs): Here, the response models are MNL choice models. The "cmt_exmaple.py" file provides an example of running CMTs on a synthetic dataset.

Dependencies:
* mtp.py code: numpy, pandas, joblib, sklearn
* irmt_example.py code: numpy, pandas, joblib, sklearn
* cmt_example.py code: numpy, pandas, joblib, tensorflow
