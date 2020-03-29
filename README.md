# MST

This is a Python code base for training "Market Segmentation Trees" (MSTs) (formerly known as "Model Trees for Personalization" (MTPs)). MSTs provide a general framework for jointly performing market segmentation and response modeling. 

"mst.py" contains the Python class "MST" used for training MSTs. This class supports the following methods:
* \_\_init\_\_(): initializes the MST
* fit(): trains the MST on data: contexts X, decisions P (labeled as A in this code base), responses Y
* traverse(): prints out the learned MST
* prune(): prunes the tree on a held-out validation set to prevent overfitting
* predict(): predict response distribution given new contexts X and decisions P

Users can build their own response models for the MST class by filling in the template "leaf_model_template.py".

Two examples of MSTs are provided here:
(1) "Isotonic Regression Model Trees" (IRTs): Here, the response models are isotonic regression models. The "irt_exmaple.py" file provides an example of running IRTs on a synthetic dataset.
(2) "Choice Model Trees" (CMTs): Here, the response models are MNL choice models. The "cmt_exmaple.py" file provides an example of running CMTs on a synthetic dataset.

Dependencies:
* mst.py code: numpy, pandas, joblib, sklearn
* irt_example.py code: numpy, pandas, joblib, sklearn
* cmt_example.py code: numpy, pandas, joblib, tensorflow
