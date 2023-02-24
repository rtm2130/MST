# -*- coding: utf-8 -*-
"""
MNL + CART IMPLEMENTATION
"""
#import math
import numpy as np
import pandas as pd
import copy
from cart_with_mnl_leaf_refitting import CARTWithMNLLeafRefitting


'''
This function fits and prunes a CART tree, and then as a postprocessing step fits an MNL response model in each segment.
This function outputs predicted choice probabilities on the test set. 
Input arguments:
  Xtrain, Xval, Xtest: The individual-specific feature data for the training, validation, and test sets. Can either be a pandas data frame or numpy array, with:
    (a) rows of X = observations/customers
    (b) columns of X = features about the observation/customer
  Atrain, Aval, Atest: the decision variables/alternative-specific features used in the response models for the training, validation, and test sets.
    A can take any form -- it is directly passed to the functions in leaf_model.py
  Ytrain, Yval: the responses/outcomes/choices used in the response models.
    Y must be a 1-D array with length = number of observations
  weights_train (optional), weights_val (optional): an optional numpy array of case weights for the training and validation sets. Is 1-dimensional, with weights[i] yielding weight of observation/customer i
  feats_continuous (optional): If False, all X features are treated as categorical. If True, all features are treated as continuous.
    feats_continuous can also be a boolean vector of dimension = num_features specifying how to treat each feature.
    All features satisfying feats_continuous == False will be binarized when fitting the leaf models.
  verbose (optional): if verbose=True, prints out progress in training MST
  one_SE_rule (default True): do we use 1SE rule when pruning the tree?
  MST standard input arguments:
    max_depth: the maximum depth of the pre-pruned tree (default = Inf: no depth limit)
    min_weight_per_node: the mininum number of observations (with respect to cumulative weight) per node
    min_depth: the minimum depth of the pre-pruned tree (default: set equal to max_depth)
    min_diff: if depth > min_depth, stop splitting if improvement in fit does not exceed min_diff
    quant_discret: continuous variable split points are chosen from quantiles of the variable corresponding to quant_discret,2*quant_discret,3*quant_discret, etc.. 
    run_in_parallel: if set to True, enables parallel computing among num_workers threads. If num_workers is not
      specified, uses the number of cpu cores available.

Any additional keyword arguments are passed to the leaf_model fit() function 
For leaf_model_mnl, you should pass the following:
    n_features:  integer (default is 2)
    mode : "mnl" or "exponomial" (default is "mnl")
    batch_size : size of the stochastic batch (default is 50,)
    model_type : whether the model has alternative varying coefficients or not (default is 0 meaning each alternative has a separate coeff)
    num_features : number of features under consideration (default is 2) 
    epochs : number of epochs for the estimation (default is 10) 
    is_bias : whether the utility function has an intercept (default is True)
'''
def mnl_cart_fit_prune_and_predict(Xtrain,Atrain,Ytrain,
                                   Xval,Aval,Yval,
                                   Xtest,Atest, 
                                   weights_train=None, weights_val=None,
                                   feats_continuous=True,
                                   verbose=True,
                                   one_SE_rule=True,
                                   max_depth=float("inf"),min_weights_per_node=15,
                                   min_depth=None,min_diff=0,
                                   quant_discret=0.01,
                                   run_in_parallel=False,num_workers=None,
                                   only_singleton_splits=True,
                                   *leafargs_fit,**leafkwargs_fit):
  
  my_tree = CARTWithMNLLeafRefitting(max_depth=max_depth,min_weights_per_node=min_weights_per_node,
                                      min_depth=min_depth,min_diff=min_diff,
                                      quant_discret = quant_discret,
                                      run_in_parallel=run_in_parallel,num_workers=num_workers,
                                      only_singleton_splits=only_singleton_splits)
  
  my_tree.fit(Xtrain, Atrain, Ytrain, weights=weights_train, feats_continuous=feats_continuous, verbose=verbose, *leafargs_fit,**leafkwargs_fit)
  my_tree.prune(Xval, Aval, Yval, weights_val=weights_val, one_SE_rule=one_SE_rule, verbose=verbose)
  #my_tree.traverse(verbose=True)
  my_tree.refit_leafmods_with_mnl(Xtrain, Atrain, Ytrain, weights_new=weights_train, verbose=verbose, *leafargs_fit,**leafkwargs_fit)
  #my_tree.traverse(verbose=True)
  
  Ypred_test = my_tree.predict(Xtest, Atest)
  
  return(Ypred_test,my_tree)

'''
This function takes customer feature matrix X and product feature matrix A as input.
Outputs a new product feature matrix A_new with customer features included
NOTE: A_new is encoded such that each customer feature will have an alternative-specific coefficient regardless of model_type specification
'''
def append_customer_features_to_product_features(X_train, X_valid, X_test,
                                                 A_train, A_valid, A_test,
                                                 feats_continuous=True, 
                                                 model_type=0, num_features=2):
  
  X_train_bin, X_valid_bin, X_test_bin = binarize_and_normalize_contexts(X_train, X_valid, X_test, 
                                                                         feats_continuous=feats_continuous, 
                                                                         normalize=True, 
                                                                         drop_first=True)
  n_items = int(A_train.shape[1]/num_features)
  A_train_with_custfeats = _append_customer_features_to_product_features(A_train, X_train_bin, n_items, model_type)
  A_valid_with_custfeats = _append_customer_features_to_product_features(A_valid, X_valid_bin, n_items, model_type)
  A_test_with_custfeats = _append_customer_features_to_product_features(A_test, X_test_bin, n_items, model_type)
  num_features_with_custfeats = int(A_train_with_custfeats.shape[1]/n_items)
  return(A_train_with_custfeats, A_valid_with_custfeats, A_test_with_custfeats, num_features_with_custfeats)

'''
This function takes customer feature matrix X and product feature matrix A as input.
Outputs a new product feature matrix A_new with customer features included
NOTE: A_new is encoded such that each customer feature will have an alternative-specific coefficient regardless of model_type specification
''' 
def _append_customer_features_to_product_features(A, X_bin, n_items, model_type):
  if model_type == 0:
    #MNL model is encoded as having alternative-specific coefs
    A_cust = np.repeat(X_bin, n_items, axis=1)
  else:
    #MNL model is not encoded as having alternative-specific coefs, so need to encode data in a special way to ensure
    #customer feats each have alternative-specific coefs
    A_cust = np.repeat(X_bin, n_items*(n_items-1), axis=1)
    num_obs = X_bin.shape[0]
    n_cust_feats = X_bin.shape[1]
    zeroing_mat = np.zeros((num_obs, n_items*(n_items-1)))
    zeroing_mat[:,np.arange(n_items-1) + n_items*np.arange(n_items-1)] = 1
    zeroing_mat = np.tile(zeroing_mat, n_cust_feats)
    A_cust = A_cust * zeroing_mat
  return np.concatenate((A, A_cust), axis=1)
  
'''
This function performs the following operations on contextual feature matrices X_train, X_valid, X_test:
(1) Binarizes any features which are categorical, i.e. where feats_continuous==False
(2) If normalize==True, normalizes any features which are numerical, i.e. where feats_continuous==True.
  (Normalize a feature means to transform it to have mean 0 and variance 1)
This function outputs normalized feature matrices X_train_new, X_valid_new, X_test_new w/ binary categorical features FIRST, then continuous

Input parameters:
  X_train, X_valid, X_test: contextual feature matrices for the training, validation, and test sets which have the following dimensions:
    (a) rows of X = observations/customers
    (b) columns of X = features about the observation/customer
  feats_continuous: If False, all feature are treated as categorical. If True, all feature are treated as continuous.
    feats_continuous can also be a boolean vector of dimension = num_features specifying how to treat each feature
  normalize: Whether to normalize the numerical features
  drop_first: Whether to get k-1 dummies out of k categorical levels by removing the first level
'''

def binarize_and_normalize_contexts(X_train, X_valid, X_test, feats_continuous=True, normalize=True, drop_first=False):
  X_train = copy.deepcopy(X_train)
  X_valid = copy.deepcopy(X_valid)
  X_test = copy.deepcopy(X_test)
  
  n_train = X_train.shape[0]
  n_valid = X_valid.shape[0]
  
  all_continuous = np.all(feats_continuous)
  all_categorical = np.all(np.logical_not(feats_continuous))
  
  X = np.concatenate([X_train, X_valid, X_test], axis=0)
  
  if not all_categorical:
    X_continuous = X[:,np.where(feats_continuous)[0]].astype("float")
    if normalize==True:
      X_continuous_std = np.std(X_continuous,axis=0)
      X_continuous_std[X_continuous_std == 0.0] = 1.0
      X_continuous = (X_continuous - np.mean(X_continuous, axis=0)) / X_continuous_std
  
  if not all_continuous:
    X_categorical = X[:,np.where(np.logical_not(feats_continuous))[0]]
    X_categorical = pd.DataFrame(X_categorical)
    X_categorical_bin = pd.get_dummies(X_categorical,columns=X_categorical.columns,drop_first=drop_first).values
  
  if all_continuous:
    X_new = X_continuous
    #feats_continuous_new = [True]*X_continuous.shape[1]
  elif all_categorical:
    X_new = X_categorical_bin
    #feats_continuous_new = [False]*X_categorical_bin.shape[1]
  else:
    X_new = np.concatenate([X_categorical_bin, X_continuous], axis=1)
    #feats_continuous_new = [False]*X_categorical_bin.shape[1] + [True]*X_continuous.shape[1]
  
  X_train_new = X_new[:n_train,:]
  X_valid_new = X_new[n_train:(n_train+n_valid),:]
  X_test_new = X_new[(n_train+n_valid):,:]
  
  return (X_train_new, X_valid_new, X_test_new)
  #return (X_train_new, X_valid_new, X_test_new, feats_continuous_new) 