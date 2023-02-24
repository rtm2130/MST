# -*- coding: utf-8 -*-
"""
RESPONSE MODEL + KMEANS IMPLEMENTATION
"""
#import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import tree
import copy
#from sklearn.cluster import MiniBatchKMeans

'''
Import proper leaf model import here:
'''
from leaf_model_mnl import *
#from leaf_model_isoreg import *

'''
This function uses k-means to find a market segmentation and fits a response model in each segment (corresponding to leaf model import above).
The number of clusters k is determined by finding the value k* in a user-inputted sequence of k's which minimizes response model prediction error on a validation set
This function outputs predicted choice probabilities on the test set. 
Input arguments:
  k_seq: The sequence of k's to try in fitting k-means
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
    All features satisfying feats_continuous == False will be binarized before running K-means.
  normalize_feats (optional): If True, normalizes all X features satisfying feats_continuous == True before running K-means
    Normalization of a feature means to transform it to have mean 0 and variance 1.
  tuning_loss_function (optional): the loss function used to score performance on the validation set when tuning the number of clusters k. Can be either "mse" or "loglik".
  n_init (optional): Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
  n_jobs (optional): the number of cores available for parallel processing
  verbose (optional): if verbose=True, prints out progress in k-means procedure  

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
def response_model_kmeans_fit_and_predict(k_seq,
                                          Xtrain,Atrain,Ytrain,
                                          Xval,Aval,Yval,
                                          Xtest,Atest, 
                                          weights_train=None, weights_val=None,
                                          feats_continuous=True,
                                          normalize_feats=True,
                                          n_jobs=None, n_init=10,
                                          verbose=True, tuning_loss_function="loglik",
                                          method = 'kmeans',
                                          *leafargs_fit,**leafkwargs_fit):
  
  if weights_train is None:
    weights_train = np.ones([Xtrain.shape[0]])
  if weights_val is None:
    weights_val = np.ones([Xval.shape[0]])
  
  if (isinstance(Xtrain,pd.core.frame.DataFrame)):
      Xtrain = Xtrain.values
  if (isinstance(Xval,pd.core.frame.DataFrame)):
      Xval = Xval.values
  if (isinstance(Xtest,pd.core.frame.DataFrame)):
      Xtest = Xtest.values

#  print(np.unique(Xtrain,axis = 0).shape)
  
  Xtrain, Xval, Xtest = binarize_and_normalize_contexts(Xtrain, Xval, Xtest, feats_continuous=feats_continuous, normalize=normalize_feats, drop_first=False)
  
#  print(np.unique(Xtrain,axis = 0).shape)
  
  for i,k in enumerate(k_seq):

    if method == "kmeans":
        if verbose == True:
            print("Fitting K-means with Num Clusters = " + str(k))
        predictions_train,predictions_val, predictions_test = kmeans_leafmod_fitvaltest(k,Xtrain,Atrain,Ytrain,weights_train,
                                                                  Xval,Aval,
                                                                  Xtest,Atest,
                                                                  n_jobs=n_jobs, n_init=n_init,
                                                                  *leafargs_fit,**leafkwargs_fit)
    elif method == 'tree':
        if verbose == True:
            print("Fitting Trees with Num Leafs = " + str(k))        
        predictions_val, predictions_test = dt_leafmod_fitvaltest(k,Xtrain,Atrain,Ytrain,weights_train,
                                                                  Xval,Aval,
                                                                  Xtest,Atest,
                                                                  n_jobs=n_jobs, n_init=n_init,
                                                                  *leafargs_fit,**leafkwargs_fit)    
    
    if tuning_loss_function == "mse":
      loss_val = calc_mse(Yval, predictions_val, weights=weights_val)
      mse_train = calc_mse(Ytrain, predictions_train, weights=weights_train)
      loss_train = calc_loglik(Ytrain, predictions_train, weights=weights_train)      
    elif tuning_loss_function == "loglik":
      loss_val = calc_loglik(Yval, predictions_val, weights=weights_val)
    else:
      print("Tuning loss function " + tuning_loss_function + " not supported. Must equal 'mse' or 'loglik'.")
      assert(False)
    
    if i == 0:
      best_loss_val, best_predictions_test, best_k = loss_val, predictions_test, k
    elif loss_val < best_loss_val:
      best_loss_val, best_predictions_test, best_k = loss_val, predictions_test, k
#    print("Current loss val",best_loss_val)
  if verbose == True:
    print("Optimal num. clusters: " + str(best_k))
  
#  loglik_test = calc_loglik(Ytest, best_predictions_test, weights=weights_test)
#  mse_test = calc_mse(Ytest, best_predictions_test, weights=weights_test)
#  print(loglik_test,mse_test)
  return(loss_train,mse_train,best_predictions_test,best_k)

def calc_loglik(Y, Ypred, weights=None):
  log_probas = -np.log(np.maximum(0.01,Ypred[(np.arange(Y.shape[0]),Y)]))
  loglik = np.average(log_probas, weights=weights)
  return(loglik)

def calc_mse(Y, Ypred, weights=None):
  Z = np.zeros(Ypred.shape)
  Z[(np.arange(Y.shape[0]),Y)] = 1.0
  errors = np.sum((Z-Ypred)**2.0,axis = 1)
  mse = np.average(errors, weights=weights)
  return(mse)

def dt_leafmod_fitvaltest(k,X,A,Y,weights,
                              Xval,Aval,
                              Xtest,Atest,
                              n_jobs=None, n_init=10,
                              *leafargs_fit,**leafkwargs_fit):
  
  if weights is None:
    weights = np.ones([X.shape[0]])

  my_cl = tree.DecisionTreeClassifier(max_leaf_nodes = k)
  my_cl = my_cl.fit(X, Y, sample_weight=weights)  
#  my_cl = KMeans(n_clusters=k, random_state=0, n_jobs=n_jobs, n_init=n_init)
#  my_cl.fit(X, sample_weight=weights)
  
  cl_inds = my_cl.predict(X)
  cl_inds_val = my_cl.predict(Xval)
  cl_inds_test = my_cl.predict(Xtest)
  
  for i,ind in enumerate(np.unique(cl_inds)):
    tmp = np.where(cl_inds == ind)[0]
    sub_A = get_sub(tmp,A=A,is_boolvec=False)
    sub_Y = get_sub(tmp,Y=Y,is_boolvec=False)            
    sub_weight = weights[tmp]
    
    tmp_val = (cl_inds_val == ind)
    sub_A_val = get_sub(tmp_val,A=Aval,is_boolvec=False)
    
    tmp_test = (cl_inds_test == ind)
    sub_A_test = get_sub(tmp_test,A=Atest,is_boolvec=False)
    
    lm = LeafModel()
    lm.fit(sub_A, sub_Y, sub_weight,
           refit=True,
           *leafargs_fit,**leafkwargs_fit)
    
    leaf_predictions_val = lm.predict(sub_A_val)
    if i == 0:
      if leaf_predictions_val.ndim == 1 and len(tmp_val) == len(leaf_predictions_val):
        predictions_val = np.zeros(Xval.shape[0])
      else:
        predictions_val = np.zeros((Xval.shape[0],leaf_predictions_val.shape[1]))
    predictions_val[tmp_val] = leaf_predictions_val  
    
    leaf_predictions_test = lm.predict(sub_A_test)
    if i == 0:
      if leaf_predictions_test.ndim == 1 and len(tmp_test) == len(leaf_predictions_test):
        predictions_test = np.zeros(Xtest.shape[0])
      else:
        predictions_test = np.zeros((Xtest.shape[0],leaf_predictions_test.shape[1]))
    predictions_test[tmp_test] = leaf_predictions_test
  
  return predictions_val, predictions_test

def kmeans_leafmod_fitvaltest(k,X,A,Y,weights,
                              Xval,Aval,
                              Xtest,Atest,
                              n_jobs=None, n_init=10,
                              *leafargs_fit,**leafkwargs_fit):
  
  if weights is None:
    weights = np.ones([X.shape[0]])
  
  my_cl = KMeans(n_clusters=k, random_state=0, n_jobs=n_jobs, n_init=n_init)
  my_cl.fit(X, sample_weight=weights)
  
#  print(my_cl.cluster_centers_)
  cl_inds = my_cl.predict(X)
  cl_inds_val = my_cl.predict(Xval)
  cl_inds_test = my_cl.predict(Xtest)
  
  for i,ind in enumerate(np.unique(cl_inds)):
    tmp = np.where(cl_inds == ind)[0]
    sub_A = get_sub(tmp,A=A,is_boolvec=False)
    sub_Y = get_sub(tmp,Y=Y,is_boolvec=False)            
    sub_weight = weights[tmp]
    
    tmp_val = (cl_inds_val == ind)
    sub_A_val = get_sub(tmp_val,A=Aval,is_boolvec=False)
    
    tmp_test = (cl_inds_test == ind)
    sub_A_test = get_sub(tmp_test,A=Atest,is_boolvec=False)
    
    lm = LeafModel()
    lm.fit(sub_A, sub_Y, sub_weight,
           refit=True,
           *leafargs_fit,**leafkwargs_fit)
#    print(ind,lm.to_string())
    if sub_A_val.shape[0]>0:
        leaf_predictions_val = lm.predict(sub_A_val)[:sub_A_val.shape[0],:]
    else:
        leaf_predictions_val = np.zeros((0,3))
        
    if i == 0:
      if leaf_predictions_val.ndim == 1 and len(tmp_val) == len(leaf_predictions_val):
        predictions_val = np.zeros(Xval.shape[0])
      else:
        predictions_val = np.zeros((Xval.shape[0],leaf_predictions_val.shape[1]))
    try:        
        predictions_val[tmp_val] = leaf_predictions_val  
    except:
        print(i,tmp_val.sum(),leaf_predictions_val.shape,sub_A_val.shape)   
        raise
        
    if sub_A_test.shape[0]>0:
        leaf_predictions_test = lm.predict(sub_A_test)[:sub_A_test.shape[0],:]
    else:
        leaf_predictions_test = np.zeros((0,3))
    if i == 0:
      if leaf_predictions_test.ndim == 1 and len(tmp_test) == len(leaf_predictions_test):
        predictions_test = np.zeros(Xtest.shape[0])
      else:
        predictions_test = np.zeros((Xtest.shape[0],leaf_predictions_test.shape[1]))

    try:        
        predictions_test[tmp_test] = leaf_predictions_test
    except:
        print(i,tmp_test.sum(),leaf_predictions_test.shape,sub_A_test.shape)
        raise   
        
        
    if sub_A.shape[0]>0:
        leaf_predictions_train = lm.predict(sub_A)[:sub_A.shape[0],:]
    else:
        leaf_predictions_train = np.zeros((0,3))
    if i == 0:
      if leaf_predictions_train.ndim == 1 and len(tmp) == len(leaf_predictions_train):
        predictions_train = np.zeros(X.shape[0])
      else:
        predictions_train = np.zeros((X.shape[0],leaf_predictions_train.shape[1]))

    try:        
        predictions_train[tmp] = leaf_predictions_train
    except:
        print(i,tmp.sum(),leaf_predictions_train.shape,sub_A.shape)
        raise          
  
  return predictions_train,predictions_val, predictions_test

#outputs test set predictions
#def kmeansVal_leafmod_fittest(k_seq,X,A,Y,weights,
#                             Xval,Aval,Yval,weights_val,
#                             Xtest,Atest, 
#                             n_jobs=None, n_init=10, 
#                             *leafargs_fit,**leafkwargs_fit):
#  
#  if weights is None:
#    weights = np.ones([X.shape[0]])
#  if weights_val is None:
#    weights_val = np.ones([Xval.shape[0]])
#  
#  for i,k in enumerate(k_seq):
#    print("Fitting K-means with Num Clusters = " + str(k))
#    predictions_val, predictions_test = kmeans_leafmod_fitvaltest(k,X,A,Y,weights,
#                                                                  Xval,Aval,
#                                                                  Xtest,Atest,
#                                                                  n_jobs=n_jobs, n_init=n_init,
#                                                                  *leafargs_fit,**leafkwargs_fit)
#    
#    mse_val = calc_mse(Yval, predictions_val, weights=weights_val)
#    
#    if i == 0:
#      best_mse_val, best_predictions_test, best_k = mse_val, predictions_test, k
#    elif mse_val < best_mse_val:
#      best_mse_val, best_predictions_test, best_k = mse_val, predictions_test, k
#      
#  print("Optimal num. clusters: " + str(best_k))
#  
#  #loglik_test = calc_loglik(Ytest, best_predictions_test, weights=weights_test)
#  #mse_test = calc_mse(Ytest, best_predictions_test, weights=weights_test)
#  
#  return(best_predictions_test)

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
    df_dumm = pd.get_dummies(X_categorical,columns=X_categorical.columns,drop_first=drop_first)
    feats_categorical = df_dumm.columns.tolist()
    X_categorical_bin = df_dumm.values
  
  if all_continuous:
    X_new = X_continuous
#    print('flist', feats_continuous)
    #feats_continuous_new = [True]*X_continuous.shape[1]
  elif all_categorical:
    X_new = X_categorical_bin
#    print('flist', feats_categorical)
    #feats_continuous_new = [False]*X_categorical_bin.shape[1]
  else:
    X_new = np.concatenate([X_categorical_bin, X_continuous], axis=1)
#    print('flist',feats_categorical+feats_continuous)
    #feats_continuous_new = [False]*X_categorical_bin.shape[1] + [True]*X_continuous.shape[1]
  
  X_train_new = X_new[:n_train,:]
  X_valid_new = X_new[n_train:(n_train+n_valid),:]
  X_test_new = X_new[(n_train+n_valid):,:]
  
  return (X_train_new, X_valid_new, X_test_new)
  #return (X_train_new, X_valid_new, X_test_new, feats_continuous_new)  