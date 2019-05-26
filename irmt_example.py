#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" 
SIMULATED LOGISTIC MODEL TREE DATASET FOR IRMT EVALUATION

Dataset is generated using a logistic regression model tree of depth 2 (see the Depth2Tree function).
This example runs our IRMT tree-building algorithm on the dataset to test whether it recovers the true tree.
Note that as an additional challenge to the IRMT, the leaf nodes of the data-generating tree use logistic regression models (not isotonic regression)

NOTE: for this to run properly, include the following import statement in mtp.py: "from leaf_model_isoreg import *"
"""

import numpy as np
import pandas as pd

from mtp import MTP

np.set_printoptions(suppress=True) #suppress scientific notation

"""
Given auction features X and bids P, returns vector of probabilities 
corresponding to whether or not the bid will win the auction.
Dataset X: 3 covariates:
X1: Binary in {0,1}
X2: Continuous, takes values in [0,1]
X3: Ordinal, takes values {0.0,0.2,0.4,0.6,0.8,1.0}

This depth-2 logistic model tree is used to simulate the auction data
"""
def Depth2Tree(X,P):
  
  num_obs = X.shape[0]
  
  probs = np.ones([num_obs])
  for i in range(0,num_obs):
    x = X.iloc[i,:]
    p = P[i]
    
    if (x['X3'] <= 0.6):
     if x['X1'] == 0:
       a = 20.0; #steepness of logistic curve
       p_thresh = 35.0; #price at which logistic curve is centered (i.e., L(p) = 0.5)
       b = -a/p_thresh;
     else:
       a = 20.0; #steepness of logistic curve
       p_thresh = 55.0; #price at which logistic curve is centered (i.e., L(p) = 0.5)
       b = -a/p_thresh;
     
    else:
     if x['X2'] <= 0.6:
       a = 20.0; #steepness of logistic curve
       p_thresh = 65.0; #price at which logistic curve is centered (i.e., L(p) = 0.5)
       b = -a/p_thresh;
     else:
       a = 20.0; #steepness of logistic curve
       p_thresh = 95.0; #price at which logistic curve is centered (i.e., L(p) = 0.5)
       b = -a/p_thresh;
     
    probs[i] = 1.0/(1.0+np.exp(-a-b*p));
  
  return(probs);

#SIMULATED DATA PARAMETERS 
n_train = 100000;
n_valid = 2000;
n_test = 5000;
p_min = 10;
p_max = 90;
X1range = [0,1];
X3range = [0.0,0.2,0.4,0.6,0.8,1.0];

#generates data from LRMT of depth 2
def generate_data(n):
  #auction features
  X1 = np.random.choice(X1range, size=n_train, replace=True)
  X2 = np.random.uniform(low=0.0,high=1.0,size=n_train)
  X3 = np.random.choice(X3range, size=n_train, replace=True)
  X = pd.DataFrame({'X1': X1,'X2': X2,'X3': X3})
  #bids
  P = np.random.uniform(low = p_min, high = p_max, size=n_train);
  #outcomes: auction win indicators
  Y_prob = Depth2Tree(X,P);
  Y = np.random.binomial(1,Y_prob, size=n_train);
  
  return X,P,Y,Y_prob

#GENERATE TRAINING DATA
X,P,Y,Y_prob = generate_data(n_train)

#FIT IRMT ALGORITHM
my_tree = MTP(max_depth = 5, min_weights_per_node = 20)
my_tree.fit(X,P,Y,verbose=True,feats_continuous=[False,True,True],increasing=False); #verbose specifies whether fitting procedure should print progress
#ABOVE: increasing specifies whether fit isotonic regression models should be monotonically increasing or decreasing.
#note ground truth has decreasing logistic curves.
#my_tree.traverse() #prints out the unpruned tree 

#GENERATE VALIDATION SET DATA
X,P,Y,Y_prob = generate_data(n_valid)

#PRUNE DECISION TREE USING VALIDATION SET
my_tree.prune(X, P, Y, verbose=True) #verbose specifies whether pruning procedure should print progress
my_tree.traverse() #prints out the pruned tree, compare it against depth-2 tree used to generate the data

#GENERATE TESTING DATA
X,P,Y,Y_prob = generate_data(n_test)

#USE TREE TO PREDICT TEST-SET PROBABILITIES AND MEASURE ERROR
Ypred = my_tree.predict(X,P)
print(np.mean(abs(Y_prob-Ypred)))
  


