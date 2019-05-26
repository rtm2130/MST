'''
This example shows how to apply the CMT tree algorithm to a simple synthetic dataset
The ground truth of the synethetic dataset is a CMT of depth 1

We will train the CMT on this dataset and observe whether it recovers the true CMT used to generate the data

X : 3 contexts:
  X0: binary {0,1}
  X1: binary {0,1}
  X2: ordinal in {0,0.2,0.4,0.6,0.8,1}

There are 5 products total. Each customer sees an assortment of 3 of these 5 products and chooses his favorite product in the assortment.
P encodes the offered assortment: P[i] = 1 iff item i is offered to the customer

Y in {0,1,2,3,4} encodes the customer's choice among the products

The CMT used to generate the data consists of a single split (x2 <= 0.6), with MNLs in each leaf with randomly-generated coefs

NOTE: for this to run properly, include the following import statement in mtp.py: "from leaf_model_choicemod import *"
'''

from mtp import MTP
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True) #suppress scientific notation

#generates expected utilities for each product given contexts (from a CMT with split x1 <= 0.6)
def get_expected_utilities_tree_depth_1(X, n_items):
  n = X.shape[0]
  Beta = np.random.uniform(low=-1, high=1, size=(2,n_items))
  U_exp = np.zeros((n,n_items))
  for k in range(n_items):
    U_exp[:,k] = np.where(X[:,2] <= 0.6, Beta[0,k], Beta[1,k])
   
  return U_exp

#generates responses Y, probability distribution of responses Y_prob given contexts X and assortments P
def get_choices(X,P):
  scale = 10 #dictates the level of noise in the choice probabilities. epsilon~Gumbel(0,1/scale)
  
  n = X.shape[0]
  n_items = P.shape[1]
  U_exp = get_expected_utilities_tree_depth_1(X, n_items)
  
  Y_prob = np.zeros((n,n_items))
  denom = sum([np.exp(scale*U_exp[:,k])*P[:,k] for k in range(n_items)])
  for k in range(n_items):
    Y_prob[:,k] = np.where(P[:,k] == 1, np.exp(scale*U_exp[:,k])/denom, 0)
  
  Y = np.zeros(n,dtype=int)
  for i in range(n):
    Y[i] = np.where(np.random.multinomial(1, Y_prob[i,:]))[0][0]
  
  return Y, Y_prob

#generates contexts X, assortments P, responses Y, response distributions Y_prob
def generate_data(n):
  
  n_items = 5
  assortment_size = 3
  
  #CUSTOMER FEATURES
  #X1 is in {0,1} (binary)
  X1 = np.random.choice([0,1], size=n, replace=True).reshape((n,1))
  #X2 is in {0,1} (binary)
  X2 = np.random.choice([0,1], size=n, replace=True).reshape((n,1))
  #X3 is in {0,0.2,0.4,0.6,0.8,1} (ordinal)
  X3 = np.random.choice([0,0.2, 0.4, 0.6, 0.8, 1], size=n, replace=True).reshape((n,1))
  X = np.concatenate((X1,X2,X3),axis = 1)
  
  #ASSORTMENT (5 products, choose 3 to offer to each customer)
  P = np.zeros((n,n_items))
  for i in range(n):
    assortment_items = np.random.choice(range(n_items), size=assortment_size, replace=False)
    P[i,assortment_items] = 1
  
  #P = np.ones((n,n_items))
  
  #OUTCOME
  Y, Y_prob = get_choices(X,P)
  
  return X,P,Y,Y_prob
  
  
#SIMULATED DATA PARAMETERS 
n_train = 5000;
n_valid = 2500;
n_test = 2000;

#GENERATE DATA

X,P,Y,Y_prob = generate_data(n_train+n_valid+n_test)
XV,PV,YV,Y_probV = X[n_train:(n_train+n_valid)],P[n_train:(n_train+n_valid)],Y[n_train:(n_train+n_valid)],Y_prob[n_train:(n_train+n_valid)] #valid set
XT,PT,YT,Y_probT = X[(n_train+n_valid):],P[(n_train+n_valid):],Y[(n_train+n_valid):],Y_prob[(n_train+n_valid):]  #test set
X,P,Y,Y_prob = X[:n_train],P[:n_train],Y[:n_train],Y_prob[:n_train] #training set

#APPLY TREE ALGORITHM. TRAIN TO DEPTH 1
my_tree = MTP(max_depth = 2, min_weights_per_node = 100, quant_discret = 0.05)
my_tree.fit(X,P,Y,verbose=True,
                feats_continuous=[False, False, True],
                refit_leaves=True,
                num_features = 1, mode = "mnl", batch_size = 100, epochs = 100, 
                steps = 5000, is_bias = True, model_type = 0);

#PRINT OUT THE UNPRUNED TREE. OBSERVE THAT THE FIRST SPLIT IS CORRECT, BUT THERE ARE UNNECESSARY SPLITS AFTER THAT
my_tree.traverse(verbose=True)   
#PRUNE THE TREE    
my_tree.prune(XV, PV, YV, verbose=False)
#PRINT OUT THE PRUNED TREE. OBSERVE THAT THE UNNECESSARY SPLITS HAVE BEEN PRUNED FROM THE TREE
my_tree.traverse(verbose=True)
#OBSERVE TEST SET MEAN-SQUARED-ERROR
Y_pred_pruned = my_tree.predict(XT,PT)        
score_pruned = np.sqrt(np.mean(np.power(Y_pred_pruned-Y_probT,2)))        
print(score_pruned)
    
   
