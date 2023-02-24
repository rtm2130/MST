'''
This example shows how to apply the MST CMT tree algorithm to a simple synthetic dataset
The ground truth of the synethetic dataset is a CMT of depth 1

We will train the CMT on this dataset and observe whether it recovers the true CMT used to generate the data

n = number of historical customers (i.e., training observations)
X is a n x 3 matrix containing customers' contextual information
3 contexts:
  X0: binary {0,1}
  X1: binary {0,1}
  X2: ordinal in {0,0.2,0.4,0.6,0.8,1}

There are 5 products total. Each customer sees a random assortment of 3 of these 5 products and chooses his favorite product in the assortment.
P[:,:5] encodes the offered assortment: P[i,j] = 1 iff item j was offered to customer i
There are two other product features (besides the assortment indicators) which can be interpreted as price and quality rating. 
  Product prices are stored in P[:,5:10]. P[i,(j+5)] = price of item j offered to customer i
  Quality ratings are stored in P[:,10:15]. P[i,(j+10)] = quality rating of item j offered to customer i

Y is an n-dim vector, where Y[i] in {0,1,2,3,4} encodes customer i's choice among the products

The CMT used to generate the data consists of a single split (x2 <= 0.6), with MNLs in each leaf with randomly-generated coefs

NOTE: for this to run properly, include the following import statement in mst.py: "from leaf_model_mnl import *"
'''

from mst import MST
from GenMNL import GenMNL
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True) #suppress scientific notation
np.random.seed(0)

'''
Generates responses Y, probability distribution of responses Y_prob given contexts X and assortments P

The CMT used to generate the data consists of a single split (x2 <= 0.6), with MNLs in each leaf with randomly-generated coefs

Arguments specifying MNL model type
n_items: number of products (integer)
num_features:  number of product features (integer), INCLUDING the binary availability feature
model_type: whether the model has alternative varying coefficients (0) or not (1). Type integer (0/1)
  (default is 0 meaning each alternative has a separate coeff)
is_bias: whether the utility function has an intercept (default is True). Type boolean (True/False).
'''
def get_choices(X, P, n_items, num_features, model_type, is_bias):
  n = X.shape[0]
  
  left_inds = np.where(X[:,2] <= 0.6)[0]
  right_inds = np.where(X[:,2] > 0.6)[0]
  
  left_mnl = GenMNL(n_items, num_features, model_type, is_bias)
  right_mnl = GenMNL(n_items, num_features, model_type, is_bias)
  
  Y_prob = np.zeros((n,n_items))
  Y_prob[left_inds] = left_mnl.get_choice_probs(P[left_inds])
  Y_prob[right_inds] = right_mnl.get_choice_probs(P[right_inds])
  
  Y = np.zeros(n,dtype=int)
  for i in range(n):
    Y[i] = np.where(np.random.multinomial(1, Y_prob[i,:]))[0][0]
  
  return Y, Y_prob

'''
Generates contexts X, assortments P, responses Y, response probability distributions Y_prob

Arguments:
n: number of training observations (integer)
n_items: number of products (integer)
num_features:  number of product features (integer), INCLUDING the binary availability feature
model_type: whether the model has alternative varying coefficients (0) or not (1). Type integer (0/1)
  (default is 0 meaning each alternative has a separate coeff)
is_bias: whether the utility function has an intercept (default is True). Type boolean (True/False).
'''
def generate_data(n, n_items, assortment_size, num_features, model_type, is_bias):
  
  #GENERATE CUSTOMER FEATURES
  #X1 is in {0,1} (binary)
  X1 = np.random.choice([0,1], size=n, replace=True).reshape((n,1))
  #X2 is in {0,1} (binary)
  X2 = np.random.choice([0,1], size=n, replace=True).reshape((n,1))
  #X3 is in {0,0.2,0.4,0.6,0.8,1} (ordinal)
  X3 = np.random.choice([0,0.2, 0.4, 0.6, 0.8, 1], size=n, replace=True).reshape((n,1))
  X = np.concatenate((X1,X2,X3),axis = 1)
  
  #GENERATE ASSORTMENT AND PRODUCT FEATURES 
  P = np.zeros((n,num_features*n_items))
  for i in range(n):
    #generate assortment features (5 products, choose 3 to offer to each customer)
    assortment_items = np.random.choice(range(n_items), size=assortment_size, replace=False)
    P[i,assortment_items] = 1
    #generate price and quality rating features
    P[i,n_items:] = np.random.uniform(size=(num_features-1)*n_items)
  
  #GENERATE OUTCOMES (response probability distributions Y_prob, observed responses Y)
  Y, Y_prob = get_choices(X, P, n_items, num_features, model_type, is_bias)
  
  return X,P,Y,Y_prob
  
  
#SIMULATED DATA PARAMETERS 
n_train = 5000;
n_valid = 2500;
n_test = 2000;

n_items = 5
assortment_size = 3
num_features = 3
model_type = 0
is_bias = True

#GENERATE DATA
X,P,Y,Y_prob = generate_data(n_train+n_valid+n_test, n_items, assortment_size, num_features, model_type, is_bias)
XV,PV,YV,Y_probV = X[n_train:(n_train+n_valid)],P[n_train:(n_train+n_valid)],Y[n_train:(n_train+n_valid)],Y_prob[n_train:(n_train+n_valid)] #valid set
XT,PT,YT,Y_probT = X[(n_train+n_valid):],P[(n_train+n_valid):],Y[(n_train+n_valid):],Y_prob[(n_train+n_valid):]  #test set
X,P,Y,Y_prob = X[:n_train],P[:n_train],Y[:n_train],Y_prob[:n_train] #training set

#APPLY TREE ALGORITHM. TRAIN TO DEPTH 1
my_tree = MST(max_depth = 2, min_weights_per_node = 100, quant_discret = 0.05)
my_tree.fit(X,P,Y,verbose=False,
                feats_continuous=[False, False, True],
                refit_leaves=True,
                num_features = num_features, is_bias = is_bias, model_type = model_type,
                mode = "mnl", batch_size = 100, epochs = 100, steps = 5000,
                leaf_mod_thresh=10000000);
#ABOVE: leaf_mod_thresh controls whether when fitting a leaf node we apply Newton's method or stochastic gradient descent.
# If the number of training observations in a leaf node <= leaf_mod_thresh, then newton's method
# is applied; otherwise, stochastic gradient descent is applied.
            
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
    
   
