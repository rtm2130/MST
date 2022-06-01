'''
This example shows how to apply the MST CMT tree algorithm to a simple synthetic dataset
The ground truth of the synethetic dataset is a CMT of depth 1

We will train the CMT on this dataset and observe whether it recovers the true CMT used to generate the data

n = number of historical customers (i.e., training observations)
X is a n x 3 matrix containing customers' contextual information
3 contexts:
  X0: binary {0,1}
  X1: multi-categorical {0, 1, 2, 3}
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

#from mst import MST
#from GenMNL import GenMNL
#from responsemodel_kmeans import response_model_kmeans_fit_and_predict
from src.cart_customerfeats_in_leafmod import mnl_cart_fit_prune_and_predict, append_customer_features_to_product_features
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True) #suppress scientific notation
np.random.seed(0)
  

c_features = ["GROUP", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE", 
              "MALE", "INCOME", "GA", "ORIGIN", "DEST"]
p_features = ["TRAIN_AV", "SM_AV", "CAR_AV",
              "TRAIN_TT", "SM_TT", "CAR_TT",
              "TRAIN_CO", "SM_CO", "CAR_CO",
              "TRAIN_HE", "SM_HE", "CAR_HE"]

num_features = 4
model_type = 0
is_bias = True
is_continuous = [False for k in range(len(c_features))]
is_continuous[6] = True 
is_continuous[8] = False 

scores_np = np.zeros((10,2,5,15))

for i in range(10):
    
    X = np.load('data/X'+str(i)+'.npy')
    P = np.load('data/P'+str(i)+'.npy')
    Y = np.load('data/Y'+str(i)+'.npy')
    
    XV = np.load('data/XV'+str(i)+'.npy')
    PV = np.load('data/PV'+str(i)+'.npy')
    YV = np.load('data/YV'+str(i)+'.npy')
    
    XT = np.load('data/XT'+str(i)+'.npy')
    PT = np.load('data/PT'+str(i)+'.npy')
    YT = np.load('data/YT'+str(i)+'.npy')
    
    P = P.astype(float)
    PV = PV.astype(float)
    PT = PT.astype(float)   
    
    P[:,-1] = 0.001*np.random.rand(P.shape[0])
    PV[:,-1] = 0.001*np.random.rand(PV.shape[0])
    PT[:,-1] = 0.001*np.random.rand(PT.shape[0])      
    
    #############################################################################
    #(3) Run MNL-CART (CART with MNL response model refit in each leaf)
    print("Running MNL-CART")
    
    for d in [14]:
#    for d in range(3,5):        
        #In this benchmark, we fit MNLs in each leaf using *both* the customer features and product features
        #Therefore, we use this function to add the customer features (X) to the MNL features matrix (P)
        #NOTE: this function handles binarization of customer features (X) internally prior to appending to product feature matrix P
        #----Specifically, the function will binarize all customer features in X satisfying feats_continuous = False
        #NOTE: if model_type = 1 (alternative-general coefs), then this function still encodes Pnew in such a way that the customer features have alt-specific coefs
        Pnew, PVnew, PTnew, num_features_new = P,PV,PT,num_features        
#        Pnew, PVnew, PTnew, num_features_new = append_customer_features_to_product_features(X, XV, XT,
#                                                                                            P, PV, PT,
#                                                                                            feats_continuous=is_continuous, 
#                                                                                            model_type=model_type, num_features=num_features)
        #fit MNL-CART and output test set predictions. See code cart_customerfeats_in_leafmod.py for more details
        Y_predT,my_tree = mnl_cart_fit_prune_and_predict(X,Pnew,Y,
                                                 XV,PVnew,YV,
                                                 XT,PTnew,  
                                                 feats_continuous=is_continuous,
                                                 verbose=True,
                                                 one_SE_rule=True,
                                                 max_depth=d, min_weights_per_node=50, quant_discret=0.05,
                                                 run_in_parallel=False,num_workers=None,
                                                 num_features = num_features_new, is_bias = is_bias, model_type = model_type,
                                                 mode = "mnl", batch_size = 100, epochs = 100, steps = 5000,
                                                 leaf_mod_thresh=1000000000000)
        
        
        YT_flat = np.zeros((YT.shape[0],3))
        YT_flat[np.arange(YT.shape[0]),YT] = 1
        
        s_Y = Y_predT.shape[0]
        scores_np[i,0,0,d] = np.mean(np.log(np.maximum(0.01,Y_predT[np.arange(s_Y),YT])))
        scores_np[i,0,1,d] = np.mean(np.sum(np.power(Y_predT-YT_flat,2),axis = 1))

