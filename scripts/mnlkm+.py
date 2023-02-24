
import numpy as np
import pandas as pd
from src.mst import MST
from src.responsemodel_kmeans import response_model_kmeans_fit_and_predict
from src.cart_customerfeats_in_leafmod import append_customer_features_to_product_features



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

scores_np = np.zeros((10,2,5,30))

for i in range(10):
    num_features = 4
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
    
    P, PV, PT, num_features = append_customer_features_to_product_features(X, XV, XT,
                                                                           P, PV, PT,
                                                                           feats_continuous=is_continuous, 
                                                                           model_type=model_type, num_features=num_features)    
    print(P.shape,PV.shape,PT.shape,X.shape)
    #############################################################################
    #(2) Run MNLKM (k-means with MNL response model)
    for n_clusters in range(1): 
        k_seq = range(5,300,10)
#        k_seq = [11]        
#    for n_clusters in range(30):        
#        k_seq = [5+10*n_clusters]         
        #fit MNLKM and output test set predictions. See code responsemodel_kmeans.py for more details
        loss,mse,Y_predT,best_k = response_model_kmeans_fit_and_predict(k_seq,
                                                        X,P,Y,
                                                        XV,PV,YV,
                                                        XT,PT, 
                                                        feats_continuous=is_continuous, normalize_feats=True,
                                                        n_init=10, verbose=True, tuning_loss_function="mse",
                                                        num_features = num_features, is_bias = is_bias, model_type = model_type,
                                                        mode = "mnl", batch_size = 100, epochs = 100, steps = 5000,
                                                        method = 'kmeans',
                                                        leaf_mod_thresh=1000000000000)
        
        YT_flat = np.zeros((YT.shape[0],3))
        YT_flat[np.arange(YT.shape[0]),YT] = 1
        
        s_Y = Y_predT.shape[0]
        scores_np[i,0,0,n_clusters] = np.mean(np.log(np.maximum(0.01,Y_predT[np.arange(s_Y),YT])))
        scores_np[i,0,1,n_clusters] = np.mean(np.sum(np.power(Y_predT-YT_flat,2),axis = 1))
        scores_np[i,0,2,n_clusters] = loss 
        scores_np[i,0,3,n_clusters] = mse         
        scores_np[i,0,4,n_clusters] = best_k
        
                      