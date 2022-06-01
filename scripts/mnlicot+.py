
#from GenMNL import GenMNL
from mst import MST
import numpy as np
from cart_customerfeats_in_leafmod import append_customer_features_to_product_features
# import pandas as pd


#np.set_printoptions(suppress=True) #suppress scientific notation
#np.random.seed(0)
# df = pd.read_csv('swissmetro.dat',sep='\t')
# df["CAR_HE"] = 0

c_features = ["GROUP", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE", 
              "MALE", "INCOME", "GA", "ORIGIN", "DEST"]
p_features = ["TRAIN_AV", "SM_AV", "CAR_AV",
              "TRAIN_TT", "SM_TT", "CAR_TT",
              "TRAIN_CO", "SM_CO", "CAR_CO",
              "TRAIN_HE", "SM_HE", "CAR_HE"]
# target = "CHOICE"
# df = df[df[target] > 0]
# df.loc[:,target] = df.loc[:,target]-1
# df = df.reset_index()

# df = df.sample(frac=1).reset_index(drop=True)


# def prepare_data(df, k1, k2):
#   '''
#     Prepares a partition of the data into train, validation and test sets
    
#     Args:
#         k1 -> number of observations in the test set
#         k2 -> number of observations in the validation set    
#   '''
#   n = df.shape[0]
#   selected_indices = np.random.choice(range(n), size=k1 + k2, replace=False)
#   test_indices = np.random.choice(selected_indices, size=k1, replace=False)
#   validation_indices = np.setdiff1d(selected_indices,test_indices)
#   train_indices = np.setdiff1d(np.arange(n),selected_indices)
#   Y = df.loc[train_indices,target].values
#   X = df.loc[train_indices,c_features].values 
#   P = df.loc[train_indices,p_features].values   
#   YT = df.loc[test_indices,target].values
#   XT = df.loc[test_indices,c_features].values 
#   PT = df.loc[test_indices,p_features].values 
#   YV = df.loc[validation_indices,target].values
#   XV = df.loc[validation_indices,c_features].values 
#   PV = df.loc[validation_indices,p_features].values   
  
#   return X,P,Y,XV,PV,YV,XT,PT,YT,train_indices,validation_indices,test_indices


  
  
# #SIMULATED DATA PARAMETERS 
# #n_train = 5000;
# n_valid = int(df.shape[0]/10);
# n_test = int(df.shape[0]/10);

# X,P,Y,XV,PV,YV,XT,PT,YT,train_indices,validation_indices,test_indices = prepare_data(df, n_test, n_valid)

# #np.save('prepared_data/X.npy',X)
# #np.save('prepared_data/P.npy',P)
# #np.save('prepared_data/Y.npy',Y)
# #
# #np.save('prepared_data/XV.npy',XV)
# #np.save('prepared_data/PV.npy',PV)
# #np.save('prepared_data/YV.npy',YV)
# #
# #np.save('prepared_data/XT.npy',XT)
# #np.save('prepared_data/PT.npy',PT)
# #np.save('prepared_data/YT.npy',YT)


# X = np.load('prepared_data/X.npy')
# P = np.load('prepared_data/P.npy')
# Y = np.load('prepared_data/Y.npy')

# XV = np.load('prepared_data/XV.npy')
# PV = np.load('prepared_data/PV.npy')
# YV = np.load('prepared_data/YV.npy')

# XT = np.load('prepared_data/XT.npy')
# PT = np.load('prepared_data/PT.npy')
# YT = np.load('prepared_data/YT.npy')

num_features = 4
model_type = 0
is_bias = True

#scores per data set, pre prune post prune, MSE + LL (test,train)
scores_np = np.zeros((10,2,4))

scores_np = np.zeros((10,2,6,15))

def return_leaf(X):
    labels = np.zeros(X.shape[0])
    labels = 1*((X[:,9]>0.5) &(X[:,7]>0.5)& (X[:,2]>0.5))
    labels += 2*((X[:,9]>0.5) &(X[:,7]>0.5)&(X[:,2]<0.5))
    labels += 3*((X[:,9]>0.5) &(X[:,7]<0.5))
    labels += 4*((X[:,9]<0.5) &(X[:,2]>0.5)&(X[:,7]>0.5)&(X[:,0]>2.5)) 
    labels += 5*((X[:,9]<0.5) &(X[:,2]>0.5)&(X[:,7]>0.5)&(X[:,0]<2.5))
    labels += 6*((X[:,9]<0.5) &(X[:,2]>0.5)&(X[:,7]<0.5)&(X[:,0]>2.5)) 
    labels += 7*((X[:,9]<0.5) &(X[:,2]>0.5)&(X[:,7]<0.5)&(X[:,0]<2.5))
    labels += 8*((X[:,9]<0.5) &(X[:,2]<0.5)&(X[:,7]>0.5)&(X[:,0]>2.5)) 
    labels += 9*((X[:,9]<0.5) &(X[:,2]<0.5)&(X[:,7]>0.5)&(X[:,0]<2.5))
    labels += 10*((X[:,9]<0.5) &(X[:,2]<0.5)&(X[:,7]<0.5)&(X[:,0]>2.5)) 
    labels += 11*((X[:,9]<0.5) &(X[:,2]<0.5)&(X[:,7]<0.5)&(X[:,0]<2.5))     
    return(labels)

for i in range(10):
#for i in [1]:
    num_features = 4
    X = np.load('prepared_data2/X'+str(i)+'.npy')
    P = np.load('prepared_data2/P'+str(i)+'.npy')
    Y = np.load('prepared_data2/Y'+str(i)+'.npy')
    
    XV = np.load('prepared_data2/XV'+str(i)+'.npy')
    PV = np.load('prepared_data2/PV'+str(i)+'.npy')
    YV = np.load('prepared_data2/YV'+str(i)+'.npy')
    
    XT = np.load('prepared_data2/XT'+str(i)+'.npy')
    PT = np.load('prepared_data2/PT'+str(i)+'.npy')
    YT = np.load('prepared_data2/YT'+str(i)+'.npy')


    P = P.astype(float)
    PV = PV.astype(float)
    PT = PT.astype(float)   
    
    P[:,-1] = 0.001*np.random.rand(P.shape[0])
    PV[:,-1] = 0.001*np.random.rand(PV.shape[0])
    PT[:,-1] = 0.001*np.random.rand(PT.shape[0])  
    
    is_continuous = [False for k in range(len(c_features))]
    is_continuous[6] = True 
    is_continuous[8] = False 
    P, PV, PT, num_features = append_customer_features_to_product_features(X, XV, XT,
                                                                           P, PV, PT,
                                                                           feats_continuous=is_continuous, 
                                                                           model_type=model_type, num_features=num_features)    
#    for depth in [4]:
#    for depth in range(15):
    labels = return_leaf(X)
    labelsT = return_leaf(XT)  
    Y_pred = np.zeros((YT.shape[0],3))
    Y_pred2 = np.zeros((Y.shape[0],3))    
    for l in range(1,12):
        #APPLY TREE ALGORITHM. TRAIN TO DEPTH 1
        my_tree = MST(max_depth = 0, min_weights_per_node = 20, only_singleton_splits = True, quant_discret = 0.05)
    #    my_tree = MST(max_depth = 12 min_weights_per_node = 20, quant_discret = 0.05)    
        my_tree.fit(X[labels == l],P[labels == l],Y[labels == l],verbose=True,
                         feats_continuous= is_continuous,
                         refit_leaves=True,only_singleton_splits = True,
                         num_features = num_features, is_bias = is_bias, model_type = model_type,
                         mode = "mnl", batch_size = 100, epochs = 50, steps = 6000,
                         leaf_mod_thresh=10000000);
        Y_pred[labelsT == l] = my_tree.predict(XT[labelsT == l],PT[labelsT == l])
        Y_pred2[labels == l] = my_tree.predict(X[labels == l],P[labels == l])        
        #ABOVE: leaf_mod_thresh controls whether when fitting a leaf node we apply Newton's method or stochastic gradient descent.
        # If the number of training observations in a leaf node <= leaf_mod_thresh, then newton's method
        # is applied; otherwise, stochastic gradient descent is applied.
                    
        ## PRINT OUT THE UNPRUNED TREE. OBSERVE THAT THE FIRST SPLIT IS CORRECT, BUT THERE ARE UNNECESSARY SPLITS AFTER THAT
        #my_tree.traverse(verbose=True)   
        ##PRUNE THE TREE    
        #my_tree.prune(XV, PV, YV, verbose=False)
        # #PRINT OUT THE PRUNED TREE. OBSERVE THAT THE UNNECESSARY SPLITS HAVE BEEN PRUNED FROM THE TREE
        #my_tree.traverse(verbose=True)
        #print(my_tree._error(XT,PT,YT, use_pruning_error = True))
        #print(my_tree._error(XT,PT,YT, use_pruning_error = False))    
        
    YT_flat = np.zeros((YT.shape[0],3))
    YT_flat[np.arange(YT.shape[0]),YT] = 1
    Y_flat = np.zeros((Y.shape[0],3))
    Y_flat[np.arange(Y.shape[0]),Y] = 1 
    
    
    s_Y = Y_pred.shape[0]
    scores_np[i,0,0,0] = np.mean(np.log(np.maximum(0.01,Y_pred[np.arange(s_Y),YT])))
    scores_np[i,0,1,0] = np.mean(np.sum(np.power(Y_pred-YT_flat,2),axis = 1))
    
    s_Y = Y_pred2.shape[0]
    scores_np[i,0,2,0] = np.mean(np.log(np.maximum(0.01,Y_pred2[np.arange(s_Y),Y])))
    scores_np[i,0,3,0] = np.mean(np.sum(np.power(Y_pred2-Y_flat,2),axis = 1))
    
    
        
      

   
