
import numpy as np
from src.mst import MST
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

#scores per data set, pre prune post prune, MSE + LL (test,train)
scores_np = np.zeros((10,2,4))

scores_np = np.zeros((10,2,6,15))

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


    is_continuous = [False for k in range(len(c_features))]
    is_continuous[6] = True 
    is_continuous[8] = False 
#    
#    P, PV, PT, num_features = append_customer_features_to_product_features(X, XV, XT,
#                                                                           P, PV, PT,
#                                                                           feats_continuous=is_continuous, 
#                                                                           model_type=model_type, num_features=num_features)    
    for depth in [14]:  
#    for depth in range(15):               
        #APPLY TREE ALGORITHM. TRAIN TO DEPTH 1
        my_tree = MST(max_depth = depth, min_weights_per_node = 50, only_singleton_splits = True, quant_discret = 0.05)
        my_tree.fit(X,P,Y,verbose=True,
                         feats_continuous= is_continuous,
                         refit_leaves=True,only_singleton_splits = True,
                         num_features = num_features, is_bias = is_bias, model_type = model_type,
                         mode = "mnl", batch_size = 100, epochs = 50, steps = 6000,
                         leaf_mod_thresh=10000000);
        #ABOVE: leaf_mod_thresh controls whether when fitting a leaf node we apply Newton's method or stochastic gradient descent.
        # If the number of training observations in a leaf node <= leaf_mod_thresh, then newton's method
        # is applied; otherwise, stochastic gradient descent is applied.
                       
        
        YT_flat = np.zeros((YT.shape[0],3))
        YT_flat[np.arange(YT.shape[0]),YT] = 1
        Y_flat = np.zeros((Y.shape[0],3))
        Y_flat[np.arange(Y.shape[0]),Y] = 1 
        
        Y_pred = my_tree.predict(XT,PT)
        s_Y = Y_pred.shape[0]
        scores_np[i,0,0,depth] = np.mean(np.log(np.maximum(0.01,Y_pred[np.arange(s_Y),YT])))
        scores_np[i,0,1,depth] = np.mean(np.sum(np.power(Y_pred-YT_flat,2),axis = 1))
        
        Y_pred = my_tree.predict(X,P)
        s_Y = Y_pred.shape[0]
        scores_np[i,0,2,depth] = np.mean(np.log(np.maximum(0.01,Y_pred[np.arange(s_Y),Y])))
        scores_np[i,0,3,depth] = np.mean(np.sum(np.power(Y_pred-Y_flat,2),axis = 1))
        
        scores_np[i,0,4,depth] = sum(map(lambda x: x.is_leaf,my_tree.tree))
        scores_np[i,0,5,depth] = sum(map(lambda x: x.alpha_thresh < np.inf,my_tree.tree))
        
#        my_tree.traverse(verbose=True)
        
        # Post-pruning metrics
        my_tree.prune(XV, PV, YV, verbose=False)
        
#        my_tree.traverse(verbose=True)
        
        Y_pred = my_tree.predict(XT,PT)
        s_Y = Y_pred.shape[0]
        scores_np[i,1,0,depth] = np.mean(np.log(np.maximum(0.01,Y_pred[np.arange(s_Y),YT])))
        scores_np[i,1,1,depth] = np.mean(np.sum(np.power(Y_pred-YT_flat,2),axis = 1))
        
        Y_pred = my_tree.predict(X,P)
        s_Y = Y_pred.shape[0]
        scores_np[i,1,2,depth] = np.mean(np.log(np.maximum(0.01,Y_pred[np.arange(s_Y),Y])))
        scores_np[i,1,3,depth] = np.mean(np.sum(np.power(Y_pred-Y_flat,2),axis = 1))   
        
        scores_np[i,1,4,depth] = sum(map(lambda x: x.is_leaf,my_tree.tree))
        scores_np[i,1,5,depth] = sum(map(lambda x: x.alpha_thresh < np.inf,my_tree.tree))        

   
