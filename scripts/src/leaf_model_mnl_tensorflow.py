import numpy as np
#import pandas as pd
import copy
#import pylogit as pl 
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from functools import reduce 

# from sklearn.linear_model import LogisticRegression
# from cvxopt import solvers, matrix, spdiag, log, exp, div
#from data_model import DataModel
from collections import OrderedDict    # For recording the model specification 

'''
MST depends on the classes and functions below. 
These classes/methods are used to define the leaf model object in each leaf node,
as well as helper functions for certain operations in the tree fitting procedure.

One can feel free to edit the code below to accommodate any leaf node model.
The leaf node model used here is an mnl model fit on data (A,Y). A is a matrix of 
dimension num-observations x (num-product-features+1)*num-products. A[:,:num-products]
denotes the availability data encoding which subsets of products are offered to each customer.
A[:,num-products*f:num-products*(f+1)] encodes the product feature f for each alternative.
(A is are the decisions "P" in the paper)
Y is a vector with each element in {0,...,K-1} corresponding to the customer's choice.

Summary of methods and functions to specify:
  Methods as a part of class LeafModel: fit(), predict(), to_string(), error(), error_pruning()
  Other helper functions: get_sub(), are_Ys_diverse()
  
'''


def exponomial_tf(features,labels,mode,params):
    '''
    Exponomial computational graph
    
    Input 
    features is a Dict of tensors, with the training set of shape (batch x 2n) 
    labels is the vector of chosen items of shape (batch,)       
    mode is either PREDICT or TRAIN   
    
    Returns a tensorflow estimator object  
    '''
    model_type = params[0]    
    n = params[1]    
    num_features = params[2]
    is_bias = params[3]
    # initialize a tensorflow graph
    input_features = tf.reshape(features["x"],[-1,num_features,n])
    learning_rate = params[4] 
#    learning_rate = 0.0003    
    m = params[5]
#    print("Batch size "+str(n) +" "+str(m))
    
    # Variables.
    if model_type == 0:
        weights = tf.Variable(tf.truncated_normal([1,num_features - 1 +is_bias,n]
                               ,stddev = 0.), name ='weights')
        weight_scaled = tf.tensordot(tf.ones([m,1]),weights,axes = [[1],[0]])
        #    biases = tf.Variable(tf.truncated_normal([1,n],stddev = 0.01), name = 'biases')
 
        # Training computation.
#        utilities = tf.matmul(input_features[:,1,:], tf.diag(weights))+ \
#                    tf.matmul(tf.ones([m,1]),biases)
    elif model_type == 1:
        weights = tf.Variable(tf.truncated_normal([1,num_features - 1 +is_bias,1]
                               ,stddev = 0.), name ='weights')
        weight_scaled = tf.tensordot(
                            tf.tensordot(tf.ones([m,1]),weights,axes = [[1],[0]]),
                            tf.ones([1,n]), axes = [[2],[0]])
    
    if is_bias:
            utilities = tf.reduce_sum(tf.multiply(input_features, weight_scaled)
                                        ,axis = 1)
    else:
            utilities = tf.reduce_sum(tf.multiply(input_features[:,1:,:]
                                               , weight_scaled),axis = 1)
     
    # Predictions for the training, validation, and test data.
    masked_utilities = tf.multiply(utilities,tf.cast(input_features[:,0,:]>0, tf.float32))\
                                     -10e9*tf.cast(input_features[:,0,:]<=0, tf.float32)
#     sorted_utilities = tf.py_func(lambda x: np.sort(x,axis =1), [masked_utilities], tf.float32, stateful=False)
#     sorted_utilities.set_shape(masked_utilities.get_shape())
#     sorted_utilities = tf.reverse(sorted_utilities,[1])
    sorted_utilities = tf.nn.top_k(masked_utilities,k=n,sorted=True,name=None)
#     sorted_utilities.set_shape(masked_utilities.get_shape())
    arg_sort_utilities = tf.reverse(tf.nn.top_k(sorted_utilities[1],k=n,sorted=True,name=None)[1],[1])
#     arg_sort_utilities = tf.py_func(lambda x: np.int32(np.argsort(np.argsort(x,axis =1)[::-1],axis=1)), [masked_utilities], tf.int32, stateful=False)
#     arg_sort_utilities.set_shape(masked_utilities.get_shape())
    
    scale_utilities =tf.convert_to_tensor(np.tile(np.arange(1,n+1),(m,1)), dtype=tf.float32)
    exp_utilities = tf.exp(tf.multiply(scale_utilities,sorted_utilities[0])-tf.cumsum(sorted_utilities[0],axis = 1))

    mat_transf = np.diag(np.ones(n))
    arr_transf = np.ones((n,1))
    
    for i in range(1,n):
        mat_transf[:i,i] = -1.0/i
        arr_transf[i,0] = 1.0/(i+1)
#     print mat_transf 
#     print arr_transf
    mat_transf =tf.convert_to_tensor(np.tile(mat_transf,(m,1,1)), dtype=tf.float32)
    arr_transf =tf.convert_to_tensor(np.tile(arr_transf,(m,1,1)), dtype=tf.float32)
    sorted_probas = tf.matmul(mat_transf,tf.multiply(arr_transf,tf.reshape(exp_utilities,[-1,n,1])))

    list_order = []
    for i in range(m):
        list_order.append(tf.gather(sorted_probas[i,:,0],arg_sort_utilities[i,:]) )
#        list_order.append(sorted_probas[i,arg_sort_utilities[i,:],0])
    probas = tf.stack(list_order, axis = 0, name="softmax_tensor")
#    probas = tf.identity(probas, name="softmax_tensor")
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=probas, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": probas,
      "utilities": utilities,
      "arg_sort_utilities":arg_sort_utilities,
      "exp_utilities":exp_utilities,
      "masked_utilities":masked_utilities          
      }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.reduce_mean([-tf.log(tf.maximum(1e-9,probas[i,tf.cast(labels[i],tf.int32)])) for i in range(m)])
#    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, 
#                                                  logits=tf.log(probas))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)        
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    ordered_probas = tf.nn.top_k(probas,k=n,sorted=True,name=None)[1]
    rank_product = tf.reverse(tf.nn.top_k(ordered_probas,k=n,sorted=True,name=None)[1],[1])
    list_order = []
#    i = 0
    list_order = tf.map_fn(lambda x:tf.gather(x[0],[x[1]])[0],(rank_product,labels),
                           dtype = tf.int32)
#    for l in labels:
#        list_order.append(tf.gather(rank_product[i,:],l))
#        i+=1
    ranks_label = tf.stack(list_order)
    ranks_norm = tf.divide(tf.cast(ranks_label,tf.float32),
                           tf.reduce_sum(tf.cast(input_features[:,0,:]>0, 
                                                      tf.float32),axis = 1))

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
            labels= labels, predictions=predictions["classes"]),
            "average_rank": tf.metrics.mean(ranks_label),
            "average_perc": tf.metrics.mean(ranks_norm)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def mnl_tf(features,labels,mode,params):
    '''
    MNL computational graph
    
    Input 
    features is a Dict of tensors, with the training set of shape (batch x 2n) 
    labels is the vector of chosen items of shape (batch,)       
    mode is either PREDICT or TRAIN   
    
    Returns a tensorflow estimator object  
    '''
    # jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    
    model_type = params[0]    
    n = params[1]    
    num_features = params[2]
    is_bias = params[3]
    learning_rate = params[4]    
#    learning_rate = 0.002
#    fit_init = params[4]
    
    input_features = tf.reshape(features["x"],[-1,num_features,n])
    m = tf.shape(input_features)[0]
#    print("Batch size "+str(n) +" "+str(m))
    
    # with jit_scope():
    weight_data = tf.reshape(features["weight_data"],[-1,1])  
    
    # initialize a tensorflow graph
    # Variables.
    if model_type == 0:
        # with jit_scope():
            weights = tf.Variable(tf.truncated_normal([1,num_features - 1 +is_bias,n]
                                   ,stddev = 0.), name ='weights')
            weight_scaled = tf.tensordot(tf.ones([m,1]),weights,axes = [[1],[0]])
        #    biases = tf.Variable(tf.truncated_normal([1,n],stddev = 0.01), name = 'biases')
 
        # Training computation.
#        utilities = tf.matmul(input_features[:,1,:], tf.diag(weights))+ \
#                    tf.matmul(tf.ones([m,1]),biases)
    elif model_type == 1:
        # with jit_scope():
#            weights = tf.Variable(tf.reshape(fit_init,[1,num_features - 1 +is_bias,1]),name = 'weights')
            weights = tf.Variable(tf.truncated_normal([1,num_features - 1 +is_bias,1]
                                   ,stddev = 0.), name ='weights')
            weight_scaled = tf.tensordot(
                                tf.tensordot(tf.ones([m,1]),weights,axes = [[1],[0]]),
                                tf.ones([1,n]), axes = [[2],[0]])
    
    if is_bias:
        # with jit_scope():
            logits = tf.reduce_sum(tf.multiply(input_features, weight_scaled),axis = 1)
    else:
        # with jit_scope():
            logits = tf.reduce_sum(tf.multiply(input_features[:,1:,:]
                                           , weight_scaled),axis = 1)
    
  
#        # Variables.
#        weights = tf.Variable(tf.truncated_normal([n],stddev = 0.1), name ='weights')
#        biases = tf.Variable(tf.truncated_normal([1,n]), name ='biases')
 
#    # Training computation.
#    with jit_scope():
#        logits = tf.matmul(input_features[:,1,:], tf.diag(weights))+ tf.matmul(tf.ones([m,1]),biases)
     
    # Predictions for the training, validation, and test data.
    # with jit_scope():
    masked_logits = tf.multiply(logits,tf.cast(input_features[:,0,:]>0, tf.float32))\
                                         -10e9*tf.cast(input_features[:,0,:]<=0, tf.float32)
#        probas0 =tf.multiply(tf.nn.softmax(logits, name="softmax_tensor"),
#                            tf.cast(input_features[:,0,:]>0, tf.float32))
#        probas =tf.multiply(tf.nn.softmax(logits, name="softmax_tensor"),
#                            tf.cast(input_features[:,0,:]>0, tf.float32))
    probas = tf.nn.softmax(masked_logits, name="softmax_tensor")
#        probas = tf.divide(probas0,tf.tensordot(tf.reshape(tf.reduce_sum(probas0, axis = 1),[-1,1]), tf.ones([1,n]),axes = [[1],[0]]))
        
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=masked_logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": probas,
      "logits": logits,
#      "masked_logits":masked_logits,
      "weight_scaled":weight_scaled
      }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # with jit_scope():
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, 
                                                      logits=masked_logits,
                                                      weights = weight_data,
                                                      reduction = tf.losses.Reduction.NONE)
#        loss = tf.reduce_mean(tf.minimum(-tf.log(0.001),loss))
    loss = tf.reduce_mean(loss)
#        list_vals= tf.map_fn(lambda x:tf.gather(x[0],[x[1]])[0],(probas,labels),
#                           dtype = tf.float32)
#        list_vals2 = tf.stack(list_vals)
#        loss = -tf.reduce_mean(tf.log(tf.maximum(0.001,list_vals2))) 
    # with jit_scope():
        # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
#            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    ordered_probas = tf.nn.top_k(probas,k=n,sorted=True,name=None)[1]
    rank_product = tf.reverse(tf.nn.top_k(ordered_probas,k=n,sorted=True,name=None)[1],[1])
    list_order = []
#    i = 0
#    for l in labels:
#        list_order.append(tf.gather(rank_product[i,:],l))
#        i+= 1
    list_order = tf.map_fn(lambda x:tf.gather(x[0],[x[1]])[0],(rank_product,labels),
                           dtype = tf.int32)
    ranks_label = tf.stack(list_order)
    ranks_norm = tf.divide(tf.cast(ranks_label,tf.float32),
                           tf.reduce_sum(tf.cast(input_features[:,0,:]>0, 
                                                      tf.float32),axis = 1))

    
  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
            labels= labels, predictions=predictions["classes"], 
            weights = weight_data),
            "average_rank": tf.metrics.mean(ranks_label, weights = weight_data),
            "average_perc": tf.metrics.mean(ranks_norm)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
     

class LeafModelTensorflow(object):
    
    #Any additional args passed to MST's init() function are directly passed here
    def __init__(self):
        return
    
    '''
    This function trains the leaf model on the data (A,Y,weights).
    
    A and Y can take any form (lists, matrices, vectors, etc.). For our applications, I recommend making Y
    the response data (e.g., choices) and A alternative-specific data (e.g., prices, choice sets)
    
    weights: a numpy array of case weights. Is 1-dimensional, with weights[i] yielding 
      weight of observation/customer i. If you know you will not be using case weights
      in your particular application, you can ignore this input entirely.
    
    refit: boolean which equals True iff leaf model is being refit after tree splits have been decided. Since
      the tree split evaluation process requires fitting a large number of leaf models, one might wish to 
      fit the leaf models on only a subset of data or for less training iterations when refit=False. Practitioners
      can feel free to ignore this parameter in their leaf model design.
    
    Returns 0 or 1.
      0: No errors occurred when fitting leaf node model
      1: An error occurred when fitting the leaf node model (probably due to insufficient data)
    If fit returns 1, then the tree will not consider the split that led to this leaf node model
    
    fit_init is a LeafModel object which represents a previously-trained leaf node model.
    If specified, fit_init is used for initialization when training this current LeafModel object.
    Useful for faster computation when fit_init's coefficients are close to the optimal solution of the new data.
    
    For those interested in defining their own leaf node functions:
      (1) It is not required to use the fit_init argument in your code
      (2) All edge cases must be handled in code below (ex: arguments
          consist of a single entry, weights are all zero, Y has one unique choice, etc.).
          In these cases, either hard-code a model that works with these edge-cases (e.g., 
          if all Ys = 1, predict 1 with probability one), or have the fit function return 1 (error)
      (3) Store the fitted model (or its coefficients) as an attribute to the self object. You can name the attribute
          anything you want (i.e., it does not have to be self.model_obj below),
          as long as its consistent with your predict_prob() and to_string() methods
    
    Any additional args passed to MST's fit() function are directly passed here
    '''
    def fit(self, A, Y, weights, fit_init=None, refit=False, mode = "mnl",
            batch_size = 50, path = "", model_type = 0, num_features = 2, 
            epochs = 10, steps = 20000, steps_refit = 60000, learning_rate = 0.01,
            learning_rate_refit = 0.001, is_bias = True, **kwargs):
        '''
        Fits a multinomial logistic regression to the data (A,Y).
        
        Here, A is a matrix m-by-(nxp), where n is the number of alternatives, 
          p is the number of product features and m is the number of observations.
        The columns of A are ordered in lexicographic order (feature, product). 
          For example, if p=2 and n=3, then the columns would be:
          (feat 1 prod 1) (feat 1 prod 2) (feat 1 prod 3) (feat 2 prod 1) (feat 2 prod 2) (feat 2 prod 3)
        The first product feature MUST correspond to binary indicator of availability of the product in the assortment
          For example with 3 products, if A[0,:3] = [1,0,1], then products 1 and 3 are in the assortment whereas product 2 was excluded
          If all products are in the assortment at all times (i.e., no time-varying assortments), then include this assortment availability
            feature but set equal to all ones.
        
        Y is an m-sized 1 dimensional vector indicating the index of the purchase decision {0,1,...,n-1} 
        weights are vector of data weights 
        num_features:  number of product features (integer), INCLUDING the binary availability feature
        
        model_type: whether the model has alternative varying coefficients (0) or not (1). Type integer (0/1)
          (default is 0 meaning each alternative has a separate coeff)
        is_bias: whether the utility function has an intercept (default is True). Type boolean (True/False).
        mode : "mnl" or "exponomial" (default is "mnl")
        epochs : number of epochs for the estimation (default is 10) 
        batch_size : size of the stochastic batch (default is 50,)
        
        Stores fitted model using two attributes:
        (1) self.model_obj: the model object outputted from tensor_flow
        (2) self.model_coef: the coefficients of the regression model, [interceptA, interceptB, etc., price_elasticityA, price_elasticityB, etc]
        '''
#         initialize a tensorflow graph
        try:
            if refit == True:
                steps = steps_refit
                learning_rate = learning_rate_refit
            
            config = tf.ConfigProto()
            # config = tf.compat.v1.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            # config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
            es_congig =tf.estimator.RunConfig(session_config = config)
            n = int(A.shape[1]/num_features)
            # tf.logging.set_verbosity(tf.logging.ERROR)
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            if mode == "mnl":
                model_estimator = tf.estimator.Estimator(
                    model_fn=mnl_tf, model_dir=path, params= [model_type,n,num_features,is_bias,learning_rate],
                    config = es_congig)
            elif mode == "exponomial":
                model_estimator = tf.estimator.Estimator(
                    model_fn=exponomial_tf, model_dir=path, params= [model_type,n,num_features,is_bias,learning_rate,batch_size],
                    config = es_congig)
                
            # Set up logging for predictions
            # Log the values in the "Softmax" tensor with label "probabilities"
            tensors_to_log = {"probabilities": "softmax_tensor"}
            logging_hook = tf.train.LoggingTensorHook(
                  tensors=tensors_to_log, every_n_iter=50)
            
            exact_size = int((A.shape[0]/batch_size)*batch_size)
            
            if exact_size> 0:
                  # Train the model
                train_input_fn = tf.estimator.inputs.numpy_input_fn(
                      x={"x": np.float32(A[:exact_size,:]),
                         "weight_data":np.float32(weights[:exact_size])},
                      y=np.int32(Y[:exact_size]),
                      batch_size=batch_size,
                      num_epochs=None,              
                      shuffle=True)
                
                model_estimator.train(
                      input_fn=train_input_fn,
                       steps=steps,
                      hooks=[logging_hook])
                
                # Evaluate the model and print results
                eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                      x={"x": np.float32(A[:exact_size,:]),
                         "weight_data":np.float32(weights[:exact_size])},
                      y=np.int32(Y[:exact_size]),
                      num_epochs=1,
                      batch_size=batch_size,
                      shuffle=False)
                eval_results = model_estimator.evaluate(input_fn=eval_input_fn)
                
        #        print(eval_results)                 
                
    #            biases = model_estimator.get_variable_value("biases").reshape((n,))
                params_model = model_estimator.get_variable_value("weights")
                size_issue = False
            else:
                size_issue = True
                params_model = np.zeros(1)
    #        print biases.shape, weights.shape
#            params_model = np.concatenate((biases,weights_price))
            if size_issue|(np.sum(np.isnan(params_model)) > 0):
              return(1)
            else:
              self.n = n
              self.model_obj = model_estimator
              self.model_coef = params_model
              self.model_quality = eval_results
              return(0) 
        except:
            raise
            return(1)
    
    '''
    This function applies model from fit() to predict response data given new data A.
    Returns a numpy vector/matrix of response probabilities (one list entry per observation, i.e. l[i] yields prediction for ith obs.).
    Note: make sure to call fit() first before this method.
    
    Any additional args passed to MST's predict() function are directly passed here
    '''
    def predict(self, A, batch_size =50, name = 'probabilities', *args,**kwargs):
        '''
        This function applies model from fit() to predict choice data given new data A.
          Returns a list/numpy array of choices (one list entry per observation, i.e. l[i] yields prediction for ith obs.).
          Note: make sure to call fit() first before this method.
  
        Any additional args passed to ChoiceModelTree's predict() function are directly passed here
        '''
        
        exact_size = int(np.ceil(A.shape[0]/float(batch_size)))*batch_size
        q = int(np.ceil(exact_size/float(A.shape[0])))
        B = np.tile(A,(q,1))[:exact_size,:]
#        print A.shape, B.shape,exact_size,q
        #Predicts choice proba values m-by-n given vector of new prices A, using coefficients self.model_coef.
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
              x={"x": np.float32(B), "weight_data": np.ones(B.shape[0])}, 
              y= None, shuffle = False, batch_size = batch_size)
        predictions = self.model_obj.predict(eval_input_fn)
#        i = 0
#        for p in predictions:
#            i +=1
        pred_mat = np.zeros((B.shape[0],self.n))
        i = 0
        for p in predictions:
            pred_mat[i,:] = p[name]
            i +=1
        pred_mat = np.maximum(0.00001,pred_mat)
        pred_mat = np.divide(pred_mat,np.sum(pred_mat,axis=1).reshape((B.shape[0],1))*np.ones((1,self.n)))
        return(pred_mat[:A.shape[0],:])
    
    '''
    This function outputs the errors for each observation in pair (A,Y).  
    Used in training when comparing different tree splits.
    Ex: log-likelihood between observed data Y and predict(A)
    Any error metric can be used, so long as:
    (1) lower error = "better" fit
    (2) error >= 0, where error = 0 means no error
    (3) error of fit on a group of data points = sum(errors of each data point)

    How to pass additional arguments to this function: simply pass these arguments to the init()/fit() functions and store them
    in the self object.
    '''
    def error(self,A,Y):  
#        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#              x={"x": np.float32(A), "weight_data": np.ones(A.shape[0])}, 
#              y= Y, shuffle = False, batch_size = 1)
#        evaluation = self.model_obj.evaluate(eval_input_fn)
        Ypred = self.predict(A)
#        log_probas = -np.log(np.maximum(Ypred[(np.arange(Y.shape[0]),Y)],0.001))
        log_probas = -np.log(Ypred[(np.arange(Y.shape[0]),Y)])
        return(log_probas)
    
    '''
    Specifies error metric used in pruning the tree rather than comparing different tree splits. See error() above
    for proper specification instructions.
    '''
    def error_pruning(self,A,Y):
        #Here I define the error metric to be weighted mean-square error (brier score)
        Ypred = self.predict(A)
        Z = np.zeros(Ypred.shape)
#        errors = (1-Ypred[(np.arange(Y.shape[0]),Y)])**2.0
        Z[(np.arange(Y.shape[0]),Y)] = 1.0
        errors = np.sum((Z-Ypred)**2,axis = 1)
        return errors
    
    
    def to_string(self,*leafargs,**leafkwargs):
        return("Model parameters: " + reduce(lambda x,y: x + '_' + str(y), self.model_coef,""))
    
#    #not needed to specify for other leaf models
#    def eval_model(self, A,Y, weights = None,batch_size = 100):
#        '''
#        Evaluates the model on a holdout dataset
#        '''
#        
#        eval_size = (A.shape[0]/batch_size)*batch_size
#        if eval_size < A.shape[0]:
#            c =  eval_size + batch_size - A.shape[0]
#            B = np.concatenate((A,A[:c,:]),axis = 0)
#            Y2 = np.concatenate((Y,Y[:c]),axis = 0)
#        else:
#            B = A
#            Y2 = Y
#        
#        if weights == None:
#            weights = np.ones(B.shape[0])        
#        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#                  x={"x": np.float32(B),
#                     "weight_data":np.float32(weights)},
#                  y=np.int32(Y2),
#                  num_epochs=1,
#                  batch_size=batch_size,
#                  shuffle=False)
#        eval_results = self.model_obj.evaluate(input_fn=eval_input_fn)
#        return(eval_results)
        

