import pandas as pd                    # For file input/output
import numpy as np                     # For vectorized math operations
                                       
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

importr('mlogit')
importr('mnlogit')
pandas2ri.activate()
ro.r.source("src/newmnlogit.R") #rewrites some functions from the mnlogit package to accommodate varying choice sets

'''
MST depends on the classes and functions below. 
These classes/methods are used to define the leaf model object in each leaf node,
as well as helper functions for certain operations in the tree fitting procedure.

One can feel free to edit the code below to accommodate any leaf node model.
The leaf node model is fit on data (A,Y). (A is are the decisions "P" in the paper).
Make sure to add an import statement to mst.py importing this leaf model class.

Summary of methods and functions to specify:
  Methods as a part of class LeafModel: fit(), predict(), to_string(), error(), error_pruning()
  Other helper functions: get_sub(), are_Ys_diverse()
  
'''

'''
LeafModel: the model used in each leaf. 
Has five methods: fit, predict, to_string, error, error_pruning
'''
class LeafModelRMNLogit(object):
  
  #Any additional args passed to MST's init() function are directly passed here
  def __init__(self,*args,**kwargs):
    self.model_fit = False
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
  def fit(self, A, Y, weights, fit_init=None, refit=False, 
          model_type=0, num_features=4, is_bias=True, **kwargs):
    
    #no need to refit this model since it is already fit to optimality
    #note: change this behavior if debias=TRUE
    if refit == True and self.model_fit == True:
      return(0)
    
    self.num_features = num_features
    n_items = int(A.shape[1]/num_features)
    
    #input verification
    #(1) check that if num_features=1 (no product features) then is_bias=True
    if num_features-1 == 0 and is_bias == False:
      import sys
      sys.exit("Error: num_features=1 and is_bias=False")
    
    #(2) check that there are no varying choice subsets (mnlogit cannot handle this case)
#    if not np.all(A[:,:n_items]==1):
#      import sys
#      sys.exit("Error: mnlogit does not support varying choice sets")
    
    try:
      #first, remove any alternatives which have not been chosen from the model
      #(mnlogit crashes when presented with such alternatives)
      if model_type == 1 and is_bias == False:
        selected_alts = np.in1d(range(n_items),range(n_items))
        self.selected_alts = selected_alts
      else:
        selected_alts = np.in1d(range(n_items),Y)
        A = A[:,np.tile(selected_alts,num_features)]
        self.selected_alts = selected_alts      
        Y = pd.factorize(Y,sort=True)[0]
      
      if len(np.unique(Y)) == 1:
        self.mymnl = None
        self.only_choice = np.unique(Y)[0]
        self.model_fit = True
        return(0)      
      
      df_long, weights = data2long_format(A, Y=Y, num_features=num_features, weights=weights)
      
      #create formula
      if num_features-1 == 0:
        form = "choice ~ Avl|1"
      else:
        prod_feat_str = ""
        for f in range(1,num_features-1):
          prod_feat_str = prod_feat_str + "F" + str(f) + "+"
        prod_feat_str = prod_feat_str + "F" + str(num_features-1)
        
        if model_type == 0:
          form = "choice ~ Avl" + "|" + str(int(is_bias)) + "|" + prod_feat_str
        else:
          form = "choice ~ " + prod_feat_str + "+ Avl" + "|" + str(int(is_bias))
      
      try:
        weights_all_same = len(np.unique(weights)) == 1
        if weights_all_same:
          mymnl = ro.r.mnlogit(ro.Formula(form), data=ro.DataFrame(df_long), choiceVar='alt')
        else:
          mymnl = ro.r.mnlogit(ro.Formula(form), data=ro.DataFrame(df_long), choiceVar='alt', weights=ro.FloatVector(weights))
      except:
        #fitting issue likely due to features being linear combinations of each other.
        #try adding random noise to each feature and refitting
        df_long['Avl'] = df_long['Avl'] + np.random.uniform(low=-0.001, high=0.001, size=len(df_long['Avl']))
        for f in range(1,num_features):
          fname = 'F'+str(f)
          df_long[fname] = df_long[fname] + np.random.uniform(low=-0.001, high=0.001, size=len(df_long[fname]))
        if weights_all_same:
          mymnl = ro.r.mnlogit(ro.Formula(form), data=ro.DataFrame(df_long), choiceVar='alt')
        else:
          mymnl = ro.r.mnlogit(ro.Formula(form), data=ro.DataFrame(df_long), choiceVar='alt', weights=ro.FloatVector(weights))
      #print(ro.r.summary(mymnl)) #see summary statistics for fit
      
      self.mymnl = mymnl
      
      self.model_fit = True
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
  def predict(self, Anew, *args,**kwargs):
    
    #bug in current predict function which doesn't allow Anew to be a single observation.
    #hack fix: duplicate it into two observations
    Anew_singleton = False
    if Anew.shape[0] == 1:
#      print("Anew_singleton = True")
      Anew_singleton = True
      Anew = np.concatenate((Anew,Anew), axis=0)
    
    if self.mymnl is None:
      #verify the single choice seen in the training set is in all given assortments
      if not np.all(Anew[:,self.only_choice]):
        #import sys
        #sys.exit("Error: presented an assortment for which no product in assortment was chosen in training set")
        inds2correct = np.where(np.logical_not(Anew[:,self.only_choice]))[0]
      else:
        inds2correct = None
        
      #predict with probability 1 the single choice seen in the training set
      Ypred = np.zeros((Anew.shape[0],len(self.selected_alts)))
      Ypred[:,self.only_choice] = 1.0
      
      if inds2correct is not None:
        n_items = int(Anew.shape[1]/self.num_features)
        Ypred[inds2correct,:] = np.multiply(Anew[inds2correct,:n_items],(1.0/np.sum(Anew[inds2correct,:n_items],axis=1))[:,np.newaxis])
      
      return(Ypred)
    
    A = Anew[:,np.tile(self.selected_alts,self.num_features)] #take away attributes to match fit() dataset
    
    num_selected_alts = sum(self.selected_alts)
    if not np.all(np.sum(A[:,:num_selected_alts],axis=1)):
      #import sys
      #sys.exit("Error: presented an assortment for which no product in assortment was chosen in training set")
      inds2correct = np.where(np.logical_not(np.sum(A[:,:num_selected_alts],axis=1)))[0]
    else:
      inds2correct = None
    
    df_long = data2long_format(A, num_features=self.num_features)
    
    Ypred = np.array(ro.r.predict(self.mymnl, newdata=ro.DataFrame(df_long), choiceVar='alt'))
    
    #add back in missing attributes with predicted probabilities of 0.0
    Ypred2 = np.zeros((Ypred.shape[0],len(self.selected_alts)))
    Ypred2[:,self.selected_alts] = Ypred
    
    if inds2correct is not None:
      n_items = int(Anew.shape[1]/self.num_features)
      Ypred2[inds2correct,:] = np.multiply(Anew[inds2correct,:n_items],(1.0/np.sum(Anew[inds2correct,:n_items],axis=1))[:,np.newaxis])
    
    if Anew_singleton is True:
      Ypred2 = Ypred2[0,None]
    
    return(Ypred2)
  
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
    Ypred = self.predict(A)
    log_probas = -np.log(np.maximum(Ypred[(np.arange(Y.shape[0]),Y)],0.01))
#    log_probas = -np.log(Ypred[(np.arange(Y.shape[0]),Y)])
    return(log_probas)
  
  '''
  This function outputs the errors for each observation in pair (A,Y).  
  Used in pruning to determine the best tree subset.
  Ex: mean-squared-error between observed data Y and predict(A)
  Any error metric can be used, so long as:
    (1) lower error = "better" fit
    (2) error >= 0, where error = 0 means no error
    (3) error of fit on a group of data points = sum(errors of each data point)
  
  How to pass additional arguments to this function: simply pass these arguments to the init()/fit() functions and store them
  in the self object.
  '''
  def error_pruning(self,A,Y):
    #Here I define the error metric to be weighted mean-square error (brier score)
    Ypred = self.predict(A)
    Z = np.zeros(Ypred.shape)
#   errors = (1-Ypred[(np.arange(Y.shape[0]),Y)])**2.0
    Z[(np.arange(Y.shape[0]),Y)] = 1.0
    errors = np.sum((Z-Ypred)**2.0,axis = 1)
    return errors
  
  '''
  This function returns the string representation of the fitted model
  Used in traverse() method, which traverses the tree and prints out all terminal node models
  
  Any additional args passed to MST's traverse() function are directly passed here
  '''
  def to_string(self,*leafargs,**leafkwargs):
    if self.mymnl is None:
      return("Model params: predicts choice " + str(self.selected_alts.argmax()) + " with prob. 1" )
    else:
      #return("Model params: " + str(np.array(self.mymnl.rx2('coefficients'))))
      return("Model params: " + str(self.mymnl.rx2('coefficients')))
    

'''
functions for assistance to leaf model class
'''

def check_columns_oneunqval(A):
  return np.any(np.all(A == A[0,:], axis = 0))

def data2long_format(A, Y=None, num_features=2, weights=None):
  n_items = int(A.shape[1]/num_features)
  
  if Y is None:
    #mnlogit expects a column of choice data in the data frame even in prediction.
    #Create dummy choices
    avails = A[:,:n_items]
    Y = avails.argmax(axis=1)
  
  #if (weights is not None) and (num_features > 1):
    #mnlogit crashes if any of the product features have exactly 1 unique value. 
    #To prevent this, include a dummy observation
    #of weight (approx) zero with unique values for each feature
    #if check_columns_oneunqval(A[:,n_items:]):
      #Y = np.append(Y,0)
      #weights = np.append(weights,min(weights)/100000.0)
      #Adummy =  A[0,n_items:] + np.random.uniform(low=-0.001, high=0.001, size=(num_features-1)*n_items)
      #Adummy = np.concatenate((np.ones(n_items),Adummy))
      #A = np.vstack((A,Adummy))
    
  df_long = {}
  for f in range(0,num_features):
    A_long_f = A[:,(n_items*f):(n_items*(f+1))].reshape(-1)
    if f == 0:
      fname = 'Avl'
    else:
      fname = 'F'+str(f)
    df_long[fname] = A_long_f
  
  alt = np.tile(range(1,n_items+1), A.shape[0])
  df_long['alt'] = alt
  
  if Y is not None:
    choice = np.zeros((A.shape[0],n_items), dtype=bool)
    choice[(range(A.shape[0]),Y)] = True
    choice = choice.reshape(-1)
  
    df_long['choice'] = choice
    
  if weights is None:
    return(df_long)
  else:
    return(df_long,weights)
      