import numpy as np
#Class for generating random MNL models. Helper class for cmt_example.py.
class GenMNL(object):
  
  '''
  Arguments specifying MNL model type
  n_items: number of products
  num_features:  number of product features (integer), INCLUDING the binary availability feature
  model_type : whether the model has alternative varying coefficients (0) or not (1) 
    (default is 0 meaning each alternative has a separate coeff)
  is_bias : whether the utility function has an intercept (default is True)
  '''
  def __init__(self, n_items, num_features, model_type, is_bias):
    
    self.n_items = n_items
    self.num_features = num_features
    self.model_type = model_type
    self.is_bias = is_bias
    #generate MNL's coefficients randomly
    self.Beta = np.random.uniform(low=-1, high=1, size=(n_items,num_features))
  
  '''
  Get choice probabilities from product features P
  '''
  def get_choice_probs(self, P):
    n = P.shape[0]
    n_items = self.n_items
    num_features = self.num_features
    model_type = self.model_type
    is_bias = self.is_bias
    Beta = self.Beta
    
    U_exp = np.zeros((n,n_items))
  
    for k in range(n_items):
      if is_bias == True:
        U_exp[:,k] = Beta[k,0]
      else:
        U_exp[:,k] = 0.0
      for l in range(num_features-1):
        if model_type == 0:
          U_exp[:,k] = U_exp[:,k] + Beta[k,l+1]*P[:,(n_items*(l+1)+k)]
        else:
          U_exp[:,k] = U_exp[:,k] + Beta[0,l+1]*P[:,(n_items*(l+1)+k)]
    
    scale = 5 #dictates the level of noise in the choice probabilities. epsilon~Gumbel(0,1/scale)
    Y_prob = np.zeros((n,n_items))
    denom = sum([np.exp(scale*U_exp[:,k])*P[:,k] for k in range(n_items)])
    for k in range(n_items):
      Y_prob[:,k] = np.where(P[:,k] == 1, np.exp(scale*U_exp[:,k])/denom, 0)
    
    return Y_prob
    