#TODO
# 1) Implement checkKKT()
# 2) Implement calculateEta() check formula (14) from http://cs229.stanford.edu/materials/smo.pdf
# 3) Finish rest of SMO implementation

import numpy as np

class svr:
  """SVR class use to train and predict dataset

    Args:
      kernel (str): Kernel type to apply on data, default to  'linear'; 'linear', 'poly', 'rbf' are all acceptables
      C (float): Penalty parameter
      epsilon (float): 
      poly_degree (int): Polynomial degree, use only with polynomial kernel calculation
      max_iter (int): Max iteration to perform during training

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
  def __init__(kernel = "linear", C = 1, epsilon = 0.1, poly_degree = 3, max_iter = 5000):
    self._kernel = kernel
    self._C = C
    self._epsilon = epsilon
    self._poly_degree = poly_degree
    self._max_iter = max_iter

  def fit(self, X, Y):
    """Train given data using simple SMO

    Args:
      X (np.array): X training data
      Y (np.array): Y training data

    return: svr_predictor

    """
    #SMO algorithm from http://cs229.stanford.edu/materials/smo.pdf
    
    self._xTrain = X
    self._yTrain = Y

    #Alpha use as lagrange multipliers
    alphas = np.zeros(len(X))
    self._alphas = alphas

    #Threshold for solution
    b = 0
    iteration = 0

    while(iteration < self.max_iter):
      num_changed_alpha = 0

      for i in range(X.shape[0]):
        #Calculate error
        error = calculateError(i)
        alpha_i = self._alphas[i]

        #Check KKT condition
        if(checkKKT(error, alpha_i)):
          j = np.random.randint(0, len(self._xTrain))

          while(j == i):
            j = np.random.randint(0, len(self._xTrain))

            #TODO

      if(num_changed_alpha == 0):
        iteration += 1

      else:
        iteration = 0

  #TODO
  def checkKKT(error, alpha_i):
    #Return True if KKT condition violated
    
    #Else return False
    
    return False

  def calculateL(alpha_j, alpha_i):
    return max(0, alpha_j - alpha_i)

  def calculateH(self, alpha_j, alpha_i):
    return min(self._C, self._C + alpha_j - alpha_i)

  #TODO
  def calculateEta():
    pass

  def predict(self, x):
    kernel = Kernel(self._kernel, self._poly_degree)
    kernelX = kernel.calculateKernel(self._xTrain, x)

    #When predicting on one x input
    if(x.shape[0] == 1):
      if(self.b_mean):
        return np.dot(self._alphas, kernelX) + self.b_mean

      else:
        return np.dot(self._alphas, kernelX) + self.b

  def calculateError(self, i):
    predictedY = self.predict(self.xTrain[i])

    return predictedY - self.yTrain[i]

