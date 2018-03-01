#TODO
# 1) Implement calculateEta() check formula (14) from http://cs229.stanford.edu/materials/smo.pdf
# 2) Finish rest of SMO implementation, might want to break it into functions
# 3) Finish score(), get R^2 score to determine how accurate the model is, check 
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR.score

import numpy as np

class SVR:
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
  def __init__(self, kernel = "linear", tol = 0.01, C = 1, epsilon = 0.1, poly_degree = 3, max_iter = 5000):
    self._kernel = kernel
    self._tol = tol
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
        if(checkKKT(self._yTrain[i], error, alpha_i)):
          j = np.random.randint(0, len(self._xTrain))

          while(j == i):
            j = np.random.randint(0, len(self._xTrain))

            #Save old ai  and aj, might need to take the _update_alpha_pair() from github into consideration (link below)
            #https://github.com/howardyclo/NTHU-Machine-Learning/blob/master/assignments/hw4-kernel-svr/svr.py
            old_alpha_i = alpha_i
            alpha_j = self._alphas[j]
            old_alpha_j = alpha_j

            #Compute bounds for lagrange multiplier
            L = calculateL(alpha_j, alpha_i)
            H = calculateH(alpha_j, alpha_i)

            if(L == H):
              continue

            #TODO
            #Calculate eta
            #If eta greater than or equal to 0, continue
            #Get new aj

      if(num_changed_alpha == 0):
        iteration += 1

      else:
        iteration = 0

  def checkKKT(y_i, error, alpha_i):
    """
    """

    #Return True if KKT condition violated
    if(y_i * error < self._tol and alpha_i < self._C):
      return True

    if(y_i * error > self._tol and alpha_i > 0):
      return True

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
    """
    """

    kernel = Kernel(self._kernel, self._poly_degree)
    kernelX = kernel.calculateKernel(self._xTrain, x)

    #When predicting on one x input
    if(x.shape[0] == 1):
      if(self.b_mean):
        return np.dot(self._alphas, kernelX) + self.b_mean

      else:
        return np.dot(self._alphas, kernelX) + self.b

  def calculateError(self, i):
    """
    """

    predictedY = self.predict(self.xTrain[i])

    return predictedY - self.yTrain[i]

  #TODO
  def score(self, xTest, yTest):
    pass


