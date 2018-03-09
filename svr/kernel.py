import numpy as np
from numpy import linalg as LA

class Kernel:

  def __init__(self, kernel, poly_degree = 3):
    self._kernel = kernel
    self._poly_degree = poly_degree

  def calculateKernel(self, xTrain, x):
    if(self.kernel == "linear"):
      return linear_kernel(xTrain, x)

    elif(self.kernel == "rbf"):
      return rbf_kernel(xTrain, x)

    elif(self.kernel == "poly"):
      return poly_kernel(xTrain, x)

  def linear_kernel(xTrain, x):
    return np.dot(xTrain, x)

  def rbf_kernel(xTrain, x):
    if(xTrain.ndim == 1):
      sqrt_norm = LA.norm(X - x)**2
    # Multiple examples
    elif(xTrain.ndim == 2):
      sqrt_norm = LA.norm(X - x, axis=1)**2

    return np.exp(-sqrt_norm / (2.0 * (self.rbf_sigma**2)))

  def poly_kernel(xTrain, x):
    return (1 + np.dot(xTrain, x))**self.poly_degree