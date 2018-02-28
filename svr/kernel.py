#TODO
#1) Complete rbf_kernel function

class kernel:

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
    pass

  def poly_kernel(xTrain, x):
    return (1 + np.dot(xTrain, x))**self.poly_degree