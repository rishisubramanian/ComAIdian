import numpy as np
from numpy.linalg import inv

''' decomposes the matrix into u, s, v using the power method svd '''
def power_method(A_sub, number_of_iterations, k):
  """
  Implement the power method for computing the top singular value and the vector for a given CSR matrix (don’t use eigs or svds).
  Input of this function is a matrix, number of iterations, and rank

  Output of this function is the leading right singular vector (array) and """

  # For sparse matrix A_sub=sparse_matrix_in_csr_format
  global u, s, v

  n = A_sub.shape[1]
  np.random.seed(1)

  V0 = np.random.rand(n, k)

  # iterative method to get estimate of top right singular
  for i in range(number_of_iterations):
    V1 = A_sub.transpose().dot(A_sub.dot(V0))
    V2 = V1 / np.linalg.norm(V1)
    V0 = V2
  # top_right_singular_vector
  v = V2
  # singular value
  V1 = A_sub.transpose().dot(A_sub.dot(V2))
  s = V2.transpose().dot(V1)
  u = A_sub.dot(v.dot(inv(s)))
  return u, s, v

''' gets the prediction vector for a given userId '''
def get_prediction(userId, u, s, v):
  ''' recompose the predicted matrix '''
  PredictedMatrix = u[userId].dot(s.dot(v.T))
  return PredictedMatrix

