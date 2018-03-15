from data_loader import load_ratings_data, create_matrix
import numpy as np
from numpy.linalg import inv

class SVD_Model:
  def __init__(self):
    # contain the decomposed matrix
    self.u = []
    self.s = []
    self.v = []

    # svd attributes
    self.k = 10
    self.number_of_iterations = 10


  ''' decomposes the matrix into u, s, v using the power method svd '''
  def power_method(self, A_sub):
    """
    Implement the power method for computing the top singular value and the vector for a given CSR matrix (dont use eigs or svds)
    Input of this function is a matrix, number of iterations, and rank

    Output of this function is the leading right singular vector (array) and """

    # For sparse matrix A_sub=sparse_matrix_in_csr_format

    n = A_sub.shape[1]
    np.random.seed(1)

    V0 = np.random.rand(n, self.k)

    # iterative method to get estimate of top right singular
    for i in range(self.number_of_iterations):
      V1 = A_sub.transpose().dot(A_sub.dot(V0))
      V2 = V1 / np.linalg.norm(V1)
      V0 = V2
    # top_right_singular_vector
    self.v = V2
    # singular value
    V1 = A_sub.transpose().dot(A_sub.dot(V2))
    self.s = V2.transpose().dot(V1)
    self.u = A_sub.dot(self.v.dot(inv(self.s)))

  ''' gets the prediction vector for a given userId '''
  def get_prediction(self, userId):
    ''' recompose the predicted matrix '''
    PredictedMatrix = self.u[userId].dot(self.s.dot(self.v.T))
    return PredictedMatrix

  def mean_squared_error(self, actual_vector, predicted_vector):
    # compare actual vector and predicted vector
    # mean squared error
    mse = np.power(predicted_vector - actual_vector, 2)
    mse = mse.sum() / predicted_vector.size
    return mse

  def max_squared_error(self, actual_vector, predicted_vector):
    # compare actual vector and predicted vector
    # mean squared error
    max_se = np.power(predicted_vector - actual_vector, 2)
    max_se = max_se.max()
    return max_se

  def remove_training_data(self, matrix, user_id_to_remove=-1, joke_ids_to_remove=[]):
    if user_id_to_remove != -1:
      # for all of the items to remove, save them, and set to 0
      num_failed_jokes = 0
      for joke in joke_ids_to_remove:
        if (matrix[user_id_to_remove, joke] != 0):
          matrix[user_id_to_remove, joke] = 0
        else:
          num_failed_jokes += 1
      print("Number of Jokes Removed: ", len(joke_ids_to_remove) - num_failed_jokes)

  def train(self):
    # get data
    ratings_data = load_ratings_data()

    # create user x item matrix
    matrix = create_matrix(ratings_data)

    # get predicted matrix
    self.power_method(matrix)

  def test(self, leave_one_out_user_id=-1, leave_one_out_joke_ids=[]):
    # get data
    ratings_data = load_ratings_data()

    # create user x item matrix
    original_matrix = create_matrix(ratings_data)
    live_matrix = original_matrix.copy()

    self.remove_training_data(live_matrix,
                         user_id_to_remove=leave_one_out_user_id,
                         joke_ids_to_remove=leave_one_out_joke_ids)

    # decompose matrix
    self.power_method(live_matrix)

    # get predicted vector and actual vector
    actual_vector = original_matrix[leave_one_out_user_id].toarray()[0]
    predicted_vector = self.get_prediction(leave_one_out_user_id)

    mse = self.mean_squared_error(actual_vector, predicted_vector)
    max_se = self.max_squared_error(actual_vector, predicted_vector)
    return mse, max_se

