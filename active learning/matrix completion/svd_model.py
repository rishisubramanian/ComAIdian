from data_loader import load_ratings_data, create_matrix
from model import power_method, get_prediction
import numpy as np

def mean_squared_error(actual_vector, predicted_vector):
  # compare actual vector and predicted vector
  # mean squared error
  mse = np.power(predicted_vector - actual_vector, 2)
  mse = mse.sum() / predicted_vector.size
  return mse

def max_squared_error(actual_vector, predicted_vector):
  # compare actual vector and predicted vector
  # mean squared error
  max_se = np.power(predicted_vector - actual_vector, 2)
  max_se = max_se.max()
  return max_se

def remove_training_data(matrix, user_id_to_remove=-1, joke_ids_to_remove=[]):
  if user_id_to_remove != -1:
    # for all of the items to remove, save them, and set to 0
    num_failed_jokes = 0
    for joke in joke_ids_to_remove:
      if(matrix[user_id_to_remove, joke] != 0):
        matrix[user_id_to_remove, joke] = 0
      else:
        print("Failed to remove joke", joke)
        num_failed_jokes += 1
    print("Failures: ", num_failed_jokes)

def train():
  # get data
  ratings_data = load_ratings_data()

  # create user x item matrix
  matrix = create_matrix(ratings_data)

  # get predicted matrix
  U, S, V = power_method(matrix, 100, k=5)


def test(leave_one_out_user_id=-1, leave_one_out_joke_ids=[]):
  # get data
  ratings_data = load_ratings_data()

  # create user x item matrix
  original_matrix = create_matrix(ratings_data)
  live_matrix = original_matrix.copy()

  remove_training_data(live_matrix,
                       user_id_to_remove=leave_one_out_user_id,
                       joke_ids_to_remove=leave_one_out_joke_ids)

  # decompose matrix
  U, S, V = power_method(live_matrix, number_of_iterations=150, k=5)

  # get predicted vector and actual vector
  actual_vector = original_matrix[leave_one_out_user_id].toarray()[0]
  predicted_vector = get_prediction(leave_one_out_user_id, U, S, V)

  mse = mean_squared_error(actual_vector, predicted_vector)
  max_se = max_squared_error(actual_vector, predicted_vector)
  return mse, max_se


if __name__ == '__main__':
  print("Testing...")
  """ because it's hard to hit jokes present in the db, this just generates 50 random entries
      hit rate is about 10%, so this is leave 5 out
      pretty hacky, but just for testing before real dataset
  """
  joke_ids = (350*np.random.randn(500) + 350).astype(int)
  mse, max_se = test(leave_one_out_user_id=5, leave_one_out_joke_ids=joke_ids)
  print("Mean Squared Error:", mse)
  print("Max Squared Error:", max_se)

