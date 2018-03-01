from data_loader import load_ratings_data, create_matrix
from model import power_method, get_prediction
import math

def remove_training_data(matrix, user_ids_to_remove=[], joke_ids_to_remove=[]):
  # for all of the items to remove, save them, and set to 0
  for user in user_ids_to_remove:
    for joke in joke_ids_to_remove:
        matrix[user, joke] = 0


def train():
  # get data
  ratings_data = load_ratings_data()

  # create user x item matrix
  matrix = create_matrix(ratings_data)

  # get predicted matrix
  U, S, V = power_method(matrix, 100, k=5)


def test(leave_one_out_user_id=[], leave_one_out_joke_ids=[]):
  # get data
  ratings_data = load_ratings_data()

  # create user x item matrix
  original_matrix = create_matrix(ratings_data)
  live_matrix = original_matrix.copy()

  remove_training_data(live_matrix,
                       user_ids_to_remove=leave_one_out_user_id,
                       joke_ids_to_remove=leave_one_out_joke_ids)

  # decompose matrix
  U, S, V = power_method(live_matrix, number_of_iterations=150, k=5)

  # get predicted vector and actual vector
  actual_vector = original_matrix[leave_one_out_user_id[0]].toarray()[0]
  predicted_vector = get_prediction(leave_one_out_user_id[0], U, S, V)

  # compare actual vector and predicted vector
  # mean squared error
  mse = 0
  number_of_rated_items = 0
  for joke in leave_one_out_joke_ids:
    if(actual_vector[joke] != 0):
      mse += math.pow((predicted_vector[joke] - actual_vector[joke]), 2)
      number_of_rated_items += 1
  if(number_of_rated_items > 0):
    mse /= number_of_rated_items
  else:
    raise ValueError('No items in the testing set were rated yet.')

  return mse


if __name__ == '__main__':
  print("Testing...")
  mse = test([5], [100, 200, 300, 400, 500, 600, 700])
  print("Mean Squared Error:", mse)


