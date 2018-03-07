import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

""" directory containing ratings file """
DATASET_DIRECTORY = "dataset"

""" name of ratings file """
RATINGS_FILENAME = "ratings.csv"

"""" Enables verbose logging, for debugging """
_DEBUG = True

""" get dataframe from csv """
def load_ratings_data():
  ratings_df = pd.read_csv(DATASET_DIRECTORY + "/" + RATINGS_FILENAME)
  if(_DEBUG):
    print("Dataframe loaded!")
  return ratings_df

""" in-place row normalizes a given matrix """
def row_normalize(matrix):
  matrix_sums = matrix.sum(axis=1)
  matrix = matrix / matrix_sums[:, np.newaxis]

""" creates user, movie matrix from ratings dataframe """
def create_matrix(ratings_df):
  #if(_DEBUG):
  #  print("Number of Users:", num_users)
  #  print("Number of Movies:", num_items)

  """ create user x movie matrix and populate with ratings """
  """ note: empty entries are populated with zeros """

  rows = np.array(ratings_df['userId'].values)
  cols = np.array(ratings_df['movieId'].values)
  values = np.array(ratings_df['rating'].values)

  matrix = sparse.csc_matrix((values, (rows, cols)))
  matrix = normalize(matrix, norm='l1', axis=1)

  return matrix
