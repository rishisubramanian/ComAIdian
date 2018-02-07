"""
metrics.py

This module will provide some basic metrics for evaluating the performance
of the separate methods for filling in the matrices.
"""

import random
import numpy as np

from sklearn.model_selection import LeaveOneOut

import matrix_complete
from data_processing import MultiLabelBinarizer


def mean_square_error(actuals, predicteds):
    """
    This will compute the mean square error of the data.
    This is defined as
        MSE = ((x1 - y1)^2 + ... + (xn - yn)^2)/n

    Args:
        actuals (list[float]): The list of actual observed values
        predicteds (list[float]): The list of predicted values
    Returns:
        float: The mean square error
    """
    xs = np.array(actuals)
    ys = np.array(predicteds)
    return np.nanmean((xs - ys) ** 2)


class MatrixTester:
    """
    This class will encapsulate all the information required for testing the
    matrix completion algorithms.

    This runs a
    """
    def __init__(self, ratings, raters, sampler='simple', completer='mean',
                 picker='simple', sample_size=5):
        self.ratings = ratings.copy()
        rater_indices = ratings.index
        self.raters = raters.copy().loc[rater_indices]
        self.sampler = sampler
        self.picker = picker
        self.completer = completer
        self.sample_size = sample_size

    def sample(self, row_index):
        if self.sampler == 'simple':
            while True:
                col_index = random.choice(list(self.test_set.columns))
                if col_index not in self.already_sampled:
                    self.already_sampled.add(col_index)
                    break
            return col_index, self.test_set[col_index][row_index]
        elif callable(self.sampler):
            col_index = self.sampler()
            return col_index, self.test_set[col_index]
        else:
            raise NotImplementedError("No other sampling methods implemented")

    def pick(self):
        """
        This picks out the specific submatrix to complete.

        Returns:
            list: The list of row indices of the submatrix
        """
        if self.picker == 'simple':
            return self.train_set.index
        elif self.picker == 'KNN':
            pass

    def complete(self, row):
        """
        This will pick out a submatrix and call a matrix completion algorithm
        on the submatrix with the row passed in.
        """
        user_indices = self.pick()
        matrix = self.ratings.loc[user_indices]
        stacked_mat = np.vstack([matrix, row])
        return matrix_complete.complete_matrix(stacked_mat, self.completer)

    def complete_row(self, row_index):
        """
        This function will take out the specific row index from the ratings
        matrix. It will then clear out the entries and refill the sample
        of entries. Then it will call the matrix completion algorithm
        with the incomplete row and return the completed row.
        """
        row = self.test_set.copy()
        # Clear out the entries
        row *= np.nan
        for _ in range(self.sample_size):
            col_index, sample = self.sample(row_index)
            row[col_index][row_index] = sample
        new_matrix = self.complete(row.values)
        return new_matrix[-1]

    def run(self):
        """
        This will run the testing algorithm and produce error metrics after the
        test. The specific test performs cross-validation using the
        leave-one-out scheme. Specifically, this will take one row of the data
        and clear it and then sample five points to refill. Then it will run
        the matrix completion algorithm on the mostly incomplete row. Then it
        will produce the mean squared error between the predicted ratings and
        the actual ratings.

        It does this for each row in the matrix and returns the errors for
        every single row.

        Returns:
            list[float]: The mean square error for every row
        """
        errors = []
        for train_index, test_index in LeaveOneOut().split(self.ratings):
            self.already_sampled = set()
            train_index = self.ratings.index[train_index]
            test_index = self.ratings.index[test_index]
            self.train_set = self.ratings.loc[train_index]
            self.test_set = self.ratings.loc[test_index]

            actual_row = self.test_set.values
            predicted_row = self.complete_row(test_index)
            errors.append(mean_square_error(actual_row, predicted_row))
        return errors


class KNNTester(MatrixTester):
    """
    This will enable the same matrix tests as for MatrixTester, yet
    specifically enables the comparison of KNN algorithms. This allows for the
    configuration of how many neighbors are actually used in the KNN algorithm,
    i.e., what K is.
    """
    def __init__(self, ratings, raters, k, sampler='simple', sample_size=5):
        """
        Args:
            k (int): The number of nearest neighbors used in the K nearest
                neighbors algorithm.
        """
        self.k = k
        MatrixTester.__init__(self, ratings, raters, completer='KNN',
                              sample_size=sample_size)

    def complete(self, row):
        """
        This mimics the parent complete algorithm, but specifies k in the
        KNN algorithm.
        """
        user_indices = self.pick()
        matrix = self.ratings.loc[user_indices]
        mat = np.vstack([matrix, row])
        return matrix_complete.complete_matrix(mat, self.completer, k=self.k)


class IncrementalTester(MatrixTester):
    def __init__(self, ratings, raters, sampler='simple', completer='mean',
                 picker='simple', sample_size=5):
        MatrixTester.__init__(self, ratings, raters, sampler, completer,
                              picker, sample_size)

    def update_row(self, row, row_index):
        col, value = self.sample(row_index)
        row[col][row_index] = value
        return row

    def fit(self, row_index=None):
        errors = []
        self.already_sampled = set()
        if row_index is None:
            row_index = random.choice(list(self.ratings.index))
        self.row_index = row_index

        training_indices = [index for index in self.ratings.index if index != row_index]
        self.train_set = self.ratings.loc[training_indices]
        self.test_set = self.ratings.loc[[row_index]]

    def run(self, n=20):
        row = self.test_set * np.nan
        errors = []
        for _ in range(n):

            matrix = np.clip(self.complete(row.values), 1, 5)
            predicted_row = matrix[-1]
            actual_row = self.test_set.values
            row = self.update_row(row, self.row_index)
            errors.append(mean_square_error(actual_row, predicted_row))
        return errors


def test_method(matrix, method):
    """
    This method will select a random row and delete all but 5 entries from it.
    It will use then complete the matrix with this incomplete row.
    It will then compute the mean square error between the actual row and the
    predicted row.

    Args:
        matrix (np.array): The matrix with possibly incomplete values
        method (str): The method for matrix completion
    Returns:
        float: The mean squared error of the row
    """
    m, n = matrix.shape
    row_index = random.choice(list(range(m)))

    matrix_copy = matrix.copy()

    actual_row = matrix[row_index]
    row_copy = actual_row.copy()

    for index in random.sample(list(range(n)), n - 5):
        row_copy[index] = np.nan
    matrix_copy[row_index] = row_copy
    complete_matrix = matrix_complete.complete_matrix(matrix_copy, method)
    predicted_row = complete_matrix[row_index]

    return mean_square_error(actual_row, predicted_row)
