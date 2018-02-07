"""
Some utilities related to matrix functions
"""

import random
import itertools

import numpy as np
import scipy.stats

def generate_matrix(m, n, distr=scipy.stats.norm()):
    """
    This will generate a mxn matrix filled with random values from a specified
    distribution, by default this is a standard normal.
    A user can pass in alternative scipy distributions to get matrices sampled
    from different distributions.

    Args:
        m (int): The number of rows
        n (int): The number of columns
        distr (scipy.stats.distribution): An instantiated 'frozen' distribution
            object to generate samples from
    Returns:
        np.array: An mxn matrix of samples from the distribution
    """
    return distr.rvs((m,n))

def drop_entries(matrix, n):
    """
    This will replace n entries in a passed in matrix with nan.
    This will produce a new matrix with the removed entries.

    Args:
        matrix (np.array): A matrix to drop entries from
        n (int): The number of entries to remove
    Returns:
        matrix
    """
    (x1, x2) = matrix.shape
    matrix_copy = matrix.copy()

    index_pairs = list(itertools.product(range(x1), range(x2)))

    for (x, y) in random.sample(index_pairs, n):
        matrix_copy[x][y] = np.nan

    return matrix_copy
