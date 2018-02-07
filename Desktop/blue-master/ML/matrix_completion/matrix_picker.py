import numpy as np

def closest_users(raters_frame, ratings_frame, user, ratings, k=50,
                  user_weight=1, ratings_weight=1):
    """
    This will return a list of the users that are "closest" to the
    features and ratings to the user passed in. "Closest" means smallest mean
    square distance.

    The distance between two users is
        d(u1, u2) = a * d(u1_features, u2_features)
                    + b * d(u1_ratings, u2_ratings)
    where a and b are the weights passed in.

    Args:
        raters_frame (pd.DataFrame): 1-hot encoded dataframe of the user
            features.
        ratings_frame (pd.DataFrame): The dataframe with the ratings for each
            joke by each user.
        user (np.array): A vector of the one-hot encoding of the user features.
        ratings (np.array): The vector of ratings for each joke by the user
            (possibly nan)
        k (int): The maximal size of the submatrix to take
        user_weight (float): The weight for the distance between users.
        ratings_weight (float): The weight for the distance between ratings
    """
    raters_matrix = raters_frame.values
    ratings_matrix = ratings_frame.values

    distances = []

    for i, (user2, ratings2) in enumerate(zip(raters_matrix, ratings_matrix)):
        distances.append((i, weighted_distance(user, user2, ratings, ratings2,
                                               user_weight, ratings_weight)))
    # We sort by the actual distances, not the indices
    sorted_distances = sorted(distances, key=lambda x: x[1])
    indices = [i for i, _ in sorted_distances[:k]]

    return raters_frame.index[indices]


def weighted_distance(user1, user2, ratings1, ratings2, user_weight=1,
                      ratings_weight=1):
    """
    Return the mean square distance between two users and their ratings.
    Args:
        user1 (np.array): A vector of the one-hot encoding of the user features.
        user2 (np.array): A vector of the one-hot encoding of the second user
            features
        rating1 (np.array): The vector of ratings for each joke by the user
            (possibly nan)
        rating2 (np.array): The vector of ratings for each joke by the second
            user
    """
    user_distance = np.nanmean(user1 - user2)
    ratings_distance = np.nanmean(ratings1 - ratings2)

    # We check for nan, as for new users there will be no ratings
    if np.isnan(ratings_distance):
        ratings_distance = 0

    return user_weight * user_distance + ratings_weight * ratings_distance

def kclosest(matrix, row, k):
    errors = (matrix - row) ** 2
    mses = np.nanmean(errors, axis=1)
    indices = np.argpartition(mses, k)[:k]
    return indices
