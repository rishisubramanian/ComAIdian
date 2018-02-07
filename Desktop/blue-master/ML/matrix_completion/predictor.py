"""
predictor.py

This module will export a Predictor class for the purposes of recommending
jokes using the matrix completion algorithm.
"""

import random

import numpy as np
import pandas as pd

from forms import IntegerField
from matrix_complete import complete_matrix
from data_processing import MultiLabelBinarizer
from matrix_picker import closest_users
from ratings import RatingsDatabase

class Predictor(RatingsDatabase):
    """
    This class will house a Matrix Completion algorithm which is used to
    predict joke ratings and use these predicted joke ratings to suggest
    jokes to a user. For this model to predict jokes, it must be fit to a
    joke raters, ratings, and jokes in a specific format (see the fit function)

    Then a user in particular can be specified for whom jokes will be
    suggested. This will prompt the user for ratings and these ratings can then
    be readded to the model to improve its performance

    This subclasses from RatingsDatabase in order to use the methods for adding
    users and setting and getting ratings.

    Example usage:
    >>> import user
    >>> from predictor import Predictor
    >>> raters, ratings, jokes = user.read_clean_data()
    >>> predictor = Predictor(raters, ratings, jokes, completer='KNN')
    >>> uid = predictor.add_user(age=12, gender='Male')
    >>> predictor.set_user(uid)
    >>> jid = predictor.get_joke()
    >>> joke_text = predictor.get_joke_text_by_id(jid)
    >>> predictor.set_rating(uid, jid, 2)
    """
    def __init__(self, raters, ratings, jokes, sampler='simple',
                 picker='simple', completer='mean'):
        """
        Args:
            raters (pd.DataFrame): All information about rater features
            ratings (pd.DataFrame): All ratings for each joke and user
            jokes (pd.DataFrame): All information abut joke features
            sampler (str): Either 'simple' indicating random choice of jokes
                or 'best' indicating joke with highest predicted rating is
                picked.
            picker (str): Either 'simple' indicating the submatrix to complete
                is the entire matrix, or 'KNN' meaning the matrix with rows
                most similar to the current row are taken. Alternatively, a
                function which accepts raters, ratings, jokes, and id of
                current user and returns a list of the uids corresponding to
                the rows of the chosen submatrix.
            completer (str): See matrix_complete.complete_matrix for a full
                specification of all posibility parameters. Alternatively, a
                function which accepts a matrix and returns a completed
                matrix.
        """
        self.sampler = sampler
        self.picker = picker
        self.completer = completer
        self.current_user = None
        self.predicted_ratings = ratings.copy()
        RatingsDatabase.__init__(self, raters, ratings, jokes)

    def set_user(self, uid):
        """
        Set the current user for whom jokes will be given.

        Args:
            uid (int): The user id to recommend jokes to
        """
        if uid not in self.raters.index:
            raise IndexError("No user with id {} exists".format(uid))

        self.current_user = uid

    def sample(self, n=1):
        """
        This method will sample n jokes depending on the method
        specified. If the method is 'simple', it will just pick n at random.
        If the method is 'best', then it will pick the joke with predicted
        best rating. This method samples without replacement.

        This may return fewer jokes than desired if there are not enough
        unsampled jokes.

        Args:
            n (int): The number of jokes to sample
        Returns:
            list[int]: The index of the joke
        """
        # We first find the jokes which haven't been rated by the current user.
        # These are the indices which are NaN.
        row = self.ratings.loc[self.current_user]
        unsampled_jids = row[np.isnan(row)].index.tolist()

        # If there are no unsampled jokes left, we will return the empty list.
        if len(unsampled_jids) <= n:
            return unsampled_jids

        if self.sampler == 'simple':
            return random.sample(unsampled_jids, n)
        elif self.sampler == 'best':
            # Get the predicted ratings for the user in question
            predictions = {jid: self.predicted_ratings[jid][self.current_user]
                           for jid in unsampled_jids}
            # Sort by predicted value in descending value
            best_jids = sorted(predictions.keys(),
                               key=lambda x: predictions[x], reverse=True)[:n]
            return best_jids
        else:
            raise NotImplementedError("No other sampling methods implemented")


    def pick(self):
        """
        This method will pick the submatrix of the method which is actually
        used for matrix completion. In the simple case, this will just be
        the entire matrix. If the method is 'KNN', it will pick the k 'closest'
        users in terms of mean square difference (with a 1-hot encoding for
        categorical features.

        This will return a list of user indices.

        Returns:
            index: The index of a dataframe
        """
        if self.current_user is None:
            raise RuntimeError("A submatrix cannot be picked if the user is "
                                "not set with set_user")

        if self.picker == 'simple':
            return self.ratings.index.tolist()
        elif self.picker == 'KNN':
            # Find the k closest users and add the current user
            uids = self.k_closest_users(self.current_user)
            return uids + [self.current_user]
        elif callable(self.picker):
            return self.picker(self.raters, self.ratings, self.jokes,
                               self.current_user)
        else:
            raise NotImplementedError("No other mathods implemented for "
                                      "picking the submatrix to complete.")

    def complete(self):
        """
        This method will complete the matrix and return the ratings for the
        current user. It will use the method specified for completing the
        matrix and then updated the predicted ratings frame.
        """
        user_indices = self.pick()

        # We need the current user to be in the chosen submatrix
        if self.current_user not in user_indices and self.current_user is not None:
            user_indices += [self.current_user]

        matrix = self.ratings.loc[user_indices].values
        # If using a custom matrix completer
        if callable(self.completer):
            new_matrix = self.completer(matrix)
        else:
            new_matrix = complete_matrix(matrix, self.completer)

        # We then update the predicted ratings
        for i, uid in enumerate(user_indices):
            self.predicted_ratings.loc[uid] = new_matrix[i]

    def get_joke(self, n=1):
        """
        This method will return a joke id by sampling as is specified by the
        sampler method.

        Args:
            n (int): The number of jokes to get
        Returns:
            int: The index of the joke
        """
        if self.current_user is None:
            raise RuntimeError("No user is set, so no joke can be recommended")
        return self.sample(n)

    def add_user(self, gender=np.nan, age=np.nan, birth_country=np.nan,
                 major=np.nan, preferred_joke_genre=np.nan,
                 preferred_joke_genre2=np.nan, preferred_joke_type=np.nan,
                 favorite_music_genre=np.nan, favorite_movie_genre=np.nan):
        """
        This will add a new user to the database with the specified user
        features. No features actually need to be specified, and if they are
        not the spots will be filled with NaNs in the user table. The new user
        id will be one larger than the highest uid currently in the table and
        it will be returned.

        This will also add a row of NaNs to the ratings table for the new user.

        This method does not do feature validation.

        Args:
            gender (str): One of "Male", "Female", "Prefer not to say"
            age (int): The age of the user
            birth_country (str): The country where the user was born
            major (str): The user's major
            preferred_joke_genre (str): One of the several joke genres specifed
                in the data spec
            preferred_joke_genre2 (str): The user's second joke genre
            preferred_joke_type (str); The preferred joke type
            favorite_music_genre (str): The preferred music genre
            favorite_movie_genre (str): The preferred movie genre
        Returns:
            int: The new user's id
        """
        uid = RatingsDatabase.add_user(
            self, gender, age, birth_country, major, preferred_joke_genre,
            preferred_joke_genre2, preferred_joke_type,
            favorite_music_genre, favorite_movie_genre)
        # The predicted ratings for the user are initially all nan
        self.predicted_ratings.loc[uid] = np.nan
        return uid

    def update(self):
        """
        This will just call the matrix completion algorithm which will update
        the predicted ratings based on the current ratings.
        """
        self.complete()

    def set_rating(self, uid, jid, rating):
        """
        Set the rating for a user for a specific joke.

        Args:
            uid (int): The user id for the rater
            jid (int): The joke id for the rated joke
            rating (int): An integer in range 1 to 5 rating the joke
        """
        RatingsDatabase.set_rating(self, uid, jid, rating)
        # We need to update the predicted frame as well
        self.predicted_ratings[jid][uid] = rating
        self.update()

    def get_predicted_rating(self, uid, jid):
        """
        Get the predicted rating for a specific joke. If this is a joke, we
        already have a rating for, this will simply be that rating and no
        prediction is actually made.

        Args:
            uid (int): The user id for the rater
            jid (int): The joke id for the rated joke
        Returns:
            int: The predicted rating of the joke by the user
        """
        if uid not in self.raters.index:
            raise IndexError("No user exists with the id {}".format(uid))

        if jid not in self.jokes.index:
            raise IndexError("No joke exists with the id {}".format(jid))

        return self.predicted_ratings[jid][uid]
