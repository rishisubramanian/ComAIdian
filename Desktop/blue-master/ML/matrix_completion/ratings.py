"""
ratings.py

This module will export a class RatingsDatabase which will have convenience
functions for accessing information about the users, ratings, and jokes.
"""

import numpy as np
import pandas as pd

from user import user_distance

class RatingsDatabase:
    """
    This class is to make accessible all of the joke, rating, and rater
    information from the dataset. It is instantiated from the three dataframes
    pulled from the joke database file.
    These are the raters, ratings, and jokes dataframes.

    The raters dataframe is indexed by user id and has columns for each
    user feature. The jokes dataframes is indexed by joke id and has columns
    for each joke feature. The ratings dataframe is indexed by user id, and
    has columns for each joke id. It is filled with each of the ratings.

    Attributes:
        raters (pd.DataFrame): All information about rater features
        ratings (pd.DataFrame): All ratings for each joke and user
        jokes (pd.DataFrame): All information abut joke features
    """
    def __init__(self, raters, ratings, jokes):
        self.ratings = ratings
        self.raters = raters.loc[ratings.index]
        self.jokes = jokes

    def get_user_by_id(self, uid):
        """
        This will return a pandas series of all the features for the specified
        user. The features are in listed in order, 'gender', 'age',
        'birth_country', 'preferred_joke_genre', 'preferred_joke_genre2',
        'preferred_joke_type'.

        Args:
            uid (int): The id for a specific user
        Returns:
            pd.Series: The features for a specific user
        """
        if uid not in self.raters.index:
            raise IndexError("No user exists with the id {}".format(uid))

        return self.raters.loc[uid]

    def get_joke_by_id(self, jid):
        """
        This will return a pandas series of all the features for a specific
        joke. The features are listed in order 'category', 'joke_type',
        'subject', 'joke_text', 'joke_submitter_id', 'joke_source'.

        Args:
            jid (int): The id for a specific joke
        Returns:
            pd.Series: The features for a specific joke
        """
        if jid not in self.jokes.index:
            raise IndexError("No joke exists with the id {}".format(jid))

        return self.jokes.loc[jid]

    def get_joke_text_by_id(self, jid):
        """
        This will get the joke text for a specified joke.

        Args:
            jid (int): The id for a specific joke
        Returns:
            str: The text of a joke
        """
        return self.get_joke_by_id(jid)['joke_text']

    def distance_between_users(self, uid1, uid2):
        """
        This will return the distance between users which is defined as the
        sum of the distances between the individual features. For numeric
        features, the distance is simply the natural metric on the real
        numbers and for categorical features, this is the discrete metric.

        Args:
            uid1 (int): The id for the first user
            uid2 (int): The id for the second user
        Returns:
            float: The distance between the users
        """
        return user_distance(self.raters, uid1, uid2)

    def k_closest_users(self, uid, k=50):
        """
        This function will return the k closest users to the user specified
        by uid using the metric specified in distance_between_users. This will
        not include the given user.

        Args:
            uid (int): The user's id
            k (int): The number of nearest neighbors
        """
        # We sort all the indices by the distance to the specified user.
        closest = sorted(self.raters.index,
                         key = lambda x:self.distance_between_users(uid, x))

        return [x for x in closest if x != uid][:k]

    def get_user_ratings(self, uid):
        """
        This will return the series of ratings for each joke indexed by joke
        id for the specified user.

        Args:
            uid (int): The user id you wish to retrieve ratings for
        Returns:
            pd.Series: The series of ratings indexed by joke id
        """
        return self.ratings.loc[uid]

    def get_joke_ratings(self, jid):
        """
        This will return the series of ratings for a specific joke by each
        user. This will be indexed by user ratings.

        Args:
            jid (int): The index of a specific joke
        Returns:
            pd.Series: The series of ratings indexed by user id
        """
        return self.ratings[jid]

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
        new_id = max(self.raters.index) + 1

        user_spec = {'gender': gender,
                     'age': age,
                     'birth_country': birth_country,
                     'major': major,
                     'preferred_joke_genre': preferred_joke_genre,
                     'preferred_joke_genre2': preferred_joke_genre2,
                     'preferred_joke_type': preferred_joke_type,
                     'favorite_music_genre': favorite_music_genre,
                     'favorite_movie_genre': favorite_movie_genre}
        self.raters.loc[new_id] = pd.Series(user_spec)
        # The new ratings are all NaN as we have no ratings for this user yet.
        self.ratings.loc[new_id] = np.nan

        return new_id

    def get_rating(self, uid, jid):
        """
        Get the rating for a user for a specific joke.

        Args:
            uid (int): The user id for the rater
            jid (int): The joke id for the rated joke
        Returns:
            int: The rating of the joke by the user
        """
        if uid not in self.raters.index:
            raise IndexError("No user exists with the id {}".format(uid))

        if jid not in self.jokes.index:
            raise IndexError("No joke exists with the id {}".format(jid))

        return self.ratings[jid][uid]

    def set_rating(self, uid, jid, rating):
        """
        Set the rating for a user for a specific joke.

        Args:
            uid (int): The user id for the rater
            jid (int): The joke id for the rated joke
            rating (int): An integer in range 1 to 5 rating the joke
        """
        if uid not in self.raters.index:
            raise IndexError("No user exists with the id {}".format(uid))

        if jid not in self.jokes.index:
            raise IndexError("No joke exists with the id {}".format(jid))

        if not (1 <= rating <= 5):
            raise ValueError("The rating must be an integer between 1 and 5")

        self.ratings[jid][uid] = rating
