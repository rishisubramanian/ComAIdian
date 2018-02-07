"""
user.py

This should include basic information about users
"""

import sqlite3 as sql
import psycopg2
import pandas as pd

from forms import IntegerField, CategoricalField

class Feature:
    """
    Simple container for a feature which stores information about the name
    and the kind (i.e., numeric or categorical).

    Attributes:
        name (str): The name of the feature
        kind (str): The kind of data stored ('numeric' or 'categorical')
    """
    def __init__(self, name, kind):
        """
        Args:
            name (str): The name of the feature
            kind (str): The kind of data stored ('numeric' or 'categorical')
        """
        self.name = name
        self.kind = kind

user_features = [
    Feature('gender', 'categorical'),
    Feature('age', 'numeric'),
    Feature('birth_country', 'categorical'),
    Feature('major', 'categorical'),
    Feature('preferred_joke_genre', 'categorical'),
    Feature('preferred_joke_genre2', 'categorical'),
    Feature('preferred_joke_type', 'categorical'),
    Feature('favorite_music_genre', 'categorical'),
    Feature('favorite_movie_genre', 'categorical')
]

def old_read_clean_data(nameOfDataBase):
    """
    This function will pull from the sqlite database 'jokedb.sqlite3' data
    related to the joke raters, joke ratings, and jokes themselves.
    It will minimally format the data, removing duplicate entries.

    It will return three DataFrames the first containing information
    regarding the joke raters, the second containing information about the
    joke ratings, and the third containing information about the jokes.

    joke_rater has columns:
    gender, age, birth_country, major, preferred_joke_genre,
    preferred_joke_genre2, preferred_joke_type, favorite_music_genre,
    favorite_movie_genre

    joke_ratings is a dataframe with columns indexing each joke and the row
    index being the index of the user rating the joke.

    jokes has columns:
    category, joke_type, subject, joke_text, joke_submitter_id, joke_source

    Returns:
        tuple(DataFrame, DataFrame, DataFrame): Three dataframes in order,
            joke raters, joke ratings, jokes
    """
    jokedb = sql.connect(nameOfDataBase)
    joke_rater = pd.read_sql_query("SELECT * from JokeRater", jokedb)
    joke = pd.read_sql_query("SELECT * from Joke", jokedb)
    joke_rating = pd.read_sql_query("SELECT * from JokeRating", jokedb)

    joke_rater = joke_rater.drop(['joke_submitter_id'], axis=1).set_index('id')

    joke = joke.set_index('id')
    joke_rating = joke_rating.drop_duplicates(['joke_id', 'joke_rater_id'], keep='first')
    pivot = joke_rating.pivot(index='joke_id', columns='joke_rater_id', values='rating')
    joke_rating = pivot.transpose()

    return joke_rater, joke_rating, joke

def read_clean_data(nameOfDataBase, user, host, password, local):
    """
    This function will pull from the sqlite database 'jokedb.sqlite3' data
    related to the joke raters, joke ratings, and jokes themselves.
    It will minimally format the data, removing duplicate entries.

    It will return three DataFrames the first containing information
    regarding the joke raters, the second containing information about the
    joke ratings, and the third containing information about the jokes.

    joke_rater has columns:
    gender, age, birth_country, major, preferred_joke_genre,
    preferred_joke_genre2, preferred_joke_type, favorite_music_genre,
    favorite_movie_genre

    joke_ratings is a dataframe with columns indexing each joke and the row
    index being the index of the user rating the joke.

    jokes has columns:
    category, joke_type, subject, joke_text, joke_submitter_id, joke_source

    Args:
        nameOfDataBase (str): The name of the database
        user (str): The username
        host (str): The host database
        password (str): The password

    Returns:
        tuple(DataFrame, DataFrame, DataFrame): Three dataframes in order,
            joke raters, joke ratings, jokes
    """

    try:
        if local=='true':
            connection = psycopg2.connect("dbname={} user={}"
                .format(nameOfDataBase, user))
        else:
            connection = psycopg2.connect("dbname={} user={} host={} password={}"
                .format(nameOfDataBase, user, host, password))
    except:
        print("I am unable to connect to the database")

    joke_rater = pd.read_sql_query("SELECT * from \"JokeRater\"", connection)
    joke = pd.read_sql_query("SELECT * from \"Joke\"", connection)
    joke_rating = pd.read_sql_query("SELECT * from \"JokeRating\"", connection)
    connection.close()

    joke_rater = joke_rater.drop(['joke_submitter_id'], axis=1).set_index('id')

    joke = joke.set_index('id')
    joke_rating = joke_rating.drop_duplicates(['joke_id', 'joke_rater_id'],
                                              keep='first')
    pivot = joke_rating.pivot(index='joke_id', columns='joke_rater_id',
                              values='rating')
    joke_rating = pivot.transpose()

    return joke_rater, joke_rating, joke


def read_user(joke_raters):
    """
    This function will prompt the user for input about all the user
    parameterizations.
    """

    gender_field = CategoricalField('gender', joke_raters['gender'].unique())
    age_field = IntegerField('age', 0, 120)
    birth_country_field = CategoricalField(
        'birth country', joke_raters['birth_country'].unique())
    major_field = CategoricalField('major', joke_raters['major'].unique())
    joke_genre_field = CategoricalField('joke genre',
        joke_raters['preferred_joke_genre'].unique())
    joke_type_field = CategoricalField('joke type',
        joke_raters['preferred_joke_type'].unique())
    music_genre_field = CategoricalField("Favorite Music Genre",
        joke_raters['favorite_music_genre'].unique())
    movie_genre_field = CategoricalField("Favorite Movie Genre",
        joke_raters['favorite_movie_genre'].unique())

    all_user_fields = [
        gender_field,
        age_field,
        birth_country_field,
        major_field,
        joke_genre_field,
        joke_genre_field, # Twice because we have two preferred genre fields
        joke_type_field,
        music_genre_field,
        movie_genre_field
    ]

    return [field.read_input() for field in all_user_fields]


def user_distance(user_frame, user1, user2):
    """
    This function will return a form of distance between the users based on
    their features. This will iterate through all the features and for numeric
    fields it will simply take the difference in values and for categorical
    values it will use the discrete metric where it is 1 if the fields are
    different and 0 otherwise. Then the distance between the users is simply
    the sum of all these values.

    Args:
        user_frame (pd.DataFrame): A dataframe containing all the user data
        user1 (int): The index of user1
        user2 (int): The index of user2
    """

    if user1 not in user_frame.index:
        raise IndexError("The user {} is not found in the user frame"
            .format(user1))
    if user2 not in user_frame.index:
        raise IndexError("The user {} is not found in the user frame"
            .format(user2))

    distance = 0
    for feature in user_features:
        if feature.kind == 'numeric':
            # For numeric features, we just use standard metric.
            distance += abs(user_frame[feature.name][user1]
                            - user_frame[feature.name][user2])
        elif feature.kind == 'categorical':
            # For categorical features we use the discrete metric.
            if (user_frame[feature.name][user1]
                == user_frame[feature.name][user2]):
                feature_dist = 0
            else:
                feature_dist = 1
            distance += feature_dist
    return distance
