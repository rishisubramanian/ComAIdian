import sqlite3 as sql
import pandas as pd
import numpy as np
from improved_similarity import find_user_similarity

def readCleanData():
    """
    This function will pull from the sqlite database 'jokedb.sqlite3' data
    related to the joke raters, joke ratings, and jokes themselves.
    It will minimally format the data, removing duplicate entries.

    It will return three DataFrames the first containing information
    regarding the joke raters, the second containing information about the
    joke ratings, and the third containing information about the jokes.

    Returns:
        tuple(DataFrame, DataFrame, DataFrame): Three dataframes in order,
            joke raters, joke ratings, jokes
    """
    jokedb = sql.connect("jokedb.sqlite3")
    JokeRater = pd.read_sql_query("SELECT * from JokeRater", jokedb)
    Joke = pd.read_sql_query("SELECT * from Joke", jokedb)
    JokeRating = pd.read_sql_query("SELECT * from JokeRating", jokedb)
    
    #drop joke_submitter_id
    JokeRater = JokeRater.drop(['joke_submitter_id'], axis = 1).set_index('id')
    #set joke index
    Joke = Joke.set_index('id')
    #drop duplicates
    JokeRating = JokeRating.drop_duplicates(['joke_id', 'joke_rater_id'], keep = 'first')
    #create pivot table
    JokeRating = JokeRating.pivot(index = 'joke_id', columns = 'joke_rater_id', values = 'rating').transpose().fillna(2.5)
    JokeRater = JokeRater.loc[JokeRating.index]

    return JokeRater, JokeRating, Joke
 
def userInput():
    #just for now, ideally we need drop down menus
    gender = input('What is your gender? ').strip()
    age = int(input('What is your age? ').strip())
    birth_country = input('What is your birth country? ').strip()
    major = input('What is your major? ').strip()
    joke_genre1 = input('What is your primary preferred joke genre? ').strip()
    joke_genre2 = input('What is your secondary preferred joke genre? ').strip()
    joke_type = input('What is your preferred joke type? ').strip()
    music_genre = input('What is your favorite music genre? ').strip()
    movie_genre = input('What is your favorite movie genre? ').strip()
    
    return [gender, age, birth_country, major, joke_genre1, joke_genre2, joke_type,
            music_genre, movie_genre]
  

def preliminaryJokeRatings(similarity, JokeRating):
    #simple dot matrix multiplication of similarity times JokeRatings
    return JokeRating.transpose().dot(similarity)
    
def printRatings(ratings, Joke):
    #simple print ratings, pretty self explanatory
    ratings = ratings.sort_values(ascending = False)
    new_user = []
    for i in ratings.index:
        print('\nJoke: ')
        print(Joke.loc[i].joke_text)
        new_user.append([input("What is your rating? ").strip(), i])
        leave = input("Do you want to exit? " ).strip()
        if leave == "True":
            break
    return new_user

def main():
    someGuy = userInput()
    JokeRater, JokeRating, Joke = readCleanData()
    attr_dict = {"gender":0, "age":1, "birth_country":2, "major":3, "preferred_joke":[4, 5], "joke_type":6, "music":7, "movie": 8}
    similarity = find_user_similarity(np.array(someGuy), np.array(JokeRater), attr_dict)
    ratings = preliminaryJokeRatings(similarity, JokeRating)
    new_user = printRatings(ratings, Joke)
    
if __name__ == "__main__": main()