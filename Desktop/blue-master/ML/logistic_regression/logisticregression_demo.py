# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:59:57 2017
@author: ckchiruka
"""
import sqlite3 as sql
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

def readCleanData():
    '''
    This function's primary concern is to read in the databases and output DataFrames that
    are for the most part clean. It also uses LabelEncoder to encode all of the categorical
    data. We keep all numerical data as is. 
    
    Output: dbfit - is a fully encoded rating and rater database that can be used 
                    in many ML techniques
            pred_cols - are the prediction columns in the correct order (THIS MATTERS)
            Joke - this is our Joke database, used for outputing the best joke
            d - this is our encoder dictionary, which is used to encode the user input
    '''
    #connect to SQL database and read in all relevant tables
    jokedb = sql.connect("jokedb.sqlite3")
    JokeRater = pd.read_sql_query("SELECT * from JokeRater", jokedb)
    Joke = pd.read_sql_query("SELECT * from Joke", jokedb)
    JokeRating = pd.read_sql_query("SELECT * from JokeRating", jokedb)
    
    #drop JokeRater['joke_submitter_id'] b/c its useless
    #replace all empty/missing values with 'Other'
    JokeRater = JokeRater.drop(['joke_submitter_id'], axis = 1).replace('', 'Other')
    
    #Replace missing values with 'Question' (there's only 1)
    #Capitalize all joke_types
    Joke['joke_type'] = Joke['joke_type'].str.capitalize().replace('', 'Question')    
    
    #Set index to id and drop duplicate ratings
    JokeRating = JokeRating.set_index('id').drop_duplicates(['joke_id', 'joke_rater_id'], keep = 'first')
    #Remove rating which was 0
    JokeRating['rating'].loc[43553] = 1

    #Merge JokeRating and JokeRater databases based on joke_rater_id
    finalJokeDB = pd.merge(JokeRating, JokeRater, left_on = 'joke_rater_id', right_on = 'id')
    #drop two columns b/c they are useless
    finalJokeDB = finalJokeDB.drop(['id', 'joke_rater_id'], axis = 1)
   #get column names
    pred_cols = finalJokeDB.columns[2:]
    
    #let d be our dictionary encoder
    d = defaultdict(LabelEncoder)
    #fit encoder using apply
    dbfit = finalJokeDB.apply(lambda x: d[x.name].fit_transform(x))
    #inverse fit, so we can reverse encoings
    dbfit.apply(lambda x: d[x.name].inverse_transform(x))
    #reset age so we can input all ages
    dbfit['age'] = dbfit['age'].apply(lambda x: d['age'].inverse_transform(x))
    #return necessary databases
    
    return dbfit, pred_cols, Joke, d


def newUser(pred_cols, d):
    '''
    This function simulates the new user input values.
    input: pred_cols - are the prediction columns in the correct order (THIS MATTERS)
           d - this is our encoder dictionary, which is used to encode the user input
    output: UserJoke - Fully encoded user features
    '''
    #simulate input values
    cols = pred_cols.drop(['age'])
    UserJoke = ['Male', 'China', 'Physics', 'Politics', 'Programming', 'Puns', 'Rock', 'Thriller']
    UserJoke = pd.DataFrame(UserJoke).transpose()
    UserJoke.columns = cols
    #encode all input values besides age
    UserJoke = UserJoke.apply(lambda x: d[x.name].transform(x))
    #add in age
    UserJoke['age'] = 19
    #put columns in correct order
    UserJoke = UserJoke[pred_cols]
    return UserJoke
    
def logRegression(dbfit, UserJoke):
    '''
    This function uses logistic regression to predict every joke independent of all other 
    jokes in the database. 
    input: dbfit - a fully encoded user database including features
           UserJoke - fully encoded new user features
    output: prediction - a 1 - 5 rating of each joke
    '''
    #initialize predictions
    prediction = []
    #get user features
    #go through all jokes
    for i in dbfit['joke_id'].sort_values().unique():
        #initialize LG
        lr = LogisticRegression()
        #get correct data
        data = dbfit.loc[(dbfit['joke_id'] == i)]
        #fit data
        lr.fit(data.iloc[:, 2:], data.iloc[:, 0])
        #predict
        prediction.append(np.sum(lr.predict_proba(UserJoke.values.reshape(1, -1)) * [1, 2, 3, 4, 5]))
    #create datafram
    prediction = pd.DataFrame(prediction)
    return prediction

def printRatings(ratings, Joke):
    '''
    This function prints ratings. 
    input: ratings - ratings for every joke
           Joke - Joke database
    '''
    ratings = ratings.sort_values(by = 0, ascending = False)
    #self-explanatory
    for i in ratings.index:
        print('\nJoke: ')
        print(Joke.loc[i].joke_text)
        rating = input('What is your rating? ').strip()
        leave = input("Do you want to exit? " ).strip()
        if leave == "True":
            break

def main():
    dbfit, predCols, Joke, d = readCleanData()
    UserJoke = newUser(predCols, d)
    prediction = logRegression(dbfit, UserJoke)
    printRatings(prediction, Joke)
    
if __name__ == "__main__": main()