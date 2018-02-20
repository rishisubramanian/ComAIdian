import numpy as np
import pandas as pd
import tensorflow as tf

def stringToBinaryDict (dictionary, array):
    uniq_array = array.unique()
    arraySize = uniq_array.size # find out the unique size of an array
    for i in range(arraySize):
        alist =  [0 for j in range(arraySize)]
        alist[i] = 1
        dictionary[uniq_array[i]] = alist
    
    return dictionary, uniq_array

def vectorization(jokes, rater, rating):
    jokerater = pd.read_csv(rater, quotechar="@")

    dictionaries = []
    longDataVector = []
    featureCountVector = pd.Series()
    categoricalFeatures = jokerater.columns.values[2:10]

    for feature in categoricalFeatures:
        dictionary = {}
        dictionaries.append(stringToBinaryDict(dictionary, jokerater[feature]))

    for i in range(len(jokerater)):
        row = jokerater.iloc[i]
        array = []
        for j in range(2,10):
            array = np.append(array, dictionaries[j-2][row[j]])

        array = np.append(array,row[10])
        longDataVector.append(array)
    
    for i in range(2,11):
        featureCountVector = featureCountVector.append(jokerater.iloc[:,i].value_counts())
    
    return pd.DataFrame(longDataVector), pd.DataFrame(featureCountVector, columns = ["value"])

annData, rfData = vectorization("Joke.csv", "JokeRater.csv", "JokeRating.csv")
