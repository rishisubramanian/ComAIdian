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
import fancyimpute
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn import svm
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

def complete_matrix(matrix, method):
    """
    This method will complete the matrix based on the method specified.
    This will not change the matrix in place instead constructing a new matrix
    with the completed entries.

    Args:
        matrix (np.array): A numpy array with nan entries representing
            missing entries
        method (str): One of 'mean', 'median', 'soft_impute', 'iterativeSVD',
            'MICE', 'matrix_factorization', 'nuclear_norm_minimization',
            'KNN', 'gauss'
    Returns:
        np.array: The completed matrix
    """
    if method == 'mean':
        imputer = fancyimpute.SimpleFill('mean')
    elif method == 'median':
        imputer = fancyimpute.SimpleFill('median')
    elif method == 'gauss':
        imputer = fancyimpute.SimpleFill('random')
    elif method == 'soft_impute':
        imputer = fancyimpute.SoftImpute()
    elif method == 'iterativeSVD':
        imputer = fancyimpute.IterativeSVD()
    elif method == 'MICE':
        imputer = fancyimpute.MICE()
    elif method == 'matrix_factorization':
        imputer = fancyimpute.MatrixFactorization()
    elif method == 'nuclear_norm_minimization':
        imputer = fancyimpute.NuclearNormMinimization()
    elif method == 'KNN':
        imputer = fancyimpute.KNN()
    else:
        raise ValueError("Unrecognized method passed in")

    return imputer.complete(matrix)


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
    jokedb = sql.connect("updatedjokedb.sqlite3")
    JokeRater = pd.read_sql_query("SELECT * from JokeRater", jokedb)
    Joke = pd.read_sql_query("SELECT * from Joke", jokedb)
    JokeRating = pd.read_sql_query("SELECT * from JokeRating", jokedb)
    
    #drop JokeRater['joke_submitter_id'] b/c its useless
    #replace all empty/missing values with 'Other'
    JokeRater = JokeRater.drop(['joke_submitter_id'], axis = 1).replace('', 'Other')
    JokeRater = JokeRater[JokeRater.id.isin(JokeRating['joke_rater_id'].unique())]

    
    #add new user to JokeRater
    userNumber = max(JokeRater['id'])+1
    userFeat = newUser(JokeRater.columns, userNumber)
    JokeRater = pd.concat([JokeRater, userFeat])
    JokeRater, userFeat = encodeDB(JokeRater, userNumber)
    
    JokeRater_train, JokeRater_test = train_test_split(JokeRater, test_size=25/93)
    
    #Replace missing values with 'Question' (there's only 1)
    #Capitalize all joke_types
    Joke['joke_type'] = Joke['joke_type'].str.capitalize().replace('', 'Question')    
    
    #Set index to id and drop duplicate ratings
    JokeRating = JokeRating.set_index('id').drop_duplicates(['joke_id', 'joke_rater_id'], keep = 'first')
    #Remove rating which was 0
    
    #completeMatrix using Matrix Completion
    JokeRating = JokeRating.pivot(index = 'joke_id', columns = 'joke_rater_id', values = 'rating')
    JokeRating = pd.DataFrame(complete_matrix(JokeRating, 'KNN'), index = JokeRating.index, columns = JokeRating.columns)
    JokeRating = JokeRating.reset_index()
    JokeRating = pd.melt(JokeRating, id_vars = ['joke_id'])
    JokeRating['joke_rater_id'] = pd.to_numeric(JokeRating['joke_rater_id'])
    JokeRating['value'] = pd.to_numeric(round(JokeRating['value']))
    
    #Merge JokeRating and JokeRater databases based on joke_rater_id
    finalJokeDB_train = pd.merge(JokeRating, JokeRater_train, left_on = 'joke_rater_id', right_on = 'id')
    #drop two columns b/c they are useless 
    finalJokeDB_train = finalJokeDB_train.drop(['joke_rater_id'], axis = 1)
    
    #Merge JokeRating and JokeRater databases based on joke_rater_id
    finalJokeDB_test = pd.merge(JokeRating, JokeRater_test, left_on = 'joke_rater_id', right_on = 'id')
    #drop two columns b/c they are useless
    finalJokeDB_test = finalJokeDB_test.drop(['joke_rater_id'], axis = 1)
    
    #Merge JokeRating and JokeRater databases based on joke_rater_id
    #finalJokeDB = pd.merge(JokeRating, JokeRater, left_on = 'joke_rater_id', right_on = 'id')
    #drop two columns b/c they are useless
    #finalJokeDB = finalJokeDB.drop(['joke_rater_id'], axis = 1)
    
    
    return finalJokeDB_train, finalJokeDB_test, Joke
    #return finalJokeDB, Joke, userFeat


def encodeDB(database, userNumber):
    '''
    This function encodes the JokeRater database using the LabelEncoder package in sklearn
    inputs: database - This is the JokeRater db
            userNumber - max of the JokeRater numbers
    outputs: dbfit - a fully encoded database
             userFeat - the userFeatures to train upon
    '''
    #let d be our dictionary encoder
    d = defaultdict(LabelEncoder)
    #fit encoder using apply
    dbfit = database.apply(lambda x: d[x.name].fit_transform(x))
    #inverse fit, so we can reverse encoings
    dbfit.apply(lambda x: d[x.name].inverse_transform(x))
    #reset age so we can input all ages
    dbfit['age'] = dbfit['age'].apply(lambda x: d['age'].inverse_transform(x))
    dbfit['id'] = dbfit['id'].apply(lambda x: d['id'].inverse_transform(x))
    #return necessary databases
    userFeat = dbfit.loc[dbfit['id'] == userNumber]
    return dbfit, userFeat


def newUser(pred_cols, number):
    '''
    This function simulates the new user input values.
    input: pred_cols - are the prediction columns in the correct order (THIS MATTERS)
           d - this is our encoder dictionary, which is used to encode the user input
    output: UserJoke - Unencoded user features
    '''
    #simulate input values
    userJoke = [number, 'Male', 'United States', 'Statistics', 'Politics', 'Sports', 'Puns', 'Rock', 'Comedy', 21]
    userJoke = pd.DataFrame(userJoke).transpose()
    userJoke.columns = pred_cols
    userJoke = userJoke[pred_cols]
    return userJoke
    
def logRegression(dbfit, dbtest):
    '''
    This function uses logistic regression to predict every joke independent of all other 
    jokes in the database. 
    input: dbfit - a fully encoded user database including features
           UserJoke - fully encoded new user features
    output: prediction - a 1 - 5 rating of each joke
    '''
    #initialize predictions
    fullPred = pd.DataFrame()
    fullTruth = pd.DataFrame()
    fullDecision = pd.DataFrame()
    #get user features
    #go through all jokes
    for j in dbtest['id'].unique():
        UserJoke = dbtest.loc[dbtest.id == j]
        prediction = []
        for i in dbfit['joke_id'].sort_values().unique():
        #initialize LG
            lr = LogisticRegression()
            #clf1 = svm.SVC(kernel='linear', probability = True)
            #eclf = VotingClassifier(estimators=[('lr', lr), ('svm', clf1)], voting='soft')
            #get correct data
            data = dbfit.loc[(dbfit['joke_id'] == i)]
            #fit data
            lr.fit(data.iloc[:, 3:], data.iloc[:, 1])
            #append prediction, ground truth, and decision function
            #for VotingClassifier there is no decision function, so you have to delete that
            prediction.append([lr.predict(UserJoke.iloc[0, 3:].values.reshape(1, -1)),
                               UserJoke['value'].loc[UserJoke.joke_id == i].reset_index(drop=True)[0], 
                               lr.decision_function(UserJoke.iloc[0, 3:].values.reshape(1, -1))])         
        #make into DataFrame
        prediction = pd.DataFrame(prediction, columns = ['ratings', 'truth', 'decision']).sort_values(by='ratings', 
                                 ascending = False).reset_index(drop = True)
        #seperate in order to not mess things up
        fullPred[j] = prediction['ratings'] - prediction['truth']
        fullTruth[j] = prediction['truth']
        fullDecision[j] = prediction['decision']
    return fullPred, fullTruth, fullDecision
    

##################################################################################
#main starts here
#get data
db_train, db_test, Joke = readCleanData()
#predicct
prediction, truth, decision = logRegression(db_train, db_test)

#plot MSE
#in order to get passive_svm/passive_ensemble run twice using those two algorithms
passive_logistic = np.mean(np.square(prediction), axis = 1)
#plt.plot(passive_svm, color = 'k', 
  #       label = 'Linear-SVM (average = {})'.format(round(np.mean(passive_svm),2)))
plt.plot(passive_logistic, 
         label = 'Logistic Regression (average = {})'.format(round(np.mean(passive_logistic),2)))
#plt.plot(passive_ensemble, 
 #        label = 'Voting Classifer (SVM/Logistic) (average = {})'.format(round(np.mean(passive_ensemble),2)))
plt.title('MSE of Different Content-Based Passive Learning Algorithms')
plt.xlabel('Number of Jokes Predicted')
plt.ylabel('Mean Squared Error')
plt.legend(loc=1, borderaxespad=0.)
plt.show()



#ROC CURVES
#only works with logistic
ovrRoc = []
for i in truth:
    #flatten the binarized ratings
    binary = label_binarize(truth[i], classes=[1, 2, 3, 4, 5]).ravel()
    #flatten the decision function
    score = pd.DataFrame([j.ravel() for j in decision[i]]).fillna(0).values.ravel()
    #calculate fpr, tpr
    fpr, tpr, _ = roc_curve(binary, score)
    #need average auc
    ovrRoc.append(auc(fpr,tpr))
    #PLOT
    plt.plot(fpr, tpr, lw=1)

#other plot stuff
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Averaged ROC Curves Using Passive Logistic Regression')
plt.text(0.705, 0.01, 'Average AUC: {}'.format(round(np.mean(ovrRoc),2)), style='italic',
        bbox={'facecolor':'white', 'alpha':1, 'pad':10})
plt.show()  



#PR CURVES
#only works with logistic
ovrPr = []
for i in truth:
    #flatten the binarized ratings
    binary = label_binarize(truth[i], classes=[1, 2, 3, 4, 5]).ravel()
    #flatten the decision function
    score = pd.DataFrame([j.ravel() for j in decision[i]]).fillna(0).values.ravel()
    #calculate precision, recall
    precision, recall, _ = precision_recall_curve(binary, score)
    #need average auprc
    ovrPr.append(auc(recall,precision))
    #PLOT
    plt.plot(recall, precision, lw=1)

#other plot stuff
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Micro-Averaged PR Curves Using Passive Logistic Regression')
plt.text(0.663, 0.01, 'Average AUPRC: {}'.format(round(np.mean(ovrPr),2)), style='italic',
        bbox={'facecolor':'white', 'alpha':1, 'pad':10})
plt.show()





