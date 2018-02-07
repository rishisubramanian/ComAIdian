import sqlite3 as sql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import fancyimpute
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import VotingClassifier
import time

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
  

def logRegression(db_train, db_test):
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
    for i in db_train['joke_id'].sort_values().unique():
        #initialize LG
        lr = LogisticRegression()
        #clf1 = svm.SVC(kernel='linear', probability = True)
        #eclf = VotingClassifier(estimators=[('lr', lr), ('svm', clf1)], voting='soft')
        #get correct data
        data_train = db_train.loc[(db_train['joke_id'] == i)]
        #fit data
        lr.fit(data_train.iloc[:, 3:], data_train.iloc[:, 1])
        #predict
        prediction.append([i, lr.predict(db_test), lr.decision_function(db_test)])
    #create dataframe
    prediction = pd.DataFrame(prediction, columns = ['joke_id', 'ratings', 'decision'])
    return  prediction.sort_values(by = 'ratings', ascending = False).iloc[0]


def updateDB(finalJokeDB, userFeat, prediction, num):
    '''
    This function updates finalJokeDB and userFeat by adding a new row to userFeat and
    finalJokeDB for logistic regression to use. 
    input: finalJokeDB - database to update
           userFeat - the user features to update
           num - the rating to add to the user featuers
           prediction - the joke to remove from the database, and add to the feature pool
    output: finalJokeDB - new updated database
            userFeat - new updated user features
    '''
    #add rating to feature pool
    prediction = prediction['joke_id']
    userFeat[prediction] = num
    #add predicted joke to feature pool
    finalJokeDB[prediction] = 0
    #create temp to give predicted joke the ratings given by users
    temp = finalJokeDB.loc[finalJokeDB['joke_id'] == prediction].iloc[:, 1:3]
    #give new feature the rating what the users rated them
    for i in range(len(temp)):
        finalJokeDB.loc[finalJokeDB.id == temp['id'].iloc[i], prediction] = temp['value'].iloc[i]
    
    #delete the joke from the database
    finalJokeDB = finalJokeDB[finalJokeDB['joke_id'] != prediction]
            
    return finalJokeDB, userFeat
    
    
####################################################################################################
#main starts here
#get data 
finalJokeDB_train, finalJokeDB_test, Joke = readCleanData()
#initialize things necessary for graphing
allPred = pd.DataFrame()
allROC = []
allPR = []
#simulate active learning
for j in finalJokeDB_test['id'].unique():
    #get user ratings
    userDB = finalJokeDB_test.loc[finalJokeDB_test.id == j]
    #get user features
    userFeat = userDB.drop_duplicates(['gender', 'major'], keep = 'first').iloc[:, 3:]
    #initialize predictions and ground truth
    prediction = pd.DataFrame()
    groundTruth = []
    #need need finalDB_train to not change
    temp_train = finalJokeDB_train.iloc[:, 0:12]
    #imualate active learning
    for i in range(len(userDB)):
        #predict using regression or svm or both
        temp = logRegression(temp_train, userFeat)
        #get ground truth for that joke
        num = userDB['value'].loc[userDB.joke_id == temp['joke_id']].reset_index(drop = True)[0]
        #update temporary database
        temp_train, userFeat = updateDB(temp_train, userFeat, temp, num)
        #find error
        temp['ratings'] = temp['ratings'] - num
        #append stuff
        groundTruth.append(num)
        prediction = prediction.append(temp)
        #make sure its working
        
    #needs to be in order
    prediction = prediction.reset_index(drop = True)
    #get ratings
    allPred[j] = prediction['ratings']   
    #binarize groundTruth
    binTruth = label_binarize(groundTruth, classes=[1, 2, 3, 4, 5])
    #get fpr, tpr, roc_auc
    fpr, tpr, _ = roc_curve(binTruth.ravel(), pd.DataFrame([i.ravel() for i in prediction['decision']]).fillna(0).values.ravel())
    roc_auc = auc(fpr,tpr)
    allROC.append([fpr, tpr, roc_auc, j])
    #get precision, recall, pr_auc
    precision, recall, _ = precision_recall_curve(binTruth.ravel(), pd.DataFrame([i.ravel() for i in prediction['decision']]).fillna(0).values.ravel())
    pr_auc = auc(recall,precision)
    allPR.append([precision, recall, pr_auc, j])

    
active_logistic = np.mean(np.square(allPred), axis = 1)
#run 3 times as only svm, only logisitic, and both using the voting classifer
#plt.plot(np.mean(np.square(active_svm), axis = 1), color = 'k',
#         label = 'Linear-SVM (average = {})'.format(round(np.mean(np.mean(np.square(active_svm))),2)))
plt.plot(active_logistic, 
         label = 'Logistic Regression (average = {})'.format(round(np.mean(active_logistic),2)))
#plt.plot(np.mean(np.square(active_ensemble), axis = 1), 
#         label = 'Voting Classifer (SVM/Logistic) (average = {})'.format(round(np.mean(np.mean(np.square(active_ensemble))),2)))
plt.title('MSE of Different Content-Based Active Learning Algorithms')
plt.xlabel('Number of Jokes Predicted')
plt.ylabel('Mean Squared Error')
plt.legend(loc=1, borderaxespad=0.)
plt.show()

#plot ROC for all 25 users
for i in allROC:
    plt.plot(i[0], i[1], lw=1)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-Averaged ROC Curves Using Active Logistic Regression')
plt.text(0.727, 0.01, 'Average AUC: {}'.format(round(np.mean([i[2] for i in allROC]),2)), style='italic',
        bbox={'facecolor':'white', 'alpha':1, 'pad':10})
plt.show()  

#plot PR curves for all 25 users
for i in allPR:
    plt.plot(i[1], i[0], lw=1)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Micro-Averaged PR Curves Using Active Logistic Regression')
plt.text(0.663, 0.01, 'Average AUPRC: {}'.format(round(np.mean([i[2] for i in allPR]),2)), style='italic',
        bbox={'facecolor':'white', 'alpha':1, 'pad':10})
plt.show()
    



