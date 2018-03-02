import nltk

import numpy as np
import pandas as pd

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation

# nltk.download()
# nltk.download('punkt')
# if you have not downloaded the nltk corpus, then uncomment the lines above

def CreateMyStopWords ():
    stopword = stopwords.words("english")
    stopword.remove(u't')
    stopword.remove(u's')
    stopword.append(u"'s")
    stopword.append(u"'t")
    stopword.append(u"n't")
    stopword.append(u"'d")
    stopword.append(u"'re")
    stopword.append(u"cannot")
    stopword.append(u"'ll")
    return stopword
    
def is_valid_hyphen_word(str):
    flag = False
    
    if str[0].isalpha() and str[len(str) - 1].isalpha():
        for chr in str:
            if chr.isalpha():
                flag = False
            elif chr == "-":
                if flag:
                    return False
                else:
                    flag = True
            else:
                return False
        return True
    return False
    
def DataCleaningForKaggleQA(csvfile, nrows):
    data = pd.read_csv(csvfile, nrows = nrows)
    stopword = CreateMyStopWords()
    porterStemmer = PorterStemmer()
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        for j in [1, 2]:
            sentence = row[j].replace("’", "'").lower()
            for chr in sentence:
                if (ord(chr) >= 128):
                    sentence = sentence.replace(chr, '')
                    
           # print j, sentence
            words = word_tokenize(sentence)

            cleanData = []

            for w in words:
                if w not in stopword:
                    if all(chr not in punctuation for chr in w) or is_valid_hyphen_word(w):
                        cleanData.append(porterStemmer.stem(w))
            
            cleanSentence = ' '.join(cleanData)
            data.set_value(i, data.columns[j], cleanSentence)
    
    return data

def DataCleaningForKaggleSA(csvfile, nrows):
    data = pd.read_csv(csvfile, nrows = nrows)
    stopword = CreateMyStopWords()
    porterStemmer = PorterStemmer()
    
    for i in range(len(data)):
        row = data.iloc[i]
        sentence = row["Joke"].replace("’", "'").lower()
        for chr in sentence:
            if (ord(chr) >= 128):
                sentence = sentence.replace(chr, '')
                
        words = word_tokenize(sentence)
        cleanData = []
        
        for w in words:
            if w not in stopword:
                if all(chr not in punctuation for chr in w) or is_valid_hyphen_word(w):
                    cleanData.append(porterStemmer.stem(w))
            
        cleanSentence = ' '.join(cleanData)
        data.set_value(i, "Joke", cleanSentence)
        
    return data
    
def DataCleaningForOtherQA(csvfile):
    cols = ["question", "answer"]
    data = pd.read_csv(csvfile, names = cols, sep = "?", lineterminator = "#")
    data = data.drop(data.index[-1])
    stopword = CreateMyStopWords()
    porterStemmer = PorterStemmer()
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        for j in [0, 1]:
            sentence = row[j].replace("’", "'").replace("‘", "'").lower()
            for chr in sentence:
                if (ord(chr) >= 128):
                    sentence = sentence.replace(chr, '')

            words = word_tokenize(sentence)
            cleanData = []

            for w in words:
                if w not in stopword:
                    if all(chr not in punctuation for chr in w) or is_valid_hyphen_word(w):
                        cleanData.append(porterStemmer.stem(w))

            cleanSentence = ' '.join(cleanData)
            data.set_value(i, data.columns[j], cleanSentence)
        
    return data
