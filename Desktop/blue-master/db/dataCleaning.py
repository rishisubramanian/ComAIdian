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
    
def DataCleaning(csvfile, nrows):
    data = pd.read_csv(csvfile, nrows = nrows)
    stopword = CreateMyStopWords()
    porterStemmer = PorterStemmer()
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        for j in [1, 2]:
            sentence = row[j].replace("’", "'").lower()
            for k in range(len(sentence)):
                if (ord(sentence[k]) >= 128):
                   sentence = sentence.replace(sentence[k], ' ')
                    
            words = word_tokenize(sentence)

            cleanData = []

            for w in words:
                if w not in stopword and w not in punctuation:
                    cleanData.append(porterStemmer.stem(w))
            
            cleanSentence = ' '.join(cleanData)
            data.set_value(i, data.columns[j], cleanSentence)
    
    return data
