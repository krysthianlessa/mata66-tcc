import string

import pandas as pd
from nltk import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from unidecode import unidecode
import nltk

class NLPProcessor():
    
    def __init__(self) -> None:
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def __nlp_pre_process(self, corpus, stop_words_remv = True, lemmatization = True, stemmization = True) -> str:
    
        corpus = corpus.lower()
        corpus = unidecode(corpus) # remove non-ascii characters

        punctuations = list(string.punctuation)
        punctuations.append('...')
        
        corpus = " ".join([token for token in wordpunct_tokenize(corpus) if token not in punctuations])
        
        if (stop_words_remv):
            stopset = stopwords.words('english')
            corpus = " ".join([token for token in word_tokenize(corpus) if token not in stopset])

        if (lemmatization):
            lemmatizer = WordNetLemmatizer()
            corpus = " ".join([lemmatizer.lemmatize(token) for token in word_tokenize(corpus)])
        
        if (stemmization):
            stemmer = PorterStemmer()
            corpus = " ".join([stemmer.stem(token) for token in word_tokenize(corpus)])
            
        return corpus
    
    def pre_process(self, text_series:pd.Series, stopwords_removal=True, lemmatization=True, stemmization=True) -> pd.DataFrame:
        return text_series.apply(self.__nlp_pre_process, args=(stopwords_removal, lemmatization, stemmization))    