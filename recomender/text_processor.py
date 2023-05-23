import string

import pandas as pd
from nltk import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from unidecode import unidecode
import nltk


class TextProcessor():
    
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

    def __create_bag_of_words(self, df,columns=["overview", "genres"]) -> pd.DataFrame:
        df = df.copy()
        df.loc[:,'bag_of_words'] = ''

        for index, row in df[columns].iterrows():
        
            bag_words = ""
            for col in columns:
                bag_words = ' '.join(row[col])
            df.loc[index,'bag_of_words'] = bag_words
        
        return df[["bag_of_words", "title"]]
    
    def pre_process(self, movies_df, stopwords_removal=True, lemmatization=True, stemmization=True) -> pd.DataFrame:
        df = movies_df.copy()
        df['overview'] = df['overview'].apply(self.__nlp_pre_process, args=(stopwords_removal, lemmatization, stemmization))
        df['overview'] = df['overview'].apply(str.split)
        df = self.__create_bag_of_words(df)

        return df
    
    