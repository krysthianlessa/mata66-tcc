from recomender.nlp_processor import TextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame

class RecomenderHandler():
    
    def __init__(self, movies_df):
        
        self.cosine_sim = DataFrame(self.generate_similarity_matrix(movies_df), 
                                  columns=movies_df.index.to_list(), 
                                  index=movies_df.index.to_list())
    
    def recommender(self, movies_interacteds, movies_to_recomend, cosine_similarity)->list:

        similaritys = cosine_similarity[movies_to_recomend][cosine_similarity.index.isin(movies_interacteds)]
        average_similarity = similaritys.mean()
        top_10_movies = average_similarity.sort_values(ascending = False).iloc[0:10].index.to_list()

        return top_10_movies
    
    def generate_similarity_matrix(self, df):
        count = TfidfVectorizer()
        count_matrix = count.fit_transform(df['bag_of_words'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        return cosine_sim

        
    def get_recomendations(self, movies_df, profile, frac, seed):
    
        train_items = profile.sample(frac=frac, random_state=seed)
        test_items = profile[~profile.movieId.isin(train_items.movieId)]

        movies_interacteds = train_items.sample(15).movieId.to_list()

        movies_to_recomend = test_items.sample(5).movieId.to_list() + movies_df.sample(20).index.to_list()

        movies_recomended = self.recommender(movies_interacteds, movies_to_recomend, self.cosine_sim)

        relevance = [True if movie in test_items.movieId.to_list() else False for movie in movies_recomended]

        return relevance
    
    