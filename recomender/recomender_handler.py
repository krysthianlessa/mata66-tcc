from recomender.text_processor import TextProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecomenderHandler():
    
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

        
    def get_recomendations(self, movies_df, profile, cosine_sim, frac, seed):
    
        train_items = profile.sample(frac=frac, random_state=seed)
        test_items = profile[~profile.movieId.isin(train_items.movieId)]

        movies_interacteds = train_items.sample(15).movieId.to_list()

        movies_to_recomend = test_items.sample(5).movieId.to_list() + movies_df.sample(20).index.to_list()

        movies_recomended = self.recommender(movies_interacteds, movies_to_recomend, cosine_sim)

        relevance = [True if movie in test_items.movieId.to_list() else False for movie in movies_recomended]

        return relevance
    
    