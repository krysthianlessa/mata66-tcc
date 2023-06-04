from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame

class RecomenderHandler():

    def __init__(self, items_df:DataFrame):

        items_ids = items_df.index.to_list()
        vec_matrix = TfidfVectorizer().fit_transform(items_df["description"])

        self.similarity_df = DataFrame(cosine_similarity(vec_matrix, vec_matrix), 
                                    columns=items_ids, 
                                    index=items_ids)
        
    def __recommender(self, items_interacteds_ids, items_to_recomend, cosine_similarity_matrix)->list:

        similaritys = cosine_similarity_matrix[items_to_recomend][cosine_similarity_matrix.index.isin(items_interacteds_ids)]
        top_10_items = similaritys.mean().sort_values(ascending = False).iloc[0:10].index.to_list()

        return top_10_items
    
    def get_recomendations(self, items_df:DataFrame, profile_ratings_df:DataFrame, frac:float, seed) -> list:
        train_profile_ratings = profile_ratings_df.sample(frac=frac, random_state=seed)
        test_profile_ratings = profile_ratings_df[~profile_ratings_df.itemId.isin(train_profile_ratings.itemId)]

        items_interacteds = train_profile_ratings.sample(15).itemId.to_list()
        items_to_recomend = test_profile_ratings.sample(5).itemId.to_list() + items_df.sample(20).index.to_list()

        items_recomended = self.__recommender(items_interacteds, items_to_recomend, self.similarity_df)
        relevance = [True if movie in test_profile_ratings.itemId.to_list() else False for movie in items_recomended]

        return relevance
    
    