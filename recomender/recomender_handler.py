from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame, Series

class RecomenderHandler():

    COSINE_SIMILARITY = 'cosine'
    PEARSON_SIMILARITY = 'pearson'
    JACCARD_SIMILARITY = 'jaccard'
    
    def __init__(self, items_df:DataFrame, similarity:str=COSINE_SIMILARITY):

        items_ids = items_df.index.to_list()
        vec_matrix = TfidfVectorizer().fit_transform(items_df["description"])

        if similarity == self.COSINE_SIMILARITY:
            self.similarity_df = DataFrame(cosine_similarity(vec_matrix, vec_matrix), 
                                    columns=items_ids, 
                                    index=items_ids)
        elif similarity == self.PEARSON_SIMILARITY:
            self.similarity_df = DataFrame(vec_matrix.todense(),
                                           columns=items_ids,
                                           index=items_ids).corr(method="pearson")
        else:
            self.similarity_df = self.__jaccard_simiarity(items_df["description"], items_ids)

        
    def __jaccard_sim(self, text_x_set:set, text_y_set:set):

        c = text_x_set.intersection(text_y_set)
        return len(c)*1.0 / (len(text_x_set) + len(text_y_set) - len(c))
    
    
    def __jaccard_simiarity(self, text_array:Series, items_ids):

        text_df = DataFrame({"text": text_array, "index": items_ids})
        text_df.loc[:,"text"] = text_df.text.str.replace(",","").str.replace(".","").str.split()

        text_cartesian_df = text_df.merge(text_df, how='cross')
        text_cartesian_df['similarity'] = [self.__jaccard_sim(set(text_x), set(text_y)) for text_x, text_y in zip(text_cartesian_df.text_x, text_cartesian_df.text_y)]

        similarity_df = DataFrame(columns=items_ids, index=items_ids)

        for i in text_cartesian_df.index:
            cols_row = text_cartesian_df.loc[i]
            similarity_df.loc[cols_row['index_x'], cols_row['index_y']] = cols_row['similarity']

        return similarity_df
    
        
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
    
    