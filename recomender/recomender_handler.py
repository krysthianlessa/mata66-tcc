from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch

class RecomenderHandler():

    COSINE_SIMILARITY = 'cosine'
    JACCARD_SIMILARITY = 'jaccard'
    
    def __init__(self, items_df:DataFrame, similarity:str=COSINE_SIMILARITY):

        items_ids = items_df.index
        
        if similarity == self.COSINE_SIMILARITY:
            vec_matrix = TfidfVectorizer().fit_transform(items_df["description"])
            self.similarity_df = DataFrame(cosine_similarity(vec_matrix, vec_matrix), 
                                    columns=items_ids, 
                                    index=items_ids)
        elif similarity == self.JACCARD_SIMILARITY:
            self.similarity_df = self.__jarccard_similarity(items_df[['description']])

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def __jarccard_similarity(self, items_df:DataFrame, batch_size=10240) -> DataFrame:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        len_ = len(items_df)

        vectorizer = CountVectorizer(max_features=500, binary=True)
        text_count_tensor = torch.tensor(vectorizer.fit_transform(items_df['description']).toarray()).to(device)

        index_tensor = torch.arange(len_).to(device)
        cartesian_index = torch.cartesian_prod(index_tensor, index_tensor)
       
        union_tensor = None
        intersection_tensor = None
        i = 1
        for index_batch in self.batch(cartesian_index, batch_size):
            
            x_cartesian_batch = text_count_tensor[index_batch[:,0]]
            y_cartesian_batch = text_count_tensor[index_batch[:,1]]

            union_tensor_batch = (x_cartesian_batch+y_cartesian_batch)
            union_tensor_batch[union_tensor_batch > 1] = 1

            union_tensor_batch = union_tensor_batch.sum(axis=1)
            intersection_batch = (x_cartesian_batch*y_cartesian_batch).sum(axis=1)

            if union_tensor is None:
                union_tensor = union_tensor_batch
                intersection_tensor = intersection_batch
            else:
                union_tensor = torch.cat((union_tensor, union_tensor_batch))
                intersection_tensor = torch.cat((intersection_tensor,  intersection_batch))

            print(str(i*batch_size), end="\r")
            i += 1

        return DataFrame(np.reshape((intersection_tensor / union_tensor).cpu().numpy(), (len_,len_)), 
                                    columns=items_df.index,
                                    index=items_df.index)


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
    
    