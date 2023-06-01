import pandas as pd
import numpy as np

class RatingItemsLimiter():

    def __init__(self, max_users:int=6000, top_users_quantile=0.75, bottom_users_quantile=0.25, min_ratings=20, data_uri:str=None):

        self.items_cut_uri = f"{data_uri}/items_c_i{max_users}_t{top_users_quantile}_b{bottom_users_quantile}.csv"
        self.ratings_cut_uri = f"{data_uri}/ratings_c_i{max_users}_t{top_users_quantile}_b{bottom_users_quantile}.csv"

        self.data_uri = data_uri
        self.max_users = max_users
        self.top_users_quantile = top_users_quantile
        self.bottom_users_quantile = bottom_users_quantile
        self.min_ratings = min_ratings
        
    def limit(self, items_df:pd.DataFrame, ratings_df:pd.DataFrame):
        
        items_df, ratings_df = self.__limit(items_df, ratings_df)

        if self.data_uri is None:
            return items_df, ratings_df
        
        items_df.to_csv(self.items_cut_uri)
        ratings_df.to_csv(self.ratings_cut_uri, index=False)

        return items_df, ratings_df
        
    def __limit(self, items_df:pd.DataFrame, ratings_df:pd.DataFrame):

        if len(ratings_df.userId.unique()) <= self.max_users:
            return items_df, ratings_df

        users_ratings_counts_df = pd.DataFrame(ratings_df['userId'].value_counts())
        bottom_cut = max(users_ratings_counts_df['count'].quantile(self.bottom_users_quantile), self.min_ratings)
        users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] < users_ratings_counts_df['count'].quantile(self.top_users_quantile)]
        users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] > bottom_cut].iloc[0:self.max_users]
   
        ratings_df = ratings_df.loc[ratings_df['userId'].isin(users_ratings_counts_df.index)]
        items_df = items_df.loc[np.intersect1d(items_df.index, ratings_df.itemId.unique())]

        return items_df, ratings_df