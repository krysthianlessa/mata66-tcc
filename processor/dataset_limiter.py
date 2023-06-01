from processor.items_processor import ItemProcessor
from processor.ratings_procesor import RatingProcessor

import pandas as pd

class RatingItemsLimiter():

    def __init__(self, items_df:pd.DataFrame, ratings_df:pd.DataFrame):
        self.items_df = items_df
        self.ratings_df = ratings_df
        
    def limit(self, max_items:int=600, top_users_quantile=0.75, buttom_users_quantile=0.25, data_uri:str=None):
        
        if not self.__limit(max_items, top_users_quantile, buttom_users_quantile) and data_uri is None:
            return self
        
        self.item_cut_uri = f"{data_uri}/items_c_i{max_items}_t{top_users_quantile}_b{buttom_users_quantile}.csv"
        self.items_df.to_csv(self.item_cut_uri)

        self.ratings_cut_uri = f"{data_uri}/ratings_c_i{max_items}_t{top_users_quantile}_b{buttom_users_quantile}.csv"
        self.ratings_df.to_csv(self.ratings_cut_uri)

        return self
        
    def __limit(self, max_items:int=600, top_users_quantile=0.75, buttom_users_quantile=0.25):

        if len(self.items_df) <= max_items:
            return False
        
        items_ratings_counts_df = pd.DataFrame(self.ratings_df['itemId'].value_counts())
        items_ratings_counts_df = items_ratings_counts_df[items_ratings_counts_df['count'] < items_ratings_counts_df['count'].quantile(0.75)]
        items_ratings_counts_df = items_ratings_counts_df[items_ratings_counts_df['count'] > items_ratings_counts_df['count'].quantile(0.25)].iloc[0:max_items]
        self.ratings_df = self.ratings_df[self.ratings_df['itemId'].isin(items_ratings_counts_df.index)]
        self.items_df = self.items_df[self.items_df.index.isin(items_ratings_counts_df.index)]
        
        users_ratings_counts_df = pd.DataFrame(self.ratings_df['userId'].value_counts())
        users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] < users_ratings_counts_df['count'].quantile(top_users_quantile)]
        users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] > users_ratings_counts_df['count'].quantile(buttom_users_quantile)]
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(users_ratings_counts_df.index)]
        self.items_df = self.items_df[self.items_df.index.isin(self.ratings_df.itemId.unique())]

        return True