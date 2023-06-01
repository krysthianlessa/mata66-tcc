import pandas as pd

class RatingItemsLimiter():

    def __init__(self, max_items:int=600, top_users_quantile=0.75, bottom_users_quantile=0.25, data_uri:str=None):

        self.item_cut_uri = f"{data_uri}/items_c_i{max_items}_t{top_users_quantile}_b{bottom_users_quantile}.csv"
        self.ratings_cut_uri = f"{data_uri}/ratings_c_i{max_items}_t{top_users_quantile}_b{bottom_users_quantile}.csv"

        self.data_uri = data_uri
        self.max_items = max_items
        self.top_users_quantile = top_users_quantile
        self.bottom_users_quantitle = bottom_users_quantile

        
    def limit(self, items_df:pd.DataFrame, ratings_df:pd.DataFrame):
        
        items_df, ratings_df = self.__limit()

        if self.data_uri is None:
            return items_df, ratings_df
        
        items_df.to_csv(self.item_cut_uri)
        ratings_df.to_csv(self.ratings_cut_uri)

        return items_df, ratings_df
        
    def __limit(self, items_df, ratings_df):

        if len(items_df) <= self.max_items:
            return items_df, ratings_df
        
        items_ratings_counts_df = pd.DataFrame(ratings_df['itemId'].value_counts())
        items_ratings_counts_df = items_ratings_counts_df[items_ratings_counts_df['count'] < items_ratings_counts_df['count'].quantile(0.75)]
        items_ratings_counts_df = items_ratings_counts_df[items_ratings_counts_df['count'] > items_ratings_counts_df['count'].quantile(0.25)].iloc[0:self.max_items]
        ratings_df = ratings_df[ratings_df['itemId'].isin(items_ratings_counts_df.index)]
        items_df = items_df[items_df.index.isin(items_ratings_counts_df.index)]
        
        users_ratings_counts_df = pd.DataFrame(ratings_df['userId'].value_counts())
        users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] < users_ratings_counts_df['count'].quantile(self.top_users_quantile)]
        users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] > users_ratings_counts_df['count'].quantile(self.bottom_users_quantile)]
        ratings_df = ratings_df[ratings_df['userId'].isin(users_ratings_counts_df.index)]
        items_df = items_df[items_df.index.isin(ratings_df.itemId.unique())]

        return items_df, ratings_df