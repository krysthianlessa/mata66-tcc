from pandas import DataFrame
from numpy import array
import re

class RatingDataset():
    
    def __init__(self, item_id_col="movieId"):
        self.item_id_col = item_id_col
    
    def to_5_range(self, ratings_arr, r_max, r_min):

        if (r_min >= 0.0 and r_max <= 5.0):
            return ratings_arr

        return (array(ratings_arr) - r_min)*(5.0/(r_max-r_min))

    def preprocess_ratings(self, ratings_df, removed_item_ids, rate_factor=0.8):

        r_max = ratings_df.rating.max()
        ratings_df.loc[:,"rating"] = self.to_5_range(ratings_df.rating, r_max, ratings_df.rating.min())

        ratings_df = ratings_df[~ratings_df[self.item_id_col].isin(removed_item_ids)]
        ratings_df.reset_index(inplace=True)
        ratings_df.drop(columns=['index'], inplace=True)
        ratings_df = ratings_df[ratings_df.rating >= r_max*rate_factor]

        user_counts = DataFrame(ratings_df.userId.value_counts())
        keep_users = user_counts[user_counts['count'] >= 20].index
        ratings_df = ratings_df[ratings_df.userId.isin(keep_users)]

        return ratings_df[[self.item_id_col,"userId", "rating"]].rename(columns={self.item_id_col: "itemId"})

    
class MovieDataset():

    def __init__(self, description_df:DataFrame, item_id_col="movieId") -> None:
        self.description_df = description_df
        self.missing_description_list = list(description_df[description_df.overview.isna()]['movieId'])
        self.description_df.dropna(inplace=True)
        self.item_id_col = item_id_col
    
    def join_movies_details(self, movies_df:DataFrame) -> DataFrame:
        movies_df = movies_df[~movies_df[self.item_id_col].isin(self.missing_description_list)]
        movie_details_df = movies_df.set_index(self.item_id_col).join(self.description_df.set_index(self.item_id_col), how='left')
        movie_details_df.reset_index(inplace=True)
        movie_details_df.replace('(no genres listed)', '', inplace=True)
        movie_details_df.loc[:,'genres'] = movie_details_df['genres'].map(lambda x: x.lower().split('|'))
        movie_details_df.drop(columns=['tmdbId'], inplace=True)
        movie_details_df.set_index(self.item_id_col, inplace=True)

        return self.__create_bag_of_words(movie_details_df).rename(columns={self.item_id_col: "itemId"})
    
    def clean_spaces(self, str):
        return re.sub(r"\s{2,}", " ", str)
    
    def __create_bag_of_words(self, df:DataFrame, columns=["overview", "genres"]) -> DataFrame:
        df = df.copy()
        df.loc[:,'bag_of_words'] = ''

        for index, row in df[columns].iterrows():
        
            bag_words = ""
            for col in columns:
                bag_words = ' '.join(row[col])
            df.loc[index,'bag_of_words'] = bag_words
        
        df.loc[:,"bag_of_words"] = df["bag_of_words"].apply(self.clean_spaces)
        return df[["bag_of_words", "title"]].rename(columns={"bag_of_words": "description"})
    
    