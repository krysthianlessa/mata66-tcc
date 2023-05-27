from pandas import DataFrame
from numpy import array
import re

class RatingDataset():
    
    def __init__(self, ratings_df:DataFrame, item_id_col:str, user_id_col:str):
        
        self.ratings_df = ratings_df.copy()
        if item_id_col != "itemId":
            self.ratings_df.rename(columns={item_id_col: "itemId"}, inplace=True)
        if user_id_col != "userId":
            self.ratings_df.rename(columns={user_id_col: "userId"}, inplace=True)
    
    def to_5_range(self, ratings_arr, r_max, r_min):

        if (r_min >= 0.0 and r_max <= 5.0):
            return ratings_arr

        return (array(ratings_arr) - r_min)*(5.0/(r_max-r_min))

    def process(self, removed_item_ids:list):

        self.ratings_df.loc[:,"rating"] = self.to_5_range(self.ratings_df.rating, self.ratings_df.rating.max(), self.ratings_df.rating.min())
        self.ratings_df = self.ratings_df[~self.ratings_df.itemId.isin(removed_item_ids)]
        self.ratings_df.reset_index(inplace=True)
        self.ratings_df.drop(columns=['index'], inplace=True)
        self.ratings_df = self.ratings_df[self.ratings_df.rating >= 4.0]

        user_counts = DataFrame(self.ratings_df.userId.value_counts())
        keep_users = user_counts[user_counts['count'] >= user_counts["count"].quantile(0.24)].index
        self.ratings_df = self.ratings_df[self.ratings_df.userId.isin(keep_users)]

        return self.ratings_df[["itemId", "userId", "rating"]]


class ItemDataset():

    def __init__(self, items_df: DataFrame, desc_col:str, item_id_col:str) -> None:
        self.items_df = items_df.copy()

        if desc_col != "description":
            self.items_df.rename(columns={desc_col: "description"}, inplace=True)

        if item_id_col != "itemId":
            self.items_df.rename(columns={item_id_col: "itemId"}, inplace=True)

        self.missing_desc_ids = list(self.items_df[self.items_df.description.isna()]['itemId'])
        self.items_df.dropna(inplace=True)

    def clean_spaces(self, str):
        return re.sub(r"\s{2,}", " ", str)
    
    def process(self):
        self.items_df.loc[:,"description"] = self.items_df.description.apply(self.clean_spaces)
        self.items_df.set_index("itemId", inplace=True)
        return self.items_df
    
class MovieDataset(ItemDataset):

    def __init__(self, description_df:DataFrame, desc_col:str, item_id_col:str) -> None:
        super(MovieDataset, self).__init__(items_df=description_df, desc_col=desc_col, item_id_col=item_id_col)
    
    def join_and_process(self, movies_df:DataFrame, item_id_col:str) -> DataFrame:

        movies_df = movies_df.copy()
        if item_id_col != "itemId":
            movies_df.rename(columns={item_id_col: "itemId"}, inplace=True)

        movies_df = movies_df[~movies_df.itemId.isin(self.missing_desc_ids)]
        movie_details_df = movies_df.set_index("itemId").join(self.items_df.set_index("itemId"), how='left')
        movie_details_df.replace('(no genres listed)', '', inplace=True)
        movie_details_df.loc[:,'genres'] = movie_details_df['genres'].str.replace("|"," ")

        return self.__create_bag_of_words(movie_details_df)
    
    
    def __create_bag_of_words(self, df:DataFrame, columns=["description", "genres"]) -> DataFrame:
        df = df.copy()
        df.loc[:,'bag_of_words'] = ''
        space_col = df['bag_of_words'].copy() + " "
        for col in columns:
            df.loc[:,"bag_of_words"] = df['bag_of_words'] + space_col + df[col]

        df.loc[:,"description"] = df["bag_of_words"].apply(self.clean_spaces)
        return df[["description", "title"]]
    
    