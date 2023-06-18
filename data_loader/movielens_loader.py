from data_loader.loader import Loader 
from data_formatter.items_formatter import ItemFormatter
from data_formatter.ratings_formatter import RatingFormatter
import pandas as pd
import os

class MovieLoader(Loader):

    def __init__(self, data_source_uri="data/ml-latest-small") -> None:
        self.data_source_uri = data_source_uri

        self.item_processor = ItemFormatter(self.__load_items())
        self.items_df = self.item_processor.process()
        self.ratings_df = RatingFormatter(self.__load_ratings()).process(self.item_processor.missing_desc_ids)
        self.ratings_df = self.ratings_df.loc[self.ratings_df.itemId.isin(self.items_df.index)]


    def __load_items(self, rebuild=False):
        
        if not rebuild and os.path.isdir(f"{self.data_source_uri}/items_df.csv"):
            return pd.read_csv(f"{self.data_source_uri}/items_df.csv")
        else:
            return self.__build_items()
        

    def __load_ratings(self, rebuild=False):
        
        if rebuild and os.path.isdir(f"{self.data_source_uri}/ratings_df.csv"):
            return pd.read_csv(f"{self.data_source_uri}/ratings_df.csv")
        else:
            return self.__build_ratings()
        

    def __build_ratings(self) -> pd.DataFrame:
        return pd.read_csv(f"{self.data_source_uri}/ratings_brute.csv").rename(columns={"movieId":"itemId"})
    

    def __build_items(self) -> pd.DataFrame:

        description_df = pd.read_csv(f"{self.data_source_uri}/overviews.csv")[['movieId', 'overview']]
        movies_df = pd.read_csv(f"{self.data_source_uri}/movies.csv")[['movieId', 'genres']]
        items_df = self.__join_and_process(description_df, movies_df)
        items_df.to_csv(f"{self.data_source_uri}/items_df.csv", index=True)
        return items_df


    def __join_and_process(self, description_df:pd.DataFrame, movies_df:pd.DataFrame) -> pd.DataFrame:

        movies_df.rename(columns={"movieId": "itemId"}, inplace=True)
        description_df.rename(columns={'movieId': "itemId"}, inplace=True)
        description_df = description_df.dropna()
        movies_df = movies_df[movies_df.itemId.isin(description_df.itemId)]
        movie_details_df = movies_df.set_index("itemId").join(description_df.set_index("itemId"), how='left')
        movie_details_df.loc[:,'genres'] = movie_details_df['genres'].str.replace('(no genres listed)', '').replace("|"," ")
        movie_details_df.reset_index(inplace=True)

        print(f"{len(movie_details_df.index)} final items.")
        return self.__create_bag_of_words(movie_details_df)
    

    def __create_bag_of_words(self, df:pd.DataFrame, columns=["overview", "genres"]) -> pd.DataFrame:

        df.loc[:,'bag_of_words'] = ''
        space_col = df['bag_of_words'].copy() + " "
        for col in columns:
            df.loc[:,"bag_of_words"] = df['bag_of_words'] + space_col + df[col]

        df.loc[:,"description"] = df["bag_of_words"]
        return df[["itemId","description"]]