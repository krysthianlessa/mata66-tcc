from data_formatter.items_formatter import ItemFormatter
from data_formatter.ratings_formatter import RatingFormatter

import pandas as pd
import os
import numpy as np

class JesterJokeLoader():

    def __init__(self, data_source_uri="data/jester-joke") -> None:
        self.data_source_uri = data_source_uri

        self.item_processor = ItemFormatter(self.__load_items())
        self.items_df = self.item_processor.process()
        self.ratings_df = RatingFormatter(self.__load_ratings()).process(self.item_processor.missing_desc_ids)
        self.ratings_df = self.ratings_df.loc[self.ratings_df.itemId.isin(self.items_df.index)]

    def __load_items(self, description_matrix_uri:str="descriptions_matrix.csv", export_name="items.csv", rebuild=False):
        
        if not rebuild and os.path.isfile(f"{self.data_source_uri}/{export_name}"):
            return pd.read_csv(f"{self.data_source_uri}/{export_name}")
        else:
            return self.__save_and_load_items(description_matrix_uri, export_name)


    def __load_ratings(self, matrix_ratings_name:str="ratings_matrix.csv", export_name="ratings.csv", rebuild=False):
        
        if not rebuild and os.path.isfile(f"{self.data_source_uri}/{export_name}"):
            return pd.read_csv(f"{self.data_source_uri}/{export_name}")
        else:
            return self.__save_and_load_ratings(matrix_ratings_name, export_name)
        

    def __save_and_load_items(self, description_matrix_uri:str="descriptions_matrix.csv", export_name="items.csv") -> pd.DataFrame:

        joke_desc_df = None
        with open(file=f"{self.data_source_uri}/{description_matrix_uri}", mode = "r", encoding="utf8") as brute_desc_file:
            joke_desc_df = pd.DataFrame({"description": brute_desc_file.readlines()})

        joke_desc_df.loc[:,"description"] = joke_desc_df["description"].replace("\n", "")
        joke_desc_df.loc[:,"itemId"] = joke_desc_df.index
        joke_desc_df.to_csv(f"{self.data_source_uri}/{export_name}", index=False)
        return joke_desc_df
    

    def __save_and_load_ratings(self, matrix_name:str="ratings_matrix.csv", export_name="ratings.csv") -> pd.DataFrame:

        user_ratings_matrix = pd.read_csv(f"{self.data_source_uri}/{matrix_name}", header=None)
        jokes_ids = np.arange(1,len(user_ratings_matrix.loc[0]))
        user_ratings = []

        for user_id in user_ratings_matrix.index:

            jokes_ratings = user_ratings_matrix.loc[user_id].values
            
            for joke_id in jokes_ids:
                if "99" == jokes_ratings[joke_id]:
                    continue
                user_ratings.append({"userId": user_id, "itemId": joke_id-1, "rating": float(jokes_ratings[joke_id])})

        user_ratings_df = pd.DataFrame(user_ratings)

        user_ratings_df.replace(99, None, inplace=True)
        user_ratings_df.dropna(inplace=True)
        user_ratings_df.to_csv(f"{self.data_source_uri}/{export_name}", index=False)

        return user_ratings_df
