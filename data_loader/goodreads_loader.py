import pandas as pd
import numpy as np
import os

from data_loader.loader import Loader


class GoodReadsLoader(Loader):

    def __init__(self, data_source_uri="data/goodreads-datasets", rebuild=False) -> None:
        super().__init__()
        self.data_source_uri = data_source_uri

    def load_items(self, rebuild=False) -> pd.DataFrame:
        items_df = None
        if not rebuild and os.path.isfile(f"{self.data_source_uri}/items_df.csv"):
            items_df = pd.read_csv(f"{self.data_source_uri}/items_df.csv")
        else:
            items_df = self.__build_items_df()
        print(f"{len(items_df.index)} items.")
        return items_df

    def load_ratings(self, rebuild=False):
        ratings_df = None
        if not rebuild and os.path.isdir(f"{self.data_source_uri}/ratings_df.csv"):
            ratings_df = pd.read_csv(f"{self.data_source_uri}/ratings_df.csv")
        else:
            ratings_df = self.__build_ratings_df()
        print(f"{len(ratings_df.userId.unique())} users.")
        print(f"{len(ratings_df.index)} ratings.")
        return ratings_df
    
    def __build_ratings_df(self) -> pd.DataFrame:

        chunk_ratings_df = pd.read_csv(f"{self.data_source_uri}/goodreads_interactions.csv")[['user_id', 'book_id', 'rating']]
        chunk_ratings_df = chunk_ratings_df[chunk_ratings_df.rating > 0].rename(columns={"book_id": "itemId", "user_id": "userId"})
        chunk_ratings_df.to_csv(f"{self.data_source_uri}/ratings_df.csv", index=False)
        return chunk_ratings_df

    def __build_items_df(self) -> pd.DataFrame:
        print("building items_df...")

        dataset_list = []
        chunks = pd.read_json(f"{self.data_source_uri}/descriptions.json", lines=True, chunksize = 128)
        i = 0
        for chumk_items_df in chunks:
            
            chumk_items_df = chumk_items_df[['book_id','description','title','ratings_count']]
            chumk_items_df.loc[:,"ratings_count"] = chumk_items_df['ratings_count'].fillna("0").replace("","0").astype(np.int32)
            dataset_list.append(chumk_items_df[chumk_items_df['ratings_count'] > 20])

            i += 1

        chunks.close()
        dataset_df = pd.concat(dataset_list).rename(columns={"book_id": "itemId"})
        dataset_df.to_csv(f"{self.data_source_uri}/items_df.csv", index=False)
        return dataset_df