import pandas as pd
import numpy as np
import os

from data_loader.loader import Loader
from data_formatter.items_formatter import ItemFormatter
from data_formatter.ratings_formatter import RatingFormatter
from data_formatter.dataset_limiter import RatingItemsLimiter

class GoodReadsLoader(Loader):

    def __init__(self, data_source_uri="data/goodreads/children", max_users=6000, top_users_quantile=0.75, bottom_users_quantile=0.25, min_ratings=30) -> None:
        super().__init__()
        
        self.data_source_uri = data_source_uri
        self.limiter = RatingItemsLimiter(max_users, top_users_quantile, bottom_users_quantile, min_ratings, data_source_uri)

        self.item_processor = ItemFormatter(self.__load_items())
        self.items_df = self.item_processor.process()
        self.ratings_df = RatingFormatter(self.__load_ratings()).process(self.item_processor.missing_desc_ids)
        self.items_df, self.ratings_df = self.limiter.limit(self.items_df, self.ratings_df)

    def __load_items(self, rebuild=False) -> pd.DataFrame:

        if not rebuild and os.path.isfile(self.limiter.items_cut_uri):
            items_df = pd.read_csv(self.limiter.items_cut_uri)

            if len(items_df) < 100:
                items_df = pd.read_csv(f"{self.data_source_uri}/items_df.csv")
 
            return items_df
        elif not rebuild and os.path.isfile(f"{self.data_source_uri}/items_df.csv"):
            return pd.read_csv(f"{self.data_source_uri}/items_df.csv")
        else:
            return self.__build_items_df()

    def __load_ratings(self, rebuild=False):

        if not rebuild and os.path.isfile(self.limiter.ratings_cut_uri):
            ratings_df = pd.read_csv(self.limiter.ratings_cut_uri)

            if len(ratings_df) < 100:
                return pd.read_csv(f"{self.data_source_uri}/ratings_df.csv")
            
            return ratings_df
        
        elif not rebuild and os.path.isfile(f"{self.data_source_uri}/ratings_df.csv"):
            return pd.read_csv(f"{self.data_source_uri}/ratings_df.csv")
        else:
            return self.__build_ratings_df()
    
    def __build_ratings_df(self) -> pd.DataFrame:
        
        print("building ratings_df...")
        chunks = pd.read_json(f"{self.data_source_uri}/ratings.json", lines=True, chunksize = 128)
        dataset_list = []
        for chunk_ratings_df in chunks:
            chunk_ratings_df = chunk_ratings_df[['user_id','book_id', 'rating']]
            dataset_list.append(chunk_ratings_df[chunk_ratings_df['rating'] > 0])
        
        chunks.close()
        dataset = pd.concat(dataset_list).rename(columns={"user_id": "userId", "book_id": "itemId"})
        dataset.to_csv(f"{self.data_source_uri}/ratings_df.csv", index=False)
        return dataset

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
        dataset_df.loc[:,"empty"] = " "
        dataset_df.loc[:,"description"] = dataset_df['description'].astype(str) + dataset_df['empty'].astype(str) + dataset_df['title'].astype(str)
        dataset_df = dataset_df[['itemId','description']]
        dataset_df.to_csv(f"{self.data_source_uri}/items_df.csv", index=False)
        return dataset_df