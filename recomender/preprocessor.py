import re
import os
import pandas as pd

class RatingDataset():
    
    def __init__(self, ratings_df:pd.DataFrame, item_id_col:str, user_id_col:str):
        
        self.ratings_df = ratings_df.copy()
        print(f"{len(ratings_df.userId.unique())} initial users.")
        print(f"{len(ratings_df.index)} initial ratings.")
        
    def process(self, item_ids:list):
        
        self.ratings_df = self.ratings_df[self.ratings_df.itemId.isin(item_ids)]
        self.ratings_df.reset_index(inplace=True)
        self.ratings_df.drop(columns=['index'], inplace=True)
        self.ratings_df = self.ratings_df[self.ratings_df.rating >= self.ratings_df.rating.quantile(0.75)]
        user_counts = pd.DataFrame(self.ratings_df.userId.value_counts())
        self.min_user_ratings = max(user_counts["count"].quantile(0.24), 20)
        keep_users = user_counts[user_counts['count'] >= self.min_user_ratings].index
        self.ratings_df = self.ratings_df[self.ratings_df.userId.isin(keep_users)]

        print(f"{len(self.ratings_df.index)} final ratings.")
        print(f"{len(self.ratings_df.userId.unique())} final users.")

        self.ratings_df = self.ratings_df[["itemId", "userId", "rating"]]
        return self.ratings_df

class ItemDataset():

    def __init__(self, items_df: pd.DataFrame, desc_col:str, item_id_col:str) -> None:
        print(f"{len(items_df.index)} initial items.")
        self.items_df = items_df.copy()

        if desc_col != "description":
            self.items_df.rename(columns={desc_col: "description"}, inplace=True)

        if item_id_col != "itemId":
            self.items_df.rename(columns={item_id_col: "itemId"}, inplace=True)

        self.missing_desc_ids = list(self.items_df[self.items_df.description.isna()]['itemId'])
        self.items_df.dropna(inplace=True)

    def clean_spaces(self, text):
        return re.sub(r'(?<=[.,])(?=[^\s])', r' ', re.sub(r"\s{2,}", " ", re.sub('[\W+\s[.]]',' ', text))) 
    
    def process(self):
        self.items_df.loc[:,"description"] = self.items_df.description.apply(self.clean_spaces).str.replace('""', "").replace("\t"," ")
        self.items_df.set_index("itemId", inplace=True)
        print(f"{len(self.items_df.index)} final items.")
        return self.items_df