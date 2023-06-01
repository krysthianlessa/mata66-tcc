import pandas as pd

class RatingProcessor():
    
    def __init__(self, ratings_df:pd.DataFrame):
        
        self.ratings_df = ratings_df.copy()
        print(f"{len(ratings_df.userId.unique())} initial users.")
        print(f"{len(ratings_df.index)} initial ratings.")
        
    def process(self, missing_desc_ids:list):
        
        self.ratings_df = self.ratings_df[~self.ratings_df.itemId.isin(missing_desc_ids)]
        self.ratings_df.reset_index(inplace=True)
        self.ratings_df.drop(columns=['index'], inplace=True)
        self.ratings_df = self.ratings_df[self.ratings_df.rating >= self.ratings_df.rating.quantile(0.75)]
        user_counts = pd.DataFrame(self.ratings_df.userId.value_counts())
        self.min_user_ratings = max(user_counts["count"].quantile(0.24), 20)
        keep_users = user_counts[user_counts['count'] >= self.min_user_ratings].index
        self.ratings_df = self.ratings_df[self.ratings_df.userId.isin(keep_users)]

        self.ratings_df = self.ratings_df[["itemId", "userId", "rating"]]
        return self.ratings_df