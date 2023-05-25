import pandas as pd
import os

class JesterJokeDataset():

    def __init__(self, data_source_uri:str) -> None:
        self.data_source_uri = data_source_uri


    def load_items(self, description_matrix_uri:str="description_matrix.csv", export_name="items.csv", rebuild=False):

        if not rebuild and os.path.isfile(f"{self.data_source_uri}/{export_name}"):
            return pd.read_csv(f"{self.data_source_uri}/{export_name}")
        else:
            return self.save_and_load_items(description_matrix_uri, export_name)
        
    def load_ratings(self, matrix_ratings_name:str="ratings_matrix.csv", export_name="ratings.csv", rebuild=False):

        if not rebuild and os.path.isfile(f"{self.data_source_uri}/{export_name}"):
            return pd.read_csv(f"{self.data_source_uri}/{export_name}")
        else:
            return self.save_and_load_ratings(matrix_ratings_name, export_name)


    def save_and_load_items(self, description_matrix_uri:str="description_matrix.csv", export_name="items.csv") -> pd.DataFrame:

        joke_desc_df = None
        with open(file=f"{self.data_source_uri}/{description_matrix_uri}", mode = "r", encoding="utf8") as brute_desc_file:
            joke_desc_df = pd.DataFrame({"description": brute_desc_file.readlines()})

        joke_desc_df.loc[:,"description"] = joke_desc_df["description"].replace("\n", "")
        joke_desc_df.loc[:,"itemId"] = joke_desc_df.index
        joke_desc_df.to_csv(f"{self.data_source_uri}/{export_name}", index=False)
        return joke_desc_df
    
    def save_and_load_ratings(self, matrix_name:str="ratings_matrix.csv", export_name="ratings.csv") -> pd.DataFrame:

        user_ratings = []
        with open(file=f"{self.data_source_uri}/{matrix_name}", mode = "r", encoding="utf8") as brute_desc_file:
            
            users_lines = brute_desc_file.readlines()
            jokes_len = len(users_lines[1].split(","))

            for user_id in range(len(users_lines)):

                jokes_ratings = users_lines[user_id].split(",")
                for joke_id in range(1,jokes_len):
                    if "99" == jokes_ratings[joke_id]:
                        continue
                    user_ratings.append({"userId": user_id, "itemId": joke_id, "rating": float(jokes_ratings[joke_id])})

        user_ratings_df = pd.DataFrame(user_ratings)
        user_ratings_df.replace(99, None, inplace=True)
        user_ratings_df.dropna(inplace=True)
        user_ratings_df.to_csv(f"{self.data_source_uri}/{export_name}", index=False)

        return user_ratings_df
