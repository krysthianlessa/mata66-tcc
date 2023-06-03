from processor.ratings_procesor import RatingProcessor
from data_loader.movielens_loader import MovieLoader

import pandas as pd
import numpy as np
import unittest

class TestRatingDataset(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_preprocess_ratings(self):
        
        rating_range = np.arange(10, -10, -0.5)        
        ratings_df = pd.DataFrame({"rating": rating_range,
                                   "itemId": np.arange(len(rating_range)),
                                   "userId": np.arange(len(rating_range))})
        rating_dataset = RatingProcessor(ratings_df, item_id_col="itemId", user_id_col="userId")
        ratings_p_df = rating_dataset.process([1,2,3,4])

        self.assertTrue(len(ratings_p_df.index) == 4, len(ratings_p_df.index))

class TestMovieDataset(unittest.TestCase):

    def test_join_and_process(self):
        movie_loader = MovieLoader("data/ml-latest-small/")
        items_df = movie_loader.__load_itens()
   
        nan_values_df = items_df.isna().sum()
        self.assertTrue(nan_values_df['description'] == 0, nan_values_df['description'])
        
        movie_overviews_df = pd.read_csv(f"data/ml-latest-small/overviews.csv")
        items_df.set_index("itemId", inplace=True)
        for movie_id in items_df.index:
            self.assertTrue(len(items_df.loc[movie_id]['description'])*0.5 <= len(movie_overviews_df.loc[movie_id]['overview']),
                            f"{items_df.loc[movie_id]['description']} > {movie_overviews_df.loc[movie_id]['overview']}" )

if __name__ == '__main__':
    unittest.main()