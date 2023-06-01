from processor.dataset_limiter import RatingProcessor, MovieDataset

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
        movies_desc_df = pd.read_csv("data/ml-latest-small/overviews.csv")
        movie_processor = MovieDataset(movies_desc_df,
                            desc_col="overview",
                            item_id_col="movieId")

        movie_details_df = movie_processor.join_and_process(movies_df=pd.read_csv('data/ml-latest-small/movies.csv'),
                                                            item_id_col="movieId")
        
        movies_desc_df = movie_processor.items_df
        nan_values_df = movies_desc_df.isna().sum()
        self.assertTrue(nan_values_df['description'] == 0, nan_values_df['description'])
        
        movies_desc_df.set_index("itemId", inplace=True)
        for movie_id in movies_desc_df.index:
            self.assertTrue(len(movies_desc_df.loc[movie_id]['description'])*0.5 <= len(movie_details_df.loc[movie_id]['description']),
                            f"{movies_desc_df.loc[movie_id]['description']} > {movie_details_df.loc[movie_id]['description']}" )

if __name__ == '__main__':
    unittest.main()