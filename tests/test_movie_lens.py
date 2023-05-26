from recomender.preprocessor import MovieDataset, RatingDataset
import pandas as pd

import unittest

class TestMovieLensRecomenderEvaluation(unittest.TestCase):

    def test_main(self):

        movies_desc_df = pd.read_csv("data/ml-latest-small/overviews.csv")
        movie_processor = MovieDataset(movies_desc_df,
                            desc_col="overview",
                            item_id_col="movieId")

        self.assertTrue("description" in movie_processor.items_df.columns, movie_processor.items_df.columns)
        self.assertTrue("itemId" in movie_processor.items_df.columns, movie_processor.items_df.columns)

        movie_details_df = movie_processor.join_and_process(movies_df=pd.read_csv('data/ml-latest-small/movies.csv'),
                                                            item_id_col="movieId")
        
        self.assertTrue(len(movie_details_df.index) > len(movies_desc_df.index)*0.3)
        
        init_ratings_df = pd.read_csv('data/ml-latest-small/ratings.csv')
        ratings_dataset = RatingDataset(ratings_df=init_ratings_df, 
                                item_id_col="movieId", 
                                user_id_col="userId")
        self.assertTrue("itemId" in ratings_dataset.ratings_df.columns, ratings_dataset.ratings_df.columns)
        
        ratings_df = ratings_dataset.process(movie_processor.missing_desc_ids)
        self.assertTrue(len(ratings_df.index) > len(init_ratings_df.index)*0.3)
        self.assertTrue(len(ratings_df.userId.unique()) == 465)

if __name__ == '__main__':
    unittest.main()