from recomender.preprocessor import RatingDataset

import pandas as pd
import numpy as np
import unittest

class TestRatingDataset(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_to_5_range(self):

        rating_range = np.arange(-14,120)
        ratings_df = pd.DataFrame({"rating": rating_range,
                                   "itemId": np.arange(len(rating_range)),
                                   "userId": np.arange(len(rating_range))})
        rating_dataset = RatingDataset(ratings_df, item_id_col="itemId", user_id_col="userId")

        rating_5_range = rating_dataset.to_5_range(rating_range, rating_range.max(), rating_range.min())

        self.assertTrue(rating_5_range.max() == 5.0, rating_5_range.max())
        self.assertTrue(rating_5_range.min() == 0.0, rating_5_range.min())

    def test_preprocess_ratings(self):
        
        rating_range = np.arange(10, -10, -0.5)        
        ratings_df = pd.DataFrame({"rating": rating_range,
                                   "itemId": np.arange(len(rating_range)),
                                   "userId": np.arange(len(rating_range))})
        rating_dataset = RatingDataset(ratings_df, item_id_col="itemId", user_id_col="userId")
        ratings_p_df = rating_dataset.process([1,2,3,4])

        self.assertTrue(len(ratings_p_df.index) == 4, len(ratings_p_df.index))

if __name__ == '__main__':
    unittest.main()