from recomender.preprocessor import ItemDataset, RatingDataset
import pandas as pd

import unittest

class TestJesterJokeRecomender(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
  
    def test_main(self):

        items_desc_df = pd.read_csv("data/jester-joke/items.csv")
        items_processor = ItemDataset(items_desc_df,
                            desc_col="description",
                            item_id_col="itemId")
        items_df = items_processor.process()
        self.assertTrue(not items_df['description'].hasnans)

        init_ratings_df = pd.read_csv('data/jester-joke/ratings.csv')
        ratings_processor = RatingDataset(ratings_df=init_ratings_df, 
                                item_id_col="movieId", 
                                user_id_col="userId")
        self.assertTrue("itemId" in ratings_processor.ratings_df.columns, ratings_processor.ratings_df.columns)
        
        ratings_df = ratings_processor.process(items_processor.missing_desc_ids)

        self.assertTrue(len(ratings_df.userId.unique()) > 0)

if __name__ == '__main__':
    unittest.main()