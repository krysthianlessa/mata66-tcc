from processor.dataset_limiter import ItemProcessor, RatingProcessor
from data_loader.jester_joke_loader import JesterJokeLoader

import pandas as pd

import unittest

class TestJesterJokeRecomender(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
  
    def test_main(self):
        
        jester_joke_loader = JesterJokeLoader(data_source_uri="data/jester-joke")

        items_processor = ItemProcessor(jester_joke_loader.load_items())
        items_df = items_processor.process()
        
        self.assertTrue(not items_df['description'].hasnans)

        ratings_processor = RatingProcessor(ratings_df=jester_joke_loader.load_ratings())
        self.assertTrue("itemId" in ratings_processor.ratings_df.columns, ratings_processor.ratings_df.columns)
        
        ratings_df = ratings_processor.process(items_processor.missing_desc_ids)

        self.assertTrue(len(ratings_df.userId.unique()) > 0)

if __name__ == '__main__':
    unittest.main()