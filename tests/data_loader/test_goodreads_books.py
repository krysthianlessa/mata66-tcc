from data_loader.goodreads_loader import GoodReadsLoader

import unittest
import numpy as np

class TestGoodReadsBooks(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_load_items_and_ratings(self):

        good_books_loader = GoodReadsLoader()
        items_loaded_df = good_books_loader.__load_itens()
        ratings_loaded_df = good_books_loader.__load_ratings()

        self.assertTrue(len(items_loaded_df.index) > 1000)
        self.assertTrue(len(ratings_loaded_df.index) > 1000)
        self.assertTrue("itemId" in items_loaded_df.columns)
        self.assertTrue("itemId" in ratings_loaded_df.columns)
        print("completed, checking if itemId matchs")
        self.assertTrue(len(np.intersect1d(ratings_loaded_df.itemId, items_loaded_df.itemId)) > 0)
        
if __name__ == '__main__':
    unittest.main()