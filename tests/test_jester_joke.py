from data_loader.jester_joke_loader import JesterJokeLoader
import unittest

class TestJesterJokeRecomender(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
  
    def test_main(self):
        
        jester_joke_loader = JesterJokeLoader(data_source_uri="data/jester-joke")
        items_df = jester_joke_loader.items_df
        ratings_df = jester_joke_loader.ratings_df

        self.assertTrue(not items_df['description'].hasnans)
        self.assertTrue("itemId" in ratings_df.columns, ratings_df.columns)
        
        self.assertTrue(len(ratings_df.userId.unique()) > 0)

if __name__ == '__main__':
    unittest.main()