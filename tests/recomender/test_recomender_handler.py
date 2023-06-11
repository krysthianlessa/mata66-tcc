from recomender.recomender_handler import RecomenderHandler

import pandas as pd
import unittest

class TestRecomenderHandler(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_preprocess_ratings(self):
        
        for similarity_method in [RecomenderHandler.JACCARD_SIMILARITY, RecomenderHandler.COSINE_SIMILARITY]:
            print(f"Testing similarity: {similarity_method}")
            items_df = pd.DataFrame({"description": ["This is a test!", 
                                                    "This is not a test.", 
                                                    "Yes, maybe this is a test. However, very simple.",
                                                    "This realy is not a test.",
                                                    "This is perfectly a test!",
                                                    "I do not agree with you!",
                                                    "You can deny whatever you want.",
                                                    "Oh no! Oh no! You going around the circule on top of my head, yeah!",
                                                    "Memories follow me left and right",
                                                    "I can feel over here! I can feel over here!",
                                                    "Ooohh yeaahh! Now is time to play video games."]})
            items_df['itemId'] = items_df.index
            items_df.set_index("itemId")

            recomender = RecomenderHandler(items_df, similarity_method)

            print(recomender.similarity_df)

            self.assertTrue(len(recomender.similarity_df.index) == len(items_df.index))
            self.assertTrue(len(recomender.similarity_df.columns) == len(items_df.index))
            self.assertTrue(recomender.similarity_df.loc[3][0] <= recomender.similarity_df.loc[4][0])
            

if __name__ == '__main__':
    unittest.main()