from recomender.evaluation import EvaluationLoader, EvaluationGenerator

import pandas as pd
import numpy as np
import unittest

class TestRatingDataset(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_gains(self):

        rec_results = EvaluationLoader().load_recomendations(result_folder="result/movie-lens-small/first_run")
        evaluation_generator = EvaluationGenerator(None, None, 0)
        gains_df = pd.DataFrame(evaluation_generator.gains(rec_results, "prc") + evaluation_generator.gains(rec_results, "ap") + evaluation_generator.gains(rec_results, "rr"))
        self.assertTrue(gains_df.max_per.min() >= 0)

if __name__ == '__main__':
    unittest.main()