from recomender.evaluation import EvaluationGenerator
from data_loader.movielens_loader import MovieLoader
import pandas as pd
import unittest
import glob

class TestMovieLensRecomenderEvaluation(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.check_folder = "result/first_run"

    def test_main(self):

        movie_lens_loader = MovieLoader()
        movies_desc_df = pd.read_csv("data/ml-latest-small/overviews.csv")
        items_df = movie_lens_loader.items_df
        ratings_df = movie_lens_loader.ratings_df

        self.assertTrue(len(items_df.index) > len(movies_desc_df.movieId.unique())*0.3)
        self.assertTrue(len(ratings_df.userId.unique()) == 465)

        evaluate_generator = EvaluationGenerator(items_df = items_df.loc, 
                                                ratings_df=ratings_df,
                                                similarity_method="cosine").generate_from_combination()
        i = 0
        for rec_uri in glob.glob("result/movie-lens-small/first_run/*.csv"):
            
            rec_metrics = evaluate_generator.recomendations[i]
            rec_metric_check = pd.read_csv(rec_uri)

            self.assertTrue(rec_metrics['dataset']['prc_3'].mean() > rec_metric_check['dataset']['prc_3'].mean()*0.7, 
                            f"{rec_metrics['dataset']['prc_3'].mean()} <= {rec_metric_check['dataset']['prc_3'].mean()*0.7}")
            
            self.assertTrue(rec_metrics['dataset']['prc_5'].mean() > rec_metric_check['dataset']['prc_5'].mean()*0.7,
                            f"{rec_metrics['dataset']['prc_5'].mean()} <= { rec_metric_check['dataset']['prc_5'].mean()*0.7}")
            
            self.assertTrue(rec_metrics['dataset']['prc_10'].mean() > rec_metric_check['dataset']['prc_10'].mean()*0.7,
                            f"{rec_metrics['dataset']['prc_10'].mean()} <= {rec_metric_check['dataset']['prc_10'].mean()*0.7}")

            self.assertTrue(rec_metrics['dataset']['ap_3'].mean() > rec_metric_check['dataset']['ap_3'].mean()*0.7,
                            f"{rec_metrics['dataset']['ap_3'].mean()} <= {rec_metric_check['dataset']['ap_3'].mean()*0.7}")
            
            self.assertTrue(rec_metrics['dataset']['ap_5'].mean() > rec_metric_check['dataset']['ap_5'].mean()*0.7,
                            f"{rec_metrics['dataset']['ap_5'].mean()} <= {rec_metric_check['dataset']['ap_5'].mean()*0.7}")
            
            self.assertTrue(rec_metrics['dataset']['ap_10'].mean() > rec_metric_check['dataset']['ap_10'].mean()*0.7,
                            f"{rec_metrics['dataset']['ap_10'].mean()} <= rec_metric_check['dataset']['ap_10'].mean()*0.7")
            
            self.assertTrue(rec_metrics['dataset']['rr_3'].mean() > rec_metric_check['dataset']['rr_3'].mean()*0.7,
                            f"{rec_metrics['dataset']['rr_3'].mean()} <= {rec_metric_check['dataset']['rr_3'].mean()*0.7}")
            
            self.assertTrue(rec_metrics['dataset']['rr_5'].mean() > rec_metric_check['dataset']['rr_5'].mean()*0.7,
                            f"{rec_metrics['dataset']['rr_5'].mean()} <= {rec_metric_check['dataset']['rr_5'].mean()*0.7}")
            
            self.assertTrue(rec_metrics['dataset']['rr_10'].mean() > rec_metric_check['dataset']['rr_10'].mean()*0.7,
                            f"{rec_metrics['dataset']['rr_10'].mean()} <= {rec_metric_check['dataset']['rr_10'].mean()*0.7}")

            i += 1

if __name__ == '__main__':
    unittest.main()