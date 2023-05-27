from recomender.preprocessor import MovieDataset, RatingDataset
from recomender.evaluation import EvaluationGenerator
from recomender.evaluation import EvaluationLoader

import pandas as pd
import unittest

class TestMovieLensRecomenderEvaluation(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.check_folder = "result/first_run"

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

        combination_pre_process_techniques = [
            (1, (False, False, False)),
            (2, (False, False, True)),
            (3, (False, True, False)),
            (4, (False, True, True)),
            (5, (True, False, False)),
            (6, (True, False, True)),
            (7, (True, True, False)),
            (8, (True, True, True)),
        ]

        evaluate_generator = EvaluationGenerator(item_df = movie_details_df, 
                                                rating_df=ratings_df).generate_from_combination(combination_pre_process_techniques)

        recs_metrics_loaded = EvaluationLoader().load_recomendations("result/movie-lens-small/first_run")
        for i in range(len(recs_metrics_loaded)):
            
            rec_metrics = evaluate_generator.recomendations[i]
            rec_metric_check = recs_metrics_loaded[f"recomendations_{i+1}"]

            self.assertTrue(rec_metrics['label'] == rec_metric_check['label'])
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

if __name__ == '__main__':
    unittest.main()