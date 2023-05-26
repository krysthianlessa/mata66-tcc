from recomender.preprocessor import RatingDataset, MovieDataset
from recomender.evaluation import EvaluationGenerator
from recomender.plotter import Plotter

import pandas as pd
import json

if __name__ == "__main__":
    
    movies_desc_df = pd.read_csv("data/ml-latest-small/overviews.csv")
    movie_processor = MovieDataset(movies_desc_df,
                        desc_col="overview",
                        item_id_col="movieId")

    movie_details_df = movie_processor.join_and_process(movies_df=pd.read_csv('data/ml-latest-small/movies.csv'),
                                                        item_id_col="movieId")

    ratings_df = RatingDataset(ratings_df=pd.read_csv('data/ml-latest-small/ratings.csv'), 
                                    item_id_col="movieId", 
                                    user_id_col="userId").process(movie_processor.missing_desc_ids)

    print("quantity of unique users: "+str(len(set(ratings_df.userId.to_list()))))
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
                                            rating_df=ratings_df)
    recomendations_dfs = evaluate_generator.get_recomendations_dfs()
    
    plotter = Plotter()
    plotter.plot_col(recomendations_dfs, "prc", "Average Precision", evaluate_generator.get_export_folder())
    plotter.plot_col(recomendations_dfs, "ap", "Mean Average Precision", evaluate_generator.get_export_folder())
    plotter.plot_col(recomendations_dfs, "rr", "Mean Reciprocal Rank", evaluate_generator.get_export_folder())