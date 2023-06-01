from processor.ratings_procesor import RatingProcessor
from processor.items_processor import ItemProcessor

from recomender.evaluation import EvaluationGenerator
from recomender.plotter import Plotter
from data_loader.movielens_loader import MovieLoader

def main():

    movie_loader = MovieLoader("data/ml-latest-small/")
    item_processor = ItemProcessor(movie_loader.load_itens())
    
    items_df = item_processor.process()
    ratings_df = RatingProcessor(ratings_df=movie_loader.load_ratings(), 
                                    item_id_col="movieId", 
                                    user_id_col="userId").process(item_processor.missing_desc_ids)

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

    evaluate_generator = EvaluationGenerator(items_df = items_df, 
                                            ratings_df=ratings_df,
                                            min_user_ratings=20).generate_from_combination(combination_pre_process_techniques)
    export_folder = evaluate_generator.export(name="movie-lens-small", replace_last=True)

    plotter = Plotter(show=True, export_folder=export_folder)
    plotter.plot_col(evaluate_generator.recomendations, "prc", "Average Precision")
    plotter.plot_col(evaluate_generator.recomendations, "ap", "Mean Average Precision")
    plotter.plot_col(evaluate_generator.recomendations, "rr", "Mean Reciprocal Rank")

    print(evaluate_generator.metrics_gains_df)

if __name__ == "__main__":
    main()
