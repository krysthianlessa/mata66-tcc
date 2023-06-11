from recomender.evaluation import EvaluationGenerator, RecomenderHandler
from recomender.plotter import Plotter
from data_loader.jester_joke_loader import JesterJokeLoader
from data_loader.goodreads_loader import GoodReadsLoader
from data_loader.movielens_loader import MovieLoader

import json
import sys

def run_by_dataset_name(name):

    loader = None
    if "jester-joke" in name:
        loader = JesterJokeLoader()
    elif "goodreads" in name:
        loader = GoodReadsLoader()
    elif "ml-latest-small":
        loader = MovieLoader()

    run_fit_and_evaluation_tests(name, loader.items_df, loader.ratings_df)

def run_fit_and_evaluation_tests(name: str, items_df, ratings_df):
    
    print(f"{len(items_df)} items, {len(ratings_df.userId.unique())} users and {len(ratings_df)} ratings")

    for similarity_method in [RecomenderHandler.JACCARD_SIMILARITY]:
        print(similarity_method)
        evaluate_generator = EvaluationGenerator(items_df,ratings_df,similarity_method).generate_from_combination()
        export_folder = evaluate_generator.export(name=name, replace_last=True)

        plotter = Plotter(show=True, export_folder=export_folder)
        plotter.plot_col(evaluate_generator.recomendations, "prc", "Average Precision")
        plotter.plot_col(evaluate_generator.recomendations, "ap", "Mean Average Precision")    
        plotter.plot_col(evaluate_generator.recomendations, "rr", "Mean Reciprocal Rank")

        print(evaluate_generator.metrics_gains_df)


if __name__ == "__main__":
    
    args = sys.argv[1:]

    file = open("data/supported_datasets.json","r")
    supported_datasets_list = list(json.loads(file.read()))
    file.close()
    
    if len(args) == 0:
        run_by_dataset_name(supported_datasets_list[0])
    elif args[0] in supported_datasets_list:
        run_by_dataset_name(args[0])
    else:
        print(f"You must provide a supported dataset name to run this algorithm.\nThis dataset is one of: {supported_datasets_list}")