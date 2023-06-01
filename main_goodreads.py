from recomender.evaluation import EvaluationGenerator
from recomender.plotter import Plotter
from data_loader.goodreads_loader import GoodReadsLoader
import numpy as np

def main():

    good_reads_loader = GoodReadsLoader()
    items_df = good_reads_loader.items_df
    ratings_df = good_reads_loader.ratings_df

    print(f"{len(items_df)} items, {len(ratings_df.userId.unique())} users and {len(ratings_df)} ratings")

    evaluate_generator = EvaluationGenerator(items_df = items_df, 
                                            ratings_df=ratings_df).generate_from_combination()
    export_folder = evaluate_generator.export(name="goodreads-children", replace_last=True)

    plotter = Plotter(show=True, export_folder=export_folder)
    plotter.plot_col(evaluate_generator.recomendations, "prc", "Average Precision")
    plotter.plot_col(evaluate_generator.recomendations, "ap", "Mean Average Precision")    
    plotter.plot_col(evaluate_generator.recomendations, "rr", "Mean Reciprocal Rank")

    print(evaluate_generator.metrics_gains_df)


if __name__ == "__main__":
    main()