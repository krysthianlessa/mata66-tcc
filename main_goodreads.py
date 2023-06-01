from recomender.evaluation import EvaluationGenerator
from recomender.plotter import Plotter
from data_loader.goodreads_loader import GoodReadsLoader

def main():

    good_reads_loader = GoodReadsLoader()
    
    items_df = good_reads_loader.items_df
    ratings_df = good_reads_loader.ratings_df
    
    print("Cut quantities: ")
    print(f"{len(items_df.index)} items")
    print(f"{len(ratings_df.index)} ratings")
    print(f"{len(ratings_df.userId.unique())} users")

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
    export_folder = evaluate_generator.export(name="goodreads-children", replace_last=True)

    plotter = Plotter(show=True, export_folder=export_folder)
    plotter.plot_col(evaluate_generator.recomendations, "prc", "Average Precision")
    plotter.plot_col(evaluate_generator.recomendations, "ap", "Mean Average Precision")    
    plotter.plot_col(evaluate_generator.recomendations, "rr", "Mean Reciprocal Rank")

    print(evaluate_generator.metrics_gains_df)


if __name__ == "__main__":
    main()