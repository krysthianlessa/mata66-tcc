from recomender.preprocessor import RatingDataset, ItemDataset
from recomender.evaluation import EvaluationGenerator
from recomender.plotter import Plotter
from data_loader.goodreads_loader import GoodReadsLoader


def main():

    good_reads_loader = GoodReadsLoader()

    items_df = good_reads_loader.load_items()
    ratings_df = good_reads_loader.load_ratings()

    item_processor = ItemDataset(items_df, desc_col="description", item_id_col="itemId")
    items_df = item_processor.process()
    rating_processor = RatingDataset(ratings_df,
                               item_id_col="itemId", 
                               user_id_col="userId")
    ratings_df = rating_processor.process(item_processor.missing_desc_ids)

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

    evaluate_generator = EvaluationGenerator(item_df = items_df, 
                                            rating_df=ratings_df,
                                            min_user_ratings=rating_processor.min_user_ratings).generate_from_combination(combination_pre_process_techniques)
    export_folder = evaluate_generator.export(name="goodreads-datasets", replace_last=True)

    plotter = Plotter(show=True, export_folder=export_folder)
    plotter.plot_col(evaluate_generator.recomendations, "prc", "Average Precision")
    plotter.plot_col(evaluate_generator.recomendations, "ap", "Mean Average Precision")    
    plotter.plot_col(evaluate_generator.recomendations, "rr", "Mean Reciprocal Rank")

    print(evaluate_generator.metrics_gains_df)


if __name__ == "__main__":
    main()