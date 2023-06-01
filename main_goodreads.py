from recomender.preprocessor import RatingDataset, ItemDataset
from recomender.evaluation import EvaluationGenerator
from recomender.plotter import Plotter
from data_loader.goodreads_loader import GoodReadsLoader

import pandas as pd
import os

def main():

    items_df = None
    ratings_df = None
    data_source_uri = "data/goodreads-datasets/children"
    
    if os.path.isfile(f"{data_source_uri}/ratings_cut_df.csv"):
        items_df = pd.read_csv(f"{data_source_uri}/items_cut_df.csv")
        ratings_df = pd.read_csv(f"{data_source_uri}/ratings_cut_df.csv")

        if len(ratings_df) < 100 or len(items_df) < 100:
            items_df = pd.read_csv(f"{data_source_uri}/items_df.csv")
            ratings_df = pd.read_csv(f"{data_source_uri}/ratings_df.csv")
            
        items_df.set_index("itemId", inplace=True)
    else:
        good_reads_loader = GoodReadsLoader()

        items_df = good_reads_loader.load_items()
        ratings_df = good_reads_loader.load_ratings()

        item_processor = ItemDataset(items_df)
        items_df = item_processor.process()
        
        rating_processor = RatingDataset(ratings_df)
        ratings_df = rating_processor.process(item_processor.missing_desc_ids)

    users_qtd = 500

    print(f"Reducing to have up to {users_qtd} users")
    
    users_ratings_counts_df = pd.DataFrame(ratings_df['userId'].value_counts())
    users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] < users_ratings_counts_df['count'].quantile(0.85)]
    users_ratings_counts_df = users_ratings_counts_df[users_ratings_counts_df['count'] > 20].iloc[0:users_qtd]
    ratings_df = ratings_df[ratings_df['userId'].isin(users_ratings_counts_df.index)]
    items_df = items_df[items_df.index.isin(ratings_df.itemId.unique())]
    
    items_ratins_counts_df = pd.DataFrame(ratings_df['itemId'].value_counts())
    items_ratins_counts_df = items_ratins_counts_df[items_ratins_counts_df['count'] < items_ratins_counts_df['count'].quantile(0.75)]
    items_ratins_counts_df = items_ratins_counts_df[items_ratins_counts_df['count'] > items_ratins_counts_df['count'].quantile(0.25)]
    ratings_df = ratings_df[ratings_df['itemId'].isin(items_ratins_counts_df.index)]
    items_df = items_df[items_df.index.isin(items_ratins_counts_df.index)]
    
    ratings_df.to_csv(f"{data_source_uri}/ratings_cut_df.csv", index=False)
    items_df.to_csv(f"{data_source_uri}/items_cut_df.csv")

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