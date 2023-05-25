from recomender.preprocessor import RatingDataset, ItemDataset
from recomender.evaluation import EvaluationGenerator
from data_loader.jester_joke_dataset import JesterJokeDataset

import json

if __name__ == "__main__":

    jester_joke_dataset = JesterJokeDataset("data/jester-joker")

    items_df = jester_joke_dataset.load_items()
    ratings_df = jester_joke_dataset.load_ratings()

    item_processor = ItemDataset(items_df, desc_col="description", item_id_col="itemId")
    items_df = item_processor.process()
    ratings_df = RatingDataset(ratings_df,
                               item_id_col="itemId", 
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

    evaluate_generator = EvaluationGenerator(item_df = items_df, rating_df=ratings_df)

    with open(f"{evaluate_generator.export_folder}/labels.json", "w", encoding="utf-8") as labels_file:
        labels_file.write(json.dumps({
        "recomendations_1": 'nenhuma técnica',
        "recomendations_2": 'stemm',
        "recomendations_3": 'lemma',
        "recomendations_4": 'stopword',
        "recomendations_5": 'stemm + lemma',
        "recomendations_6": 'stopword + stemm',
        "recomendations_7": 'stopword + lemma',
        "recomendations_8": 'todas as técnincas'
    }))
    for count, technique in combination_pre_process_techniques:
        evaluate_generator.generate( technique, count)
