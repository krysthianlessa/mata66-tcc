from recomender.nlp_processor import NLPProcessor
from recomender.recomender_handler import RecomenderHandler

import os
from datetime import datetime
from pandas import DataFrame, read_csv
import numpy as np
import json
import glob

class EvaluationGenerator():

    def __init__(self, items_df:DataFrame, ratings_df:DataFrame) :
        
        self.items_df = items_df
        self.rating_df = ratings_df
        self.recomendations = []

        self.labels = {
            (False, False, False): 'nenhuma técnica',
            (False, False, True): 'stemm',
            (False, True, False): 'lemma',
            (False, True, True): 'stopword',
            (True, False, False): 'stemm + lemma',
            (True, False, True): 'stopword + stemm',
            (True, True, False): 'stopword + lemma',
            (True, True, True): 'todas as técnincas'
        }

    def get_export_folder(self, name, replace_last) -> str:
        
        if not os.path.isdir(f"result/{name}"):
            os.makedirs(f"result/{name}")
        
        folder_qtd = len(os.listdir(f"result/{name}"))
        
        if replace_last and os.path.isdir(f"result/{name}/run{folder_qtd-1}-{datetime.now().date()}"):
            self.export_folder = f"result/{name}/run{folder_qtd-1}-{datetime.now().date()}"
        else:
            self.export_folder = f"result/{name}/run{folder_qtd}-{datetime.now().date()}"
            
        try:
            os.makedirs(self.export_folder)
        except:
            pass
        return self.export_folder
    
    def __reciprocal_rank(self, relevance_array) -> float:
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        for i in range(relevance_list_size):
            if relevance_array[i]:
                return 1 / (i + 1)
        return 0.0
    
    def __avg_precision_from_list(self, relevance_array) -> float:
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        hit_list = []
        relevant = 0
        for i in range(relevance_list_size):
            if relevance_array[i]:
                relevant += 1
            hit_list.append(relevant / (i + 1))
        ap = sum(hit_list)
        if ap > 0.0:
            return ap / relevance_list_size
        else:
            return 0.0
        
    def generate_from_combination(self, frac=0.75, seed=15):
        
        for techniques in self.labels.keys():
            self.generate(techniques, frac, seed)
        self.metrics_gains_df = self.get_gains_df()
        return self
    
    def generate(self, pre_process_tec:tuple, frac=0.75, seed=15):
        items_df = self.items_df.copy()
        
        stopwords, lemma, stemm = pre_process_tec
        
        items_df.loc[:,"description"] = NLPProcessor().pre_process(items_df["description"], 
                                                stopwords_removal = stopwords, 
                                                lemmatization = lemma, 
                                                stemmization = stemm)
        recomender = RecomenderHandler(items_df)
        user_ids = set(list(self.rating_df.userId))

        recomendations_i = []
        for user_id in user_ids:
            profile_rating = self.rating_df[self.rating_df.userId == user_id]
            
            #TODO ISSUE 1: non_user_itens_df = items_df[~items_df.itemId.isin(profile_rating.userId)]
            relevance = recomender.get_recomendations(items_df, profile_rating, frac, seed)
            relevance_5 = relevance[:5]
            relevance_3 = relevance[:3]
            recomendations_i.append({'user_id': user_id,
                                      "prc_10":relevance.count(True) / 10,
                                      "prc_5": relevance_5.count(True) / 5,
                                      "prc_3": relevance_3.count(True) / 3,
                                      "ap_10": self.__avg_precision_from_list(relevance),
                                      "ap_5": self.__avg_precision_from_list(relevance_5),
                                      "ap_3": self.__avg_precision_from_list(relevance_3),
                                      "rr_10": self.__reciprocal_rank(relevance),
                                      "rr_5": self.__reciprocal_rank(relevance_5),
                                      'rr_3': self.__reciprocal_rank(relevance_3)
                                    })
            feedback = f"Recomendantions to user {user_id} created."
            print(feedback, end="\r")

        self.recomendations.append({"label": self.labels[pre_process_tec], 
                                    "dataset": DataFrame(recomendations_i)
                                    })
        return self

    def export(self, name, replace_last=False) -> str:
        self.get_export_folder(name, replace_last)
        
        for i in range(len(self.recomendations)):
            self.recomendations[i]["dataset"].to_csv(f"{self.export_folder}/recomendations_{i+1}.csv", index=False)
            
        with open(f"{self.export_folder}/labels.json", "w", encoding="utf-8") as labels_file:
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
        self.metrics_gains_df.to_csv(f"{self.export_folder}/metrics_gains.csv", index=False)
        return self.export_folder
    
    def get_gains_df(self) -> DataFrame:
        return DataFrame(self.get_gains("prc") + self.get_gains("ap") + self.get_gains("rr"))
    
    def get_gains(self, metric:str) -> list:
        return self.gains(self.recomendations, metric)
    
    def gains(self, recomendations: list, metric="ap") -> list:
        
        gains_cuts_list = []
        combinations_range = np.arange(1, 8)
        keys_labels = list(self.labels.keys())

        for list_size in [3,5,10]:

            non_tecnique_ap = recomendations[0]['dataset'][f'{metric}_{list_size}'].mean()
            
            if non_tecnique_ap == 0:
                gains_cuts_list.append({"metric": metric, 
                                        "list_size": list_size, 
                                        "min_per": 100,
                                        "max_per": 100,
                                        "min_technique": "any",
                                        "max_techinque": "any"})
            else:
                gains_cut = []
                labels = []
                for combination_p in combinations_range:
                    gain = ((recomendations[combination_p]['dataset'][f'{metric}_{list_size}'].mean() / non_tecnique_ap) - 1)*100.0
                    if gain >= 0:
                        gains_cut.append(gain)
                        labels.append(self.labels[keys_labels[combination_p]])
                try:
                    min_p = np.argmin(gains_cut)
                    max_p = np.argmax(gains_cut)
                    gains_cuts_list.append({"metric": metric, 
                                            "list_size": list_size, 
                                            "min_per": gains_cut[min_p], 
                                            "max_per": gains_cut[max_p],
                                            "min_technique": labels[min_p],
                                            "max_techinque": labels[max_p]})
                except:
                    print(f"Error when calculate min and max of metric {metric} and list size {list_size}")

        return gains_cuts_list

class EvaluationLoader():

    def load_recomendations(self, result_folder:str) -> list:
    
        labels_f = open(f"{result_folder}/labels.json", "r")
        labels = json.loads(labels_f.read())
        labels_f.close()

        recomendations_uri = glob.glob(result_folder+"/*.csv")
        recomendations = []
        for i in range(len(recomendations_uri)):
            name =  os.path.split(recomendations_uri[i])[1].replace(".csv", "")
            recomendations.append({
                                "label": labels[name],
                                "dataset": read_csv(recomendations_uri[i])
                                })
        
        return recomendations