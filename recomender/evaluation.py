from recomender.nlp_processor import NLPProcessor
from recomender.recomender_handler import RecomenderHandler

import os
from datetime import datetime
from pandas import Dataframe, read_csv
import json
import glob

class EvaluationGenerator():
    
    def __init__(self, item_df, rating_df) :
        
        self.item_df = item_df
        self.rating_df = rating_df
        self.recomendations = []

    def get_export_folder(self, name, replace_last):
        
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
    
    def __reciprocal_rank(self, relevance_array):
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
        
    def generate(self, combination_techniques:list, frac=0.75, seed=15):
        
        for techniques in combination_techniques:
            self.generate(techniques, frac, seed)
        return self.recomendations
    
    def generate(self, pre_process_tec:tuple, frac=0.75, seed=15):
    
        stopwords, lemma, stemm = pre_process_tec
        items_df = self.item_df.copy()
        items_df.loc[:,"description"] = NLPProcessor().pre_process(self.item_df["description"], 
                                                stopwords_removal = stopwords, 
                                                lemmatization = lemma, 
                                                stemmization = stemm)
        recomender = RecomenderHandler(items_df)
        user_ids = set(list(self.rating_df.userId))

        recomendations_i = []
        for user_id in user_ids:
            profile_rating = self.rating_df[self.rating_df.userId == user_id]
            
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
                                      'rr_3': self.__reciprocal_rank(relevance_3)})
            
        self.recomendations.append(Dataframe(recomendations_i))
        return self
        
    def get_recomendations_dfs(self):
        
        recomendations = {}
        labels = {
            "recomendations_1": 'nenhuma técnica',
            "recomendations_2": 'stemm',
            "recomendations_3": 'lemma',
            "recomendations_4": 'stopword',
            "recomendations_5": 'stemm + lemma',
            "recomendations_6": 'stopword + stemm',
            "recomendations_7": 'stopword + lemma',
            "recomendations_8": 'todas as técnincas'
        }
        
        for i in range(len(self.recomendations)):
            recomendations[f"recomendations_{i}"] = {
                                                        "label": labels[i],
                                                        "dataset": recomendations[i]
                                                    }
        return recomendations
    
    def export(self, name, replace_last=False):
        self.get_export_folder(name, replace_last)
        
        for i in range(len(self.recomendations)):
            self.recomendations[i].to_csv(f"{self.export_folder}/reomendations_{i}.csv", index=False)
            
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
            
    def load_recomendations(self, result_folder:str) -> list:
    
        labels_f = open(f"{result_folder}/labels.json", "r")
        labels = json.loads(labels_f.read())
        labels_f.close()

        recomendations_uri = glob.glob(result_folder+"/*.csv")
        recomendations = {}
        for i in range(len(recomendations_uri)):
            name =  os.path.split(recomendations_uri[i])[1].replace(".csv", "")
            recomendations[name] = {
                                "label": labels[name],
                                "dataset": read_csv(recomendations_uri[i])
                                }
        
        return recomendations