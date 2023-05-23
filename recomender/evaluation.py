from recomender.text_processor import TextProcessor
from recomender.recomender_handler import RecomenderHandler

import pandas as pd
from os import listdir as os_listdir, makedirs
from datetime import datetime

class EvaluationGenerator():
    
    def __init__(self):
        self.__set_export_folder()
    
    def __set_export_folder(self):
        self.export_folder = "result/run"+str(len(os_listdir("result")))+"-"+str(datetime.now().date())
        
        try:
            makedirs(self.export_folder)
        except:
            pass
        return self.export_folder
    
    def __get_rr_from_list(self, relevance_array):
        relevance_list_size = len(relevance_array)
        if relevance_list_size == 0:
            return 0.0
        for i in range(relevance_list_size):
            if relevance_array[i]:
                return 1 / (i + 1)
        return 0.0
    
    def __get_ap_from_list(self, relevance_array):
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

    def generate(self, movies_df, ratings_df, pre_process_tec, count=1, frac=0.75, seed=15):
    
        stopwords, lemma, stemm = pre_process_tec
        recomender = RecomenderHandler()
        movies_df = TextProcessor().pre_process(movies_df, stopwords_removal=stopwords, lemmatization=lemma, stemmization=stemm)
        cosine_sim = pd.DataFrame(recomender.generate_similarity_matrix(movies_df), 
                                  columns=movies_df.index.to_list(), 
                                  index=movies_df.index.to_list())

        user_ids = set(list(ratings_df.userId))

        for user_id in user_ids:
            profile = ratings_df[ratings_df.userId == user_id]
            
            relevance = recomender.get_recomendations(movies_df, profile, cosine_sim, frac, seed)
            
            prc_10 = relevance.count(True) / 10
            ap_10 = self.__get_ap_from_list(relevance)
            rr_10 = self.__get_rr_from_list(relevance)

            relevance_5 = relevance[:5]
            prc_5 = relevance_5.count(True) / 5
            ap_5 = self.__get_ap_from_list(relevance_5)
            rr_5 = self.__get_rr_from_list(relevance_5)

            relevance_3 = relevance[:3]
            prc_3 = relevance_3.count(True) / 3
            ap_3 = self.__get_ap_from_list(relevance_3)
            rr_3 = self.__get_rr_from_list(relevance_3)

            try:
                with open(f'{self.export_folder}/recomendations_{count}.csv', 'a') as recomendations:
                    recomendations.write(f'{user_id},"{prc_10}",{prc_5},{prc_3},{ap_10},{ap_5},{ap_3},{rr_10},{rr_5},{rr_3}\n')
            except Exception:
                print('Falha ao gravar as recomendações do id {}'.format(user_id))
                with open(f'{self.export_folder}/fails_{count}.csv', 'a') as fails:
                    fails.write(f'{user_id},"{prc_10}",{prc_5},{prc_3},{ap_10},{ap_5},{ap_3},{rr_10},{rr_5},{rr_3}\n')
            else:
                print('Recomendações do id {} gravado.'.format(user_id))