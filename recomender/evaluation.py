from recomender.nlp_processor import NLPProcessor
from recomender.recomender_handler import RecomenderHandler

from os import listdir as os_listdir, makedirs
from datetime import datetime

class EvaluationGenerator():
    
    def __init__(self, item_df, rating_df, item_desc_col="description"):
        self.__set_export_folder()
        self.item_df = item_df
        self.rating_df = rating_df
        self.item_desc_col = item_desc_col

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

    def generate(self, pre_process_tec, count=1, frac=0.75, seed=15):
    
        stopwords, lemma, stemm = pre_process_tec

        items_df = self.item_df.copy()
        items_df.loc[:,self.item_desc_col] = NLPProcessor().pre_process(self.item_df[self.item_desc_col], 
                                                stopwords_removal = stopwords, 
                                                lemmatization = lemma, 
                                                stemmization = stemm)
        recomender = RecomenderHandler(items_df, self.item_desc_col)
        user_ids = set(list(self.rating_df.userId))

        if count == 1:
            try:
                with open(f'{self.export_folder}/recomendations_{count}.csv', 'w') as recomendations:
                    recomendations.write(f'user_id,prc_10,prc_5,prc_3,ap_10,ap_5,ap_3,rr_10,rr_5,rr_3\n')
            except:
                pass

        for user_id in user_ids:
            profile_rating = self.rating_df[self.rating_df.userId == user_id]
            
            relevance = recomender.get_recomendations(items_df, profile_rating, frac, seed)
            
            prc_10 = relevance.count(True) / 10
            ap_10 = self.__avg_precision_from_list(relevance)
            rr_10 = self.__get_rr_from_list(relevance)

            relevance_5 = relevance[:5]
            prc_5 = relevance_5.count(True) / 5
            ap_5 = self.__avg_precision_from_list(relevance_5)
            rr_5 = self.__get_rr_from_list(relevance_5)

            relevance_3 = relevance[:3]
            prc_3 = relevance_3.count(True) / 3
            ap_3 = self.__avg_precision_from_list(relevance_3)
            rr_3 = self.__get_rr_from_list(relevance_3)

            try:
                with open(f'{self.export_folder}/recomendations_{count}.csv', 'a') as recomendations:
                    recomendations.write(f'{user_id},"{prc_10}",{prc_5},{prc_3},{ap_10},{ap_5},{ap_3},{rr_10},{rr_5},{rr_3}\n')
            
                feedback = 'Recomendations of user id {} saved.'.format(user_id)
                print(feedback, end="\r")

            except Exception:
                print('Failure when saving the recomendations of user id {}'.format(user_id))