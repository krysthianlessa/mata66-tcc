import numpy as np
import matplotlib.pyplot as plt

class Plotter():

    def __init__(self, x = [3, 5, 10]) -> None:
        self.markers = ['.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
        self.x = x
        
    def __get_marker(self, i):
        
        len_makers = len(self.markers)
        if i < len(self.markers):
            return self.markers[i]
        else:
            return self.markers[int(i / len_makers)-1]
        
    def __plot_results(self, results:dict, title:str, export_folder:str="result/first_run"):
        
        plt.figure(figsize=(5, 4))
        plt.title(title)
        plt.grid()
        
        x2 = np.arange(len(results[list(results.keys())[0]]["x"]))
        
        for key in results.keys():
            plt.plot(x2, results[key]["y"], label=results[key]["label"], marker=results[key]["marker"])
            plt.xticks(x2, results[key]["x"])
            
        plt.legend(bbox_to_anchor=(1.04, 1))
        plt.ylabel('Valor')
        plt.xlabel('Tamanho da lista de recomendação')
        plt.savefig(f"{export_folder}/{title}.png", format="png")
        plt.show()
        plt.close()

    def plot_col(self, recomendations:dict, col:str, title:str, export_folder:str):
        
        results = {}
        keys = list(recomendations.keys())
        for i in range(len(keys)):
            results[keys[i]] = {
                "y": [recomendations[keys[i]]['dataset'][col+"_3"].mean(), recomendations[keys[i]]['dataset'][col+"_5"].mean(), recomendations[keys[i]]['dataset'][col+"_10"].mean()],
                "x": self.x,
                "label": recomendations[keys[i]]['label'],
                "marker": self.__get_marker(i)
            }
        self.__plot_results(results = results, title= title, export_folder = export_folder)