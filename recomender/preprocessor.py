import urllib.request
import zipfile
from os.path import split as path_split
from glob import glob
from pandas import read_csv

class PreprocessDatasets():
    
    def __init__(self,movie_overview_df):
        self.movie_overview_df = movie_overview_df
        self.list_movies_missing_overview = self.__get_missing_overview_movies(movie_overview_df)
    
    def load_datasets(url, to_path="data/"):
        
        filehandle, _ = urllib.request.urlretrieve(url)
        zip_file = zipfile.ZipFile(filehandle, 'r')
        zip_file.extractall(to_path)
        zip_file.close()
                
        databases = {}
        for database_uri in glob(f"{to_path}/*.csv"):
            databases[path_split(database_uri)[1].replace(".zip","")] = read_csv(database_uri)
        
        return databases
    
    def __get_missing_overview_movies(self, overview_df):
        
        list_movies_missing_overview = list(overview_df[overview_df.overview.isna()]['movieId'])
        overview_df.dropna(inplace=True)
        return list_movies_missing_overview

    def preprocess_ratings(self, ratings_df):
        
        ratings_df.drop(columns=['timestamp'], inplace=True)
        ratings_df = ratings_df[~ratings_df.movieId.isin(self.list_movies_missing_overview)]
        ratings_df.reset_index(inplace=True)
        ratings_df.drop(columns=['index'], inplace=True)
        ratings_df = ratings_df[ratings_df.rating >= 4.0]
        
        users = set(ratings_df.userId.to_list())
        users_to_remove = []

        for user in users:
            if len(ratings_df[ratings_df.userId == user]) < 20:
                users_to_remove.append(user)

        return ratings_df[~ratings_df.userId.isin(users_to_remove)]

    def create_movies_details(self, movies_df):
        
        movies_df = movies_df[~movies_df.movieId.isin(self.list_movies_missing_overview)]
        movie_details_df = movies_df.set_index('movieId').join(self.movie_overview_df.set_index('movieId'), how='left')
        movie_details_df.reset_index(inplace=True)
        movie_details_df.replace('(no genres listed)', '', inplace=True)
        movie_details_df['genres'] = movie_details_df['genres'].map(lambda x: x.lower().split('|'))
        movie_details_df.drop(columns=['tmdbId'], inplace=True)
        movie_details_df.set_index('movieId', inplace=True)
        
        return movie_details_df
    