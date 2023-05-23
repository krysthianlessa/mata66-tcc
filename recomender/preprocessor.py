from pandas import DataFrame

class RatingDataset():
    
    def __init__(self, item_id_col="movieId"):
        self.item_id_col = item_id_col
    
    def to_5_range(self, ratings_arr, r_max, r_min):
        return (ratings_arr - r_min)*(5/(r_max-r_min))

    def preprocess_ratings(self, ratings_df, rate_factor=0.8):
        r_max = ratings_df.rating.max
        ratings_df.loc[:,"rating"] = self.to_5_range(ratings_df.rating, r_max, ratings_df.rating.min)

        ratings_df = ratings_df[~ratings_df[self.item_id_col].isin(self.missing_description_list)]
        ratings_df.reset_index(inplace=True)
        ratings_df.drop(columns=['index'], inplace=True)
        ratings_df = ratings_df[ratings_df.rating >= r_max*rate_factor]
        
        users = set(ratings_df.userId.to_list())
        users_to_remove = []

        for user in users:
            if len(ratings_df[ratings_df.userId == user]) < 20:
                users_to_remove.append(user)

        ratings_df = ratings_df[~ratings_df.userId.isin(users_to_remove)]
        return ratings_df[[self.item_id_col,"userId", "rating"]]

    
class MovieDataset():


    def __get_missing_overview_movies(self, description_df:DataFrame):
        
        missing_description_list = list(description_df[description_df.overview.isna()]['movieId'])
        description_df.dropna(inplace=True)
        return missing_description_list
    
    def join_movies_details(self, movies_df:DataFrame, description_df:DataFrame) -> DataFrame:
        missing_description_list = self.__get_missing_overview_movies(description_df)
        movies_df = movies_df[~movies_df[self.item_id_col].isin(missing_description_list)]
        movie_details_df = movies_df.set_index(self.item_id_col).join(description_df.set_index(self.item_id_col), how='left')
        movie_details_df.reset_index(inplace=True)
        movie_details_df.replace('(no genres listed)', '', inplace=True)
        movie_details_df.loc[:,'genres'] = movie_details_df['genres'].map(lambda x: x.lower().split('|'))
        movie_details_df.drop(columns=['tmdbId'], inplace=True)
        movie_details_df.set_index(self.item_id_col, inplace=True)

        return self.__create_bag_of_words(movie_details_df)
    
    def __create_bag_of_words(self, df:DataFrame, columns=["description", "genres"]) -> DataFrame:
        df = df.copy()
        df.loc[:,'bag_of_words'] = ''

        for index, row in df[columns].iterrows():
        
            bag_words = ""
            for col in columns:
                bag_words = ' '.join(row[col])
            df.loc[index,'bag_of_words'] = bag_words
        
        return df[["bag_of_words", "title"]]
    