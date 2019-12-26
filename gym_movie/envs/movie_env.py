import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd

class MovieEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.info = {}
        self._data_dir = 'data'
        
        self._fn_movies = f'{self._data_dir }/movies.csv'
        self._fn_ratings = f'{self._data_dir }/ratings.csv'

        self.process_movies()
        self.process_ratings()
        

    def process_movies(self):
        '''
        Generate movie features from hot-encoding its genres
        '''
        
        df_movies = pd.read_csv(self._fn_movies)
        # Split the genres and stack into row
        genres = df_movies['genres'].str.split('|', expand=True).stack()

        # Extract the original DF values with the stack indices
        idx = genres.index.get_level_values(0)
        df_stacked = df_movies.iloc[idx].copy()

        # Fill with the genres values
        df_stacked['genres'] = genres.values

        # Pivot the stack into columns of genres
        df_expanded = df_stacked.pivot_table(index='movieId', columns='genres', aggfunc='count', fill_value=0)['title']

        # Combine with the original DF title
        self._df_movies = pd.concat([df_movies.drop('genres', axis=1).set_index('movieId'), df_expanded], axis=1)

    
    def process_ratings(self, absolute=False):
        '''
        Rating as user-movie feature and generate user features from hot-encoding its movies genres
        '''
        self._df_ratings = pd.read_csv(self._fn_ratings)
        # Drop the timestap on ratings
        self._df_ratings.drop('timestamp', axis=1, inplace=True)

        self._df_users_genres = self._df_ratings.join(self._df_movies, on='movieId').drop(['movieId', 'rating'], 
                                                                  axis=1).groupby('userId').sum()

        self._df_users_counts = self._df_ratings.groupby('userId').count()['rating']
        
        scaler = self._df_users_genres.values.sum(axis=1, keepdims=True)
        if absolute:
            scaler = self._df_users_counts.values[..., None]

        self._df_users_pref = self._df_users_genres / scaler

    
    def step(self, action):
        obs = None
        reward = 0
        done = False
        return obs, reward, done, self.info

    def reset(self):
        obs = None
        return obs

    def render(self, mode='human'):
        NotImplementedError

    def close(self):
        NotImplementedError
