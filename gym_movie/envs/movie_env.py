import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd
import numpy as np

class MovieEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.info = {}
        self._data_dir = 'data'
        
        self._fn_movies = f'{self._data_dir }/movies.csv'
        self._fn_ratings = f'{self._data_dir }/ratings.csv'
        
        self.process_movies()
        self.process_ratings()
        
        self._movies_cols = self._df_movies.columns
        self._current_movie = None
        

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

    
    def process_context(self):
        NotImplementedError


    def sample_movie(self):
        id = self._movies_ids.pop()
        self._current_movie = self._df_movies.loc[ [id], self._movies_cols[1:] ]
        return self._current_movie.values


    def step(self, action):
        done = False
        # Sample a movie
        movie_feat = self.sample_movie()
        
        rem = len(self._movies_ids)
        # print('Size:', rem)
        if rem < 1:
            done = True

        user_id = self._current_user.index.values[0]
        movie_id = self._current_movie.index.values[0]

        rating = self._df_ratings[ (self._df_ratings.userId == user_id) & (self._df_ratings.movieId == movie_id) ].rating
        reward = rating.values[0] / 5.0
        
        obs = np.concatenate((self._current_user.values, movie_feat), axis=1)
        return obs, reward, done, self.info
        

    def refresh_movies(self, offline=True):
        # Get all the movies (for the user, if offline mode)
        if offline:
            user_id = self._current_user.index.values[0]
            idx = self._df_ratings[ self._df_ratings.userId == user_id ].movieId.values
        else:
            idx = self._df_movies.index.values
        # Shuffle the movies ids
        np.random.shuffle(idx)
        return list(idx)


    def reset(self):
        # Sample a user and refresh the movies candidates list
        self._current_user = self._df_users_pref.sample()
        self._movies_ids = self.refresh_movies()
        print('User:', self._current_user.index.values[0] )
        print('Movies:', len(self._movies_ids) )

        # Sample a movie
        movie_feat = self.sample_movie()
        return np.concatenate((self._current_user.values, movie_feat), axis=1)


    def render(self, mode='human'):
        NotImplementedError

    def close(self):
        NotImplementedError
