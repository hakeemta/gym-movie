import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pandas as pd

class MovieEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._data_dir = 'data'
        
        self._fn_movies = f'{self._data_dir }/movies.csv'
        self._fn_ratings = f'{self._data_dir }/ratings.csv'

        self._df_movies = pd.read_csv(self._fn_movies)
        self._df_ratings = pd.read_csv(self._fn_ratings)

    def step(self, action):
        NotImplementedError

    def reset(self):
        NotImplementedError

    def render(self, mode='human'):
        NotImplementedError

    def close(self):
        NotImplementedError
