from gym_movie.envs.movie_env import MovieEnv

env = MovieEnv()
print(env)

print( env._df_movies.head() )
print( env._df_ratings.head() )