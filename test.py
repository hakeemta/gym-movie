from gym_movie.envs.movie_env import MovieEnv

env = MovieEnv()
print( env.reset() )
print()

print( env._df_movies.head() )
print( env._df_ratings.head() )

print( env._df_users_pref.head() )