from gym_movie.envs.movie_env import MovieEnv

env = MovieEnv()

# print( env._df_movies.head() )
# print( env._df_ratings.head() )

# print( env._df_users_pref.head() )
# print()

for i in range(10):
    print('Iter:', i)
    obs = env.reset()
    print()
    
    for t in range(1000):
        obs, reward, done, info = env.step(0)
        
        if done:
            break

    print("Episode finished after {} timesteps".format(t+1))
    print()