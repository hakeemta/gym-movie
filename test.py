from gym_movie.envs.movie_env import MovieEnv

import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

env = MovieEnv()

obs_shape = 40
n_actions = 1

model = Sequential()
model.add(Dense(64, input_shape=(obs_shape, ) ) )
model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(n_actions))
model.add(Activation('sigmoid'))
model.compile(optimizer='sgd', loss='mean_squared_error')
print( model.summary() )

# print( env._df_movies.head() )
# print( env._df_ratings.head() )

# print( env._df_users_pref.head() )
# print()

rewards = []
for i in range(4):
    print('Iter:', i)
    obs = env.reset()
    print('Obs shape:', obs.shape)
    
    for t in range(1000):
        # DQN model here
        reward = model.predict(obs)[0]
        rewards.append(reward)

        _obs, reward, done, info = env.step(0)

        # Fit model
        model.fit(obs, np.array([reward]) )
        obs = _obs
        
        if done:
            break

    print("Episode finished after {} timesteps".format(t+1))
    print()


plt.plot(rewards)
plt.show()