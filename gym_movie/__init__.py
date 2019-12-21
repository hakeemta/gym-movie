from gym.envs.registration import register

register(
    id='movie-v0',
    entry_point='gym_movie.envs:MovieEnv',
)