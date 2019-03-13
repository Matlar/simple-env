from spinup import ppo
import gym_simple
import tensorflow as tf
import gym

env_fn = lambda : gym.make('Simple-v0')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='out', exp_name='simple_ppo')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1000, epochs=10000, logger_kwargs=logger_kwargs)
