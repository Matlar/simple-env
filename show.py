import gym
import gym_simple
import time
from spinup.utils.test_policy import load_policy, run_policy

_, get_action = load_policy('out')
env = gym.make('Simple-v0')
run_policy(env, get_action)
