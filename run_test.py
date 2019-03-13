import gym
import gym_simple
import time
from spinup.utils.test_policy import load_policy, run_policy

env = gym.make('Simple-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)
