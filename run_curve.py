import gym
import gym_curve
import time
from spinup.utils.test_policy import load_policy, run_policy

env = gym.make('Curve-v0')
env.reset()
while True:
    env.render()
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
