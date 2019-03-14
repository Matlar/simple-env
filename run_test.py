import gym
import gym_simple
from spinup.utils.test_policy import load_policy, run_policy

env = gym.make('Simple-v0')
env.reset()
while True:
    env.render()
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
