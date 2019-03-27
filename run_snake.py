import gym
import gym_snake
import time

env = gym.make('Snake-v0')
env.reset()
while True:
    env.render()
    time.sleep(0.1)
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
        env.reset()
