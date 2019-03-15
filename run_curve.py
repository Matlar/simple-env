import gym
import gym_curve
import time
from spinup.utils.test_policy import load_policy, run_policy

env = gym.make('Curve-v0')
env.reset()
reward_sum = 0
while True:
    env.render()
    _, reward, done, _ = env.step(env.action_space.sample())
    reward_sum += reward
    if done:
        print("Finished! Reward:", reward_sum)
        time.sleep(2)
        reward_sum = 0
        env.reset()
