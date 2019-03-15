import os
import time
import gym
import gym_curve
import itertools
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2

def make_env():
    log_dir = f'/tmp/gym/{int(time.time())}'
    os.makedirs(log_dir, exist_ok=True)
    return Monitor(gym.make('Curve-v0'), log_dir, allow_early_resets=True)

if __name__ == '__main__':
    env = SubprocVecEnv([make_env for _ in range(4)])
    model = PPO2(CnnPolicy, env, verbose=1)
    for i in itertools.count(start=1):
        model.learn(total_timesteps=10000)
        print(f'--- Saving after {i*10000} timesteps ---')
        model.save('ppo_curve')
