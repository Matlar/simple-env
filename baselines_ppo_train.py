import os
import time
import gym
import gym_curve
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2

# Create unique log dir
log_dir = "/tmp/gym/{}".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)

env = gym.make('Curve-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1).learn(total_timesteps=1000000)
model.save("ppo_curve")
