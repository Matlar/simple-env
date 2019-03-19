import gym
import gym_curve
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = DummyVecEnv([lambda: gym.make('Curve-v0')])
model = PPO2.load('ppo_curve', env=env)

obs = env.reset()
resets = 0
while True:
    env.render()
    action, _ = model.predict(obs)
    obs, rewards, dones, _ = env.step(action)
    if dones[0]:
        obs = env.reset()
        resets += 1
    if resets > 5:
        model = PPO2.load('ppo_curve', env=env)
        resets = 0
