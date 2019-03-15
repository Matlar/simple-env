import gym
import gym_curve
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = DummyVecEnv([lambda: gym.make('Curve-v0')])
model = PPO2.load('ppo_curve', env=env)

obs = env.reset()
while True:
    env.render()
    action, _ = model.predict(obs)
    obs, rewards, dones, _ = env.step(action)
    if dones[0]:
        env.reset()
