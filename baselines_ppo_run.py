import gym
import gym_curve
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym.envs.registration import register

register(id='Curve-notrain-v0',
         entry_point='gym_curve.envs:CurveEnv',
         kwargs={'training': False})

env = gym.make('Curve-notrain-v0')
model = PPO2.load('ppo_curve')

obs = env.reset()
resets = 0
while True:
    env.render()
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()
        resets += 1
    if resets > 5:
        print('--- Reloading agent model ---')
        model = PPO2.load('ppo_curve')
        resets = 0
