import gym
import gym_curve
import time
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def evaluate(model, num_steps=10000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        if i % 1000 == 0: print(f'    Step {i}/{num_steps}')
        action, _ = model.predict(obs)
        _, rewards, dones, _ = env.step(action)
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            env.reset()
            episode_rewards.append(0.0)

    return round(np.mean(episode_rewards), 1)

print('Evaluating...')
env = DummyVecEnv([lambda: gym.make('Curve-v0')])
model = PPO2.load('ppo_curve', env=env)
mean = evaluate(model)

print("Mean reward:", mean)
print("Running", end='', flush=True)
for _ in range(10):
    time.sleep(0.5)
    print('.', end='', flush=True)
print()

obs = env.reset()
while True:
    env.render()
    action, _ = model.predict(obs)
    _, rewards, dones, _ = env.step(action)
    if dones[0]:
        env.reset()
