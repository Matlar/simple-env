import os
import re
import pickle
import argparse
import gym
import gym_snake
import numpy as np
from stable_baselines import PPO2
from policy import SmallCnnPolicy
from gym.envs.registration import register
from stable_baselines.common.vec_env import DummyVecEnv


def inference(model, env):
    reward = 0
    length = 0
    since_last_reward = 0

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, step_reward, done, _ = env.step(action)
        reward += step_reward
        length += 1
        since_last_reward = 0 if step_reward > 5 else since_last_reward + 1
        if since_last_reward > 500:
            done = True
            reward -= 10
    return reward, length

def test_model(env_name, episodes):
    env = gym.make(env_name)
    model = PPO2(SmallCnnPolicy, DummyVecEnv([lambda: gym.make(env_name)]), verbose=1)
    rewards = []
    lengths = []
    for episode in range(1, episodes+1):
        if episode % 10 == 0: print(f'Episode {episode}/{episodes}')
        reward, length = inference(model, env)
        rewards.append(reward)
        lengths.append(length)
    return rewards, lengths

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--complexity', required=True, type=float, help='complexity')
    ap.add_argument('-e', '--episodes', type=int, help='how many episodes to run', default=100)
    args = vars(ap.parse_args())

    registered_name = f'Snake-inference-v0'
    register(id=registered_name,
             entry_point='gym_snake.envs:SnakeEnv',
             kwargs={
                 'sticky': False,
                 'obstacle_rate': args['complexity'],
                 'tail': 1,
                 'fixed_randomness': True,
                 'seed_increment': 1,
                 'map_count': 10})

    data_dir = f'/home/oskar/first-inf/complexity_{args["complexity"]}'
    os.makedirs(data_dir, exist_ok=True)
    print('Evaluating model')
    rewards, lengths = test_model(registered_name, args['episodes'])
    save_path = f'{data_dir}/data_0.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump((rewards, lengths), file)
        print('Saved to', save_path)
    print()
