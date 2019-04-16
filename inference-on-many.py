import os
import re
import pickle
import argparse
import gym
import gym_snake
import numpy as np
from stable_baselines import PPO2
from gym.envs.registration import register

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

def test_model(model_path, env_name, episodes):
    env = gym.make(env_name)
    model = PPO2.load(model_path)
    rewards = []
    lengths = []
    for episode in range(1, episodes+1):
        if episode % 10 == 0: print(f'Episode {episode}/{episodes}')
        reward, length = inference(model, env)
        rewards.append(reward)
        lengths.append(length)
    return rewards, lengths

def get_filenames(directory):
    filenames = []
    for path, _, files in os.walk(directory):
        for name in files:
            if name.endswith('.pkl'):
                filename = os.path.join(path, name)
                matches = re.findall('timestep_(.+)_complexity', filename)
                assert(len(matches) == 1)
                filenames.append((int(matches[0]), filename))
    filenames.sort()
    return filenames

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, type=str, help='model name')
    ap.add_argument('-c', '--complexity', required=True, type=float, help='complexity')
    ap.add_argument('-e', '--episodes', type=int, help='how many episodes to run', default=100)
    ap.add_argument('-r', '--resume', type=int, help='the last finished inference timestep', default=0)
    args = vars(ap.parse_args())

    registered_name = f'Snake-inference-v0'
    register(id=registered_name,
             entry_point='gym_snake.envs:SnakeEnv',
             kwargs={
                 'sticky': False,
                 'obstacle_rate': args['complexity'],
                 'fixed_randomness': True,
                 'seed_increment': 0})

    data_dir = f'/home/oskar/kth/kex/simple-env/inference/complexity_{args["complexity"]:.2f}/{args["model"]}'
    os.makedirs(data_dir, exist_ok=True)
    for timesteps, filename in get_filenames(f'/home/oskar/kth/kex/simple-env/experiments/10M/models/{args["model"]}'):
        if timesteps <= args['resume']: continue
        print('Evaluating model:', filename)
        rewards, lengths = test_model(filename, registered_name, args['episodes'])
        save_path = f'{data_dir}/data_{timesteps}.pkl'
        with open(save_path, 'wb') as file:
            pickle.dump((rewards, lengths), file)
            print('Saved to', save_path)
        print()
