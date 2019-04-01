import os
import time
import gym
import gym_snake
import itertools
import argparse
from functools import partial
from policy import SmallCnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
from gym.envs.registration import register

def make_env(obstacle_rate=None):
    log_dir = f'monitor/{int(time.time())}'
    os.makedirs(log_dir, exist_ok=True)

    if obstacle_rate is None:
        name = 'Snake-v0'
    else:
        register(id='Snake-complex-v0',
                entry_point='gym_snake.envs:SnakeEnv',
                kwargs={'obstacle_rate': obstacle_rate})
        name = 'Snake-complex-v0'

    return Monitor(gym.make(name), log_dir, allow_early_resets=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, type=str, help='where to save the model')
    ap.add_argument('-t', '--timesteps', type=int, help='how many timesteps to take before saving', default=100000)
    ap.add_argument('-c', '--complexity', type=int, help='how many complexity increments to do', default=9)
    ap.add_argument('-e', '--envs', type=int, help='how many environments to run in parallel', default=4)
    args = ap.parse_args()

    env = SubprocVecEnv([make_env for _ in range(args.envs)],
                        start_method='forkserver')
    model = PPO2(SmallCnnPolicy, env, verbose=1)
    print('--- Begin training with complexity 0 ---')
    model.learn(total_timesteps=args.timesteps)
    print(f'--- Saving after {args.timesteps} timesteps ---')
    model.save(args.model)

    for c in range(1, args.complexity+1):
        env = SubprocVecEnv([partial(make_env, obstacle_rate=0.01*c) for _ in range(args.envs)],
                            start_method='forkserver')
        model = PPO2.load(args.model, env=env)
        print(f'--- Begin training with complexity {0.01*c} ---')
        model.learn(total_timesteps=args.timesteps)
        print(f'--- Saving after {c*args.timesteps} timesteps ---')
        model.save(args.model)
