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

def make_env(kwargs, obstacle_rate):
    kwargs = eval(kwargs)
    kwargs['obstacle_rate'] = obstacle_rate
    log_dir = f'monitor/{int(time.time())}'
    os.makedirs(log_dir, exist_ok=True)

    if obstacle_rate is None:
        name = 'Snake-v0'
    else:
        register(id='Snake-complex-v0',
                entry_point='gym_snake.envs:SnakeEnv',
                kwargs=kwargs)
        name = 'Snake-complex-v0'

    return Monitor(gym.make(name), log_dir, allow_early_resets=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, type=str, help='where to save the model')
    ap.add_argument('-t', '--timesteps', type=int, help='how many timesteps to take before saving', default=100000)
    ap.add_argument('-e', '--envs', type=int, help='how many environments to run in parallel', default=4)
    ap.add_argument('-s', '--start', type=float, help='starting complexity', default=0.0)
    ap.add_argument('-l', '--limit', type=float, help='complexity limit', default=0.1)
    ap.add_argument('-i', '--increase', type=float, help='complexity increase', default=0.01)
    ap.add_argument('-k', '--kwargs', type=str, help='keyword arguments', default='{}')
    args = ap.parse_args()

    complexity = args.start
    timesteps = 0
    while complexity + args.increase/10 < args.limit:
        env = SubprocVecEnv([partial(make_env, kwargs=args.kwargs, obstacle_rate=complexity) for _ in range(args.envs)],
                            start_method='forkserver')
        if timesteps == 0:
            model = PPO2(SmallCnnPolicy, env, verbose=1)
        else:
            model = PPO2.load(args.model, env=env)
        print(f'--- Begin training with complexity {complexity:.2} ---')
        model.learn(total_timesteps=args.timesteps)
        timesteps += args.timesteps
        print(f'--- Saving after {timesteps} timesteps ---')
        model.save(args.model)
        complexity += args.increase
