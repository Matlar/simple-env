import os
import time
import gym
import gym_snake
import itertools
import argparse
import random
from functools import partial
from policy import SmallCnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
from gym.envs.registration import register

def make_env(kwargs, obstacle_rate, name):
    kwargs = eval(kwargs)
    kwargs['obstacle_rate'] = obstacle_rate
    log_dir = f'monitor/{name}/{int(time.time())}_{random.randint(0, 1000000)}'
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
    ap.add_argument('-t', '--timesteps', type=int, help='how many timesteps to train for the complexity level', default=100000)
    ap.add_argument('-s', '--save', type=int, help='how often to save', default=100000)
    ap.add_argument('-c', '--complexity', type=str, help='list of complexities', default='[0.0]')
    ap.add_argument('-e', '--envs', type=int, help='how many environments to run in parallel', default=4)
    ap.add_argument('-k', '--kwargs', type=str, help='keyword arguments', default='{}')
    ap.add_argument('-n', '--name', type=str, help='name of experiment', default=f'noname_{random.randint(0, 1000000)}')
    args = ap.parse_args()

    timesteps = 0
    last_model = None
    model_name, model_ext = os.path.splitext(args.model)
    for complexity in eval(args.complexity):
        env = SubprocVecEnv([partial(make_env, kwargs=args.kwargs, obstacle_rate=complexity, name=args.name) for _ in range(args.envs)],
                            start_method='forkserver')
        if last_model is None:
            model = PPO2(SmallCnnPolicy, env, verbose=1)
            model.save(model_name + f'_timestep_0' + model_ext)
        else:
            model = PPO2.load(last_model, env=env)

        print(f'--- Begin training with complexity {complexity:.2} ---')
        remaining = args.timesteps
        while remaining > 0:
            train = min(args.save, remaining)
            model.learn(total_timesteps=train)
            remaining -= train
            timesteps += train
            print(f'--- Saving after {timesteps} timesteps ---')
            model_path = model_name + f'_timestep_{timesteps}' + model_ext
            model.save(model_path)
            last_model = model_path
