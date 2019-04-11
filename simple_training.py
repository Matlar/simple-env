import os
import gym
import gym_snake
import argparse
from functools import partial
from policy import SmallCnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
from gym.envs.registration import register

def make_env(kwargs, obstacle_rate, log_dir, n):
    kwargs = eval(kwargs)
    kwargs['obstacle_rate'] = obstacle_rate
    log_dir = f'{log_dir}/env_{n}'
    os.makedirs(log_dir, exist_ok=True)
    name = 'Snake-complex-v0'
    register(id=name,
             entry_point='gym_snake.envs:SnakeEnv',
             kwargs=kwargs)
    return Monitor(gym.make(name), log_dir, allow_early_resets=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--in', type=str, help='model to start with', default='')
    ap.add_argument('-o', '--out', required=True, type=str, help='where to save the model')
    ap.add_argument('-l', '--log', required=True, type=str, help='where to save the logs')
    ap.add_argument('-t', '--timesteps', type=int, help='how many timesteps to train', default=100000)
    ap.add_argument('-c', '--complexity', type=float, help='complexity level', default=0.0)
    ap.add_argument('-e', '--envs', type=int, help='how many environments to run in parallel', default=4)
    ap.add_argument('-k', '--kwargs', type=str, help='keyword arguments', default='{}')
    args = vars(ap.parse_args())

    env = SubprocVecEnv([partial(make_env,
                                 kwargs=args['kwargs'],
                                 obstacle_rate=args['complexity'],
                                 log_dir=args['log'],
                                 n=n) for n in range(args['envs'])],
                        start_method='forkserver')
    if args['in'] == '':
        model = PPO2(SmallCnnPolicy, env, verbose=1)
    else:
        model = PPO2.load(args['in'], env=env)
    model.save(args['out'])

    print(f'--- Begin training with args: {args} ---')
    model.learn(total_timesteps=args['timesteps'])
    model.save(args['out'])
    print('--- Finished training ---')
