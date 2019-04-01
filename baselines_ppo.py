import os
import time
import gym
import gym_snake
import itertools
import argparse
from policy import SmallCnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
from gym.envs.registration import register

def make_env():
    log_dir = f'monitor/{int(time.time())}'
    os.makedirs(log_dir, exist_ok=True)
    return Monitor(gym.make('Snake-v0'), log_dir, allow_early_resets=True)

def action_type(arg):
    if arg not in ('new', 'continue', 'show'):
        raise argparse.ArgumentTypeError("must be one of <new>, <continue>, <show>")
    return arg

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--action', required=True, type=action_type,
                    help='the mode to use: create a <new> model, <continue> old model, <show> model')
    ap.add_argument('-m', '--model', required=True, type=str, help='the model file')
    ap.add_argument('-o', '--out', type=str, help='where to save the model')
    ap.add_argument('-t', '--timesteps', type=int, help='how many timesteps to take before saving', default=10000)
    ap.add_argument('-i', '--iterations', type=int, help='how many iterations to run of training', default=0)
    ap.add_argument('-s', '--sleep', type=int, help='how many ms to sleep between renders', default=50)
    ap.add_argument('-r', '--reload', type=int, help='how often to reload models when showing', default=5)
    ap.add_argument('-e', '--envs', type=int, help='how many environments to run in parallel', default=4)
    args = ap.parse_args()
    args.out = args.out or args.model

    if args.action == 'show':
        register(id='Snake-nosticky-v0',
                 entry_point='gym_snake.envs:SnakeEnv',
                 kwargs={'sticky': False})
        env = gym.make('Snake-nosticky-v0')
        model = PPO2.load(args.model)

        obs = env.reset()
        resets = 0
        while True:
            env.render()
            time.sleep(args.sleep/1000)
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                obs = env.reset()
                resets += 1
                time.sleep(1)
            if resets >= args.reload:
                print(f'Reloading models after {resets} resets')
                model = PPO2.load(args.model)
                resets = 0
    else:
        env = SubprocVecEnv([make_env for _ in range(args.envs)], start_method='forkserver')
        if args.action == 'new':
            model = PPO2(SmallCnnPolicy, env, verbose=1)
            model.save(args.out)
        elif args.action == 'continue':
            model = PPO2.load(args.model, env=env)

        for i in itertools.count(start=1):
            model.learn(total_timesteps=args.timesteps)
            print(f'--- Saving after {i*args.timesteps} timesteps ---')
            model.save(args.out)
            if args.iterations > 0 and args.iterations == i:
                break
