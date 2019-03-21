import os
import sys
import time
import gym
import gym_curve
import itertools
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines import PPO2
from gym.envs.registration import register

def make_env():
    log_dir = f'/tmp/gym/{int(time.time())}'
    os.makedirs(log_dir, exist_ok=True)
    return Monitor(gym.make('Curve-v0'), log_dir, allow_early_resets=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python baselines_ppo.py [new/continue/show]')
        exit(1)

    if sys.argv[1] == 'show':
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
            if resets >= 1:
                print('--- Reloading models ---')
                model = PPO2.load('ppo_curve')
                env.load_model()
                resets = 0
    elif sys.argv[1] != 'new' and sys.argv[1] != 'continue':
        print('Usage: python baselines_ppo.py [new/continue/show]')
        exit(1)
    else:
        env = SubprocVecEnv([make_env for _ in range(4)])
        if sys.argv[1] == 'new':
            model = PPO2(CnnPolicy, env, verbose=1)
        else:
            model = PPO2.load('ppo_curve', env=env)

        for i in itertools.count(start=1):
            model.learn(total_timesteps=10000)
            print(f'--- Saving after {i*10000} timesteps ---')
            model.save('ppo_curve')
