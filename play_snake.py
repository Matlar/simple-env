import gym
import gym_snake
import time
from gym.envs.registration import register
from keyhandler import KeyHandler

register(id='Snake-nosticky-v0',
            entry_point='gym_snake.envs:SnakeEnv',
            kwargs={'sticky': False})
env = gym.make('Snake-nosticky-v0')
env.reset()
kh = KeyHandler()
time.sleep(2)
while True:
    env.render()
    time.sleep(0.3)
    _, _, done, _ = env.step(kh.get_action())
    if done:
        env.reset()
        time.sleep(2)
