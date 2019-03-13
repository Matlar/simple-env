import gym
import time
import os
import random
import numpy as np
from enum import IntEnum
from itertools import product

SHAPE = (3, 3)
BALLS = 1
GOALS = 1

class Tile(IntEnum):
    empty    = 0
    player   = 1
    ball     = 1 << 1
    goal     = 1 << 2
    carrying = 1 << 3

class Action(IntEnum):
    up    = 0
    down  = 1
    left  = 2
    right = 3
    ball  = 4

class SimpleEnv(gym.Env):

    def __init__(self):
        self._curr_episode = 0
        self._curr_step = 0
        self._state = self._set_state()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=15,
                shape=(self._state['shape'][0]*self._state['shape'][1],),
                dtype=np.uint8)

    def step(self, action):
        self._curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        episode_over = reward == len(self._state['balls'])
        return ob, reward, episode_over, {}

    def reset(self):
        self._curr_episode += 1
        self._curr_step = 0
        self._state = self._set_state()
        return self._get_state()

    def render(self, mode='human', close=False):
        os.system('clear')
        print("Episode:", self._curr_episode)
        print("Step:", self._curr_step)
        print("Reward:", self._get_reward())
        print()
        state = self._get_state().reshape(self._state['shape'])
        for r in range(self._state['shape'][0]):
            for c in range(self._state['shape'][1]):
                print(format(state[r,c], 'x'), end=' ')
            print()
        time.sleep(0.1)

    def _set_state(self):
        state = {'shape': SHAPE, 'carrying': -1}

        def randomize(): return (random.randrange(SHAPE[0]), random.randrange(SHAPE[1]))
        state['player'] = randomize()
        balls = set()
        while len(balls) < BALLS: balls.add(randomize())
        state['balls'] = list(balls)

        goals = set()
        while len(goals) < GOALS: goals.add(randomize())
        state['goals'] = list(goals)

        return state

    def _take_action(self, action):
        y, x = self._state['player']
        y_max, x_max = self._state['shape']
        if Action.up == action:
            if y > 0: self._state['player'] = (y-1, x)
        elif Action.down == action:
            if y < y_max-1: self._state['player'] = (y+1, x)
        elif Action.left == action:
            if x > 0: self._state['player'] = (y, x-1)
        elif Action.right == action:
            if x < x_max-1: self._state['player'] = (y, x+1)
        elif Action.ball == action:
            if self._state['carrying'] != -1:
                # Carrying a ball
                for i, ball in enumerate(self._state['balls']):
                    if self._state['player'] == ball:
                        # Swap balls
                        self._state['balls'][self._state['carrying']] = ball
                        self._state['carrying'] = i
                        self._state['balls'][i] = None
                        break
                else:
                    # Didn't stand on any ball, drop carried ball
                    self._state['balls'][self._state['carrying']] = self._state['player']
                    self._state['carrying'] = -1
            else:
                # Not carrying, try to pick up a ball
                for i, ball in enumerate(self._state['balls']):
                    if self._state['player'] == ball:
                        self._state['carrying'] = i
                        self._state['balls'][i] = None
                        break

    def _manhattan_distance(p1, p2):
        y1, x1 = p1
        y2, x2 = p2
        return abs(x2 - x1) + abs(y2 - y1)

    def _get_reward(self):
        score = 0
        for ball, goal in product(self._state['balls'], self._state['goals']):
            if ball == goal:
                score += 1
        return score

    def _get_state(self):
        state = np.zeros(self._state['shape'], dtype=np.uint8)

        state[self._state['player']] |= Tile.player
        if self._state['carrying'] != -1:
            state[self._state['player']] |= Tile.carrying

        for ball in self._state['balls']:
            if ball is not None:
                state[ball] |= Tile.ball
            else:
                state[self._state['player']] |= Tile.ball

        for goal in self._state['goals']:
            state[goal] |= Tile.goal

        return state.flatten()
