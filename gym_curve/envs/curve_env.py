import gym
import time
import os
import random
import itertools
import numpy as np
from enum import IntEnum
from stable_baselines import PPO2

SHAPE = (36, 36)
CHARACTERS = (' ', '#', 'S', 'A')
SLEEP = 0.05
AGENTS = 2
WIN = 1000

class Action(IntEnum):
    up    = 0
    down  = 1
    left  = 2
    right = 3

class CurveEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CurveEnv, self).__init__()
        self._curr_episode = 0
        self.reset()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                shape=(SHAPE[0], SHAPE[1], 3), dtype=np.uint8)
        self._move_count = [0, 0, 0, 0]

    def step(self, action):
        self._curr_step += 1
        states = [self._get_state(agent=i) for i in range(1, AGENTS)]

        # Move self and count the action
        self._move(action)
        self._move_count[action] += 1

        # Move other agents
        for i, _ in enumerate(self._state['agents'][1:], start=1):
            # TODO: Cache get_state
            self._move(self._agent_model.predict(states[i-1])[0], agent=i)

        # Remove dead agents
        for i, agent in enumerate(self._state['agents']):
            if agent is None: continue
            if self._state['walls'][agent] == 1:
                self._state['agents'][i] = None
            for j, other in enumerate(self._state['agents']):
                if i != j and agent == other:
                    # Two heads collided
                    self._state['agents'][i] = self._state['agents'][j] = None
                    self._state['walls'][agent] = 1

        # Prepare stuff to return
        reward = self._get_reward()
        ob = self._get_state()
        episode_over = self._state['agents'][0] is None

        return ob, reward, episode_over, {}

    def reset(self):
        if self._curr_episode % 1000 == 0:
            self._agent_model = PPO2.load('ppo_curve')
        self._curr_episode += 1
        self._curr_step = 0
        self._state = self._set_state()
        return self._get_state()

    def render(self, mode='human', close=False):
        os.system('clear')
        print("Episode:", self._curr_episode)
        print("Step:", self._curr_step)
        print(f'Up: {self._move_count[Action.up]}',
              f'\tDown: {self._move_count[Action.down]}',
              f'\tLeft: {self._move_count[Action.left]}',
              f'\tRight: {self._move_count[Action.right]}')

        layers = self._get_state()
        state = [factor*layers[...,layer]
                for layer, factor in zip(range(3), [1, -1, 3])]
        state = np.sum(state, axis=0, dtype=np.uint8)
        for r in range(SHAPE[0]):
            for c in range(SHAPE[1]):
                print(CHARACTERS[state[r, c]], end=' ')
            print()
        time.sleep(SLEEP)

    def _set_state(self):
        state = dict()

        # Build walls around the world
        walls = np.zeros(SHAPE, dtype=np.uint8)
        for i in range(SHAPE[0]):
            walls[i, 0] = 1
            walls[i, SHAPE[1]-1] = 1
        for i in range(SHAPE[1]):
            walls[0, i] = 1
            walls[SHAPE[0]-1, i] = 1
        state['walls'] = walls

        # Generate agents
        agents = set()
        while len(agents) < AGENTS:
            agents.add((random.randint(1, SHAPE[0]-2),
                       random.randint(1, SHAPE[1]-2)))
        agents = list(agents)
        random.shuffle(agents)
        state['agents'] = agents

        return state

    def _move(self, action, agent=0):
        if self._state['agents'][agent] is None: return

        # Move agent
        y, x = self._state['agents'][agent]
        if action == Action.up:
            self._state['agents'][agent] = (y-1, x)
        elif action == Action.down:
            self._state['agents'][agent] = (y+1, x)
        elif action == Action.left:
            self._state['agents'][agent] = (y, x-1)
        elif action == Action.right:
            self._state['agents'][agent] = (y, x+1)

        # Create a wall
        self._state['walls'][y, x] = 1

    def _get_reward(self):
        return len(self._state['agents']) - self._state['agents'].count(None)

    def _get_state(self, agent=0):
        # layers = (walls, self, agents)
        layers = (self._state['walls'], np.zeros(SHAPE), np.zeros(SHAPE))

        # Set self layer
        if self._state['agents'][agent] is not None:
            layers[1][self._state['agents'][agent]] = 1

        # Set agent layer
        for agent in self._state['agents']:
            if agent is not None:
                layers[2][agent] = 1

        return np.stack(layers, axis=2)
