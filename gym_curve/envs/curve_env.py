import gym
import time
import os
import random
import itertools
import numpy as np
from enum import IntEnum

SHAPE = (10, 10)
ALLIES = 1
ENEMIES = 1
WIN = 1000

class Action(IntEnum):
    up    = 0
    down  = 1
    left  = 2
    right = 3

class CurveEnv(gym.Env):

    def __init__(self):
        self._curr_episode = 0
        self.reset()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=4,
                shape=(SHAPE[0]*SHAPE[1],), dtype=np.uint8)

    def step(self, action):
        self._curr_step += 1
        if self._state['self'] is not None: self._move_self(action)

        for i, _ in enumerate(self._state['enemies']):
            self._move_enemy(i)
        for i, ally in enumerate(self._state['allies']):
            if ally != self._state['self']:
                self._move_ally(i)

        reward = self._get_reward()
        ob = self._get_state()

        # Check if episode over
        episode_over = True
        if not self._state['enemies'] and not self._state['allies']:
            reward = 0
        elif not self._state['enemies']:
            reward = WIN * len(self._state['allies'])
        elif not self._state['allies']:
            reward = -WIN * len(self._state['enemies'])
        elif self._state['self'] is None:
            reward = -WIN * min(len(self._state['enemies']) - len(self._state['allies']), 0)
        else:
            episode_over = False
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
        state = self._get_state().reshape(SHAPE)
        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                to_print = (' ', '#', 'S', 'A', 'E')
                print(to_print[state[r, c]], end=' ')
            print()
        time.sleep(0.5)

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

        # Generate allies and enemies
        agents = set()
        while len(agents) < ALLIES + ENEMIES:
            agents.add((random.randint(1, SHAPE[0]-2),
                       random.randint(1, SHAPE[1]-2)))
        agents = list(agents)
        random.shuffle(agents)
        state['allies'] = agents[:ALLIES]
        state['enemies'] = agents[ALLIES:]

        # Set self to a random ally
        state['self'] = state['allies'][0]
        return state

    def _move_self(self, action):
        y, x = self._state['self']
        if action == Action.up:
            self._state['self'] = self._state['allies'][0] = (y-1, x)
        elif action == Action.down:
            self._state['self'] = self._state['allies'][0] = (y+1, x)
        elif action == Action.left:
            self._state['self'] = self._state['allies'][0] = (y, x-1)
        elif action == Action.right:
            self._state['self'] = self._state['allies'][0] = (y, x+1)
        self._state['walls'][y, x] = 1

    def _move(self, agent_type, index):
        walls = np.copy(self._state['walls'])
        for agent in itertools.chain(self._state['allies'], self._state['enemies']):
            walls[agent] = 1
        y, x = self._state[agent_type][index]
        available = []
        for y_to, x_to in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if walls[y_to, x_to] != 1:
                available.append((y_to, x_to))
        # Do a random (maybe available) move and place a new wall
        move = random.choice(available) if available else (y-1, x-1)
        self._state[agent_type][index] = move
        self._state['walls'][y, x] = 1

    def _move_enemy(self, enemy):
        self._move('enemies', enemy)

    def _move_ally(self, ally):
        self._move('allies', ally)

    def _get_reward(self):
        score = 0

        # Check for dead enemies
        alive_enemies = []
        for enemy in self._state['enemies']:
            if self._state['walls'][enemy] == 1:
                score += 1
            else:
                alive_enemies.append(enemy)
        self._state['enemies'] = alive_enemies

        # Check for dead allies
        alive_allies = []
        for ally in self._state['allies']:
            if self._state['walls'][ally] == 1:
                score -= 1
                if ally == self._state['self']:
                    # We died
                    self._state['self'] = None
            else:
                alive_allies.append(ally)
        self._state['allies'] = alive_allies
        return score

    def _get_state(self):
        state = np.copy(self._state['walls'])
        for ally in self._state['allies']: state[ally] = 3
        for enemy in self._state['enemies']: state[enemy] = 4
        if self._state['self'] is not None: state[self._state['self']] = 2
        return state.flatten()
