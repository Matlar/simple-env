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
SLEEP = 0.1
AGENTS = 3
WIN = 1000

class Action(IntEnum):
    left  = -1
    noop  = 0
    right = 1

class Direction(IntEnum):
    up    = 0
    right = 1
    down  = 2
    left  = 3

class CurveEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, training=True):
        super(CurveEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                shape=(SHAPE[0], SHAPE[1], 4), dtype=np.uint8)

        self._curr_episode = 0
        self._move_count = [0, 0, 0]
        self._training = training


    def step(self, action):
        self._curr_step += 1

        # Move self and count the action
        action -= 1
        self._move(action)
        self._move_count[action] += 1

        # Move other agents
        for i, agent in enumerate(self._state['agents'][1:], start=1):
            if agent is not None:
                agent_action, _ = self._agent_model.predict(np.stack(self._layers[i], axis=2))
                self._move(agent_action-1, agent=i)

        # Remove dead agents
        for i, agent in enumerate(self._state['agents']):
            if agent is None: continue
            if self._state['walls'][agent[0]] == 1:
                self._state['agents'][i] = None
            for j, other in enumerate(self._state['agents']):
                if i != j and other is not None and agent[0] == other[0]:
                    # Two heads collided
                    self._state['agents'][i] = self._state['agents'][j] = None
                    self._state['walls'][agent[0]] = 1
        episode_over = (self._state['agents'][0] is None if self._training
                        else all([agent is None for agent in self._state['agents']]))

        # Update layers
        self._layers = self._get_states()

        # Calculate reward for this step
        reward = self._get_reward()

        return np.stack(self._layers[0], axis=2), reward, episode_over, {}


    def reset(self):
        # Reload agent models
        if self._curr_episode % 1000 == 0:
            self._agent_model = PPO2.load('ppo_curve')

        # Set up new episode
        self._curr_episode += 1
        self._curr_step = 0
        self._state = self._set_state()

        # Reset layers
        self._layers = [(self._state['walls'],
                         np.zeros(SHAPE),
                         np.zeros(SHAPE),
                         np.zeros(SHAPE)) for _ in range(AGENTS)]
        self._layers = self._get_states()

        return np.stack(self._layers[0], axis=2)


    def render(self, mode='human', close=False):
        os.system('clear')
        print("Episode:", self._curr_episode)
        print("Step:", self._curr_step)
        print(f'Right: {self._move_count[Action.right]}',
              f'\tLeft: {self._move_count[Action.left]}',
              f'\tNo-op: {self._move_count[Action.noop]}')

        layers = [np.copy(layer) for layer in self._layers[0]]
        state = [factor*layer for factor, layer in zip([1, -1, 0, 3], layers)]
        state = np.sum(state, axis=0, dtype=np.uint8)
        for r in range(SHAPE[0]):
            for c in range(SHAPE[1]):
                print(CHARACTERS[state[r, c]], end=' ')
            print()

        # print('\n--- Direction ---')
        # for r in range(SHAPE[0]):
        #     for c in range(SHAPE[1]):
        #         print('#' if self._layers[0][2][r, c] else ' ', end=' ')
        #     print()
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
            position = (random.randint(2, SHAPE[0]-3),
                        random.randint(2, SHAPE[1]-3))
            direction = random.randint(0, 3)
            agents.add((position, direction))
        agents = list(agents)
        random.shuffle(agents)
        state['agents'] = agents

        return state


    def _move(self, action, agent=0):
        if self._state['agents'][agent] is None: return

        # Unpack
        (y, x), direction = self._state['agents'][agent]

        # Change direction
        direction = (direction + action) % 4

        if direction == Direction.up:
            new_position = (y-1, x)
        elif direction == Direction.down:
            new_position = (y+1, x)
        elif direction == Direction.left:
            new_position = (y, x-1)
        elif direction == Direction.right:
            new_position = (y, x+1)

        # Create a wall at old position
        self._state['walls'][y, x] = 1

        # Place agent in new position
        self._state['agents'][agent] = (new_position, direction)


    def _get_reward(self):
        return (len(self._state['agents'])
                - self._state['agents'].count(None))


    def _get_states(self):
        # layers = (wall_layer, self_layer, direction_layer, agents_layer)
        wall_layer, _, _, agents_layer = self._layers[0]

        # Update agent layer
        agents_layer[agents_layer == 1] = 0
        for agent in self._state['agents']:
            if agent is not None:
                agents_layer[agent[0]] = 1

        # Update self layer and direction layer
        for i, (_, self_layer, direction_layer, _) in enumerate(self._layers):
            self_layer[self_layer == 1] = 0
            direction_layer[direction_layer == 1] = 0
            if self._state['agents'][i] is not None:
                self_layer[self._state['agents'][i][0]] = 1

                _, direction = self._state['agents'][i]
                if direction == Direction.up:
                    for r in range(SHAPE[0]//2):
                        for c in range(SHAPE[1]):
                            direction_layer[r, c] = 1
                elif direction == Direction.down:
                    for r in range(SHAPE[0]//2, SHAPE[0]):
                        for c in range(SHAPE[1]):
                            direction_layer[r, c] = 1
                elif direction == Direction.left:
                    for r in range(SHAPE[0]):
                        for c in range(SHAPE[1]//2):
                            direction_layer[r, c] = 1
                elif direction == Direction.right:
                    for r in range(SHAPE[0]):
                        for c in range(SHAPE[1]//2, SHAPE[1]):
                            direction_layer[r, c] = 1

        return [(wall_layer, self_layer, direction_layer, agents_layer)
                for _, self_layer, direction_layer, _ in self._layers]
