import gym
import time
import os
import random
import itertools
import numpy as np
from enum import IntEnum
from collections import deque
from itertools import product
from stable_baselines import PPO2
from maps import get_map

CHARACTERS = (' ', '#', 'S', 'X', '@', 'M')

# Configure environment
SHAPE = (12, 12)
MOVING = 0
STICKY = 0.1
AVG_LATEST = 5

class Direction(IntEnum):
    up    = 0
    right = 1
    down  = 2
    left  = 3

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, sticky=True, obstacle_rate=0.09, tail=0, level='random', fixed_randomness=False, seed_increment=1):
        super(SnakeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                shape=(SHAPE[0], SHAPE[1], 3), dtype=np.uint8)

        self._curr_episode = 0
        self._episode_reward = 0
        self._episode_rewards = deque([0]*AVG_LATEST, AVG_LATEST)

        # [up, right, down, left, failed_moves]
        self._move_count = [0, 0, 0, 0, 0]
        self._deaths_by_tail = 0
        self._deaths_by_moving = 0
        self._sticky = sticky
        self._obstacle_rate = obstacle_rate
        self._tail = tail
        self._tail_random = tail == -1
        self._map = level
        self._seed = 0
        self._fixed_randomness = fixed_randomness
        self._seed_increment = seed_increment


    def step(self, action):
        self._curr_step += 1

        # Move moving objects
        for i, _ in enumerate(self._state['moving']):
            self._move_object(i)

        # Move self and count the action
        if action is None:
            if self._state['direction'] is None:
                self._state['direction'] = 0
            action = self._state['direction']
        self._move(action)
        self._move_count[action] += 1

        # Check if we died and calculate reward
        if self._state['walls'][self._state['self']] == 1:
            episode_over = True
            reward = -10
            if self._state['self'] in self._state['tail']:
                self._deaths_by_tail += 1
            elif self._state['self'] in [pos for pos, _ in self._state['moving']]:
                self._deaths_by_moving += 1
        else:
            episode_over = False
            reward = self._get_reward()
        self._episode_reward += reward

        # Update layers
        self._layers = self._get_states()

        return np.stack(self._layers, axis=2), reward, episode_over, {}


    def reset(self):
        if self._tail_random:
            # Randomize tail length
            self._tail = random.randint(0, 10)

        # Set random seed
        if self._fixed_randomness:
            self._seed += self._seed_increment
            random.seed(self._seed)

        # Set up new episode
        self._curr_episode += 1
        self._curr_step = 0
        self._episode_rewards.append(self._episode_reward)
        self._episode_reward = 0

        # Reset state
        self._state = self._set_state()

        # Reset layers
        self._layers = (self._state['walls'],
                        np.zeros(SHAPE),
                        np.zeros(SHAPE))
        self._layers = self._get_states()

        # Return initial observations
        return np.stack(self._layers, axis=2)


    def render(self, mode='human', close=False):
        os.system('clear')
        print(f'Episode: {self._curr_episode}')
        print(f'Step: {self._curr_step}')
        print(f'Eaten fruits: {len(self._state["tail"])}')
        print(f'Accumulated reward: {self._episode_reward:.1f}')
        average = 0 if self._curr_episode < 2 else sum(self._episode_rewards) // min(AVG_LATEST, self._curr_episode-1)
        print(f'Last {AVG_LATEST} rewards: {", ".join([f"{reward:.1f}" for reward in self._episode_rewards if reward != 0])}')
        print(f'Average reward: {average:.1f}')
        print(f'Deaths by tail: {self._deaths_by_tail}')
        print(f'Deaths by moving: {self._deaths_by_moving}')
        print(f'Deaths by static: {self._curr_episode - self._deaths_by_tail - self._deaths_by_moving - 1}')
        print(f'Up: {self._move_count[Direction.up]}',
              f'\tRight: {self._move_count[Direction.right]}',
              f'\tDown: {self._move_count[Direction.down]}',
              f'\tLeft: {self._move_count[Direction.left]}',
              f'\tFailed: {self._move_count[-1]}')
        print()

        layers = [np.copy(layer) for layer in self._layers]
        for tail in self._state['tail']:
            if tail is not None:
                layers[0][tail] += 3
        for obj, _ in self._state['moving']:
            layers[0][obj] += 4
        state = [factor*layer for factor, layer in zip([1, 2, 3], layers)]
        state = np.sum(state, axis=0, dtype=np.uint8)
        for r in range(SHAPE[0]):
            for c in range(SHAPE[1]):
                print(CHARACTERS[state[r, c]], end=' ')
            if True: # True if all layers should be rendered
                print(end='          ')
                for c in range(SHAPE[1]):
                    print(CHARACTERS[int(layers[0][r, c]*1)], end=' ')
                print(end=' ')
                for c in range(SHAPE[1]):
                    if r == 0 or r == SHAPE[0]-1 or c == 0 or c == SHAPE[1]-1:
                        print(CHARACTERS[1], end=' ')
                    else:
                        print(CHARACTERS[int(layers[1][r, c]*2)], end=' ')
                print(end=' ')
                for c in range(SHAPE[1]):
                    if r == 0 or r == SHAPE[0]-1 or c == 0 or c == SHAPE[1]-1:
                        print(CHARACTERS[1], end=' ')
                    else:
                        print(CHARACTERS[int(layers[2][r, c]*3)], end=' ')
            print()


    def _set_state(self):
        state = dict()
        state['direction'] = None

        # Get map
        obstacles, state['walls'] = get_map(self._map, SHAPE, self._obstacle_rate)

        # Generate self and fruit
        objects = set()
        while len(objects) < 2 + MOVING:
            y, x = (random.randrange(SHAPE[0]),
                        random.randrange(SHAPE[1]))
            traps = [trap in obstacles for trap in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))].count(True)
            if (y, x) not in obstacles and traps < 3:
                objects.add((y, x))
        objects = list(objects)
        random.shuffle(objects)
        state['self'], state['fruit'] = objects[:2]
        state['last_distance'] = self._manhattan_distance(state['self'], state['fruit'])
        state['moving'] = []
        for obj in objects[2:]:
            state['moving'].append((obj, random.randint(0, 3)))
            state['walls'][obj] = 1

        # Generate initial tail
        state['tail'] = deque([None] * self._tail)

        return state


    def _new_position(self, position, direction):
        y, x = position
        if direction == Direction.up:
            new_position = (y-1, x)
        elif direction == Direction.down:
            new_position = (y+1, x)
        elif direction == Direction.left:
            new_position = (y, x-1)
        elif direction == Direction.right:
            new_position = (y, x+1)
        return new_position


    def _move(self, action):
        # Unpack
        (y, x) = self._state['self']

        if self._state['direction'] is None:
            self._state['direction'] = action
        elif action == (self._state['direction'] - 2) % 4:
            # Ignore action if trying to go opposite direction
            action = self._state['direction']
            self._move_count[-1] += 1
        elif self._sticky and np.random.uniform() < STICKY:
            # Sometimes, actions does not work (sticky actions)
            action = self._state['direction']

        new_position = self._new_position((y, x), action)

        # Create a wall at old position
        self._state['walls'][y, x] = 1
        self._state['tail'].append((y, x))

        # Remove a wall from tail (if we haven't eaten a fruit/growing initial tail)
        to_remove = self._state['tail'].popleft()
        if to_remove is not None:
            self._state['walls'][to_remove] = 0

        # Place agent in new position
        self._state['self'] = new_position
        self._state['direction'] = action


    def _move_object(self, index, bounced=False):
        (y, x), direction = self._state['moving'][index]
        new_position = self._new_position((y, x), direction)
        if self._state['walls'][new_position] == 0 and self._state['self'] != new_position and self._state['fruit'] != new_position:
            self._state['walls'][y, x] = 0
            self._state['walls'][new_position] = 1
            self._state['moving'][index] = new_position, direction
        elif bounced:
            # Trapped
            self._state['moving'][index] = (y, x), direction
        else:
            # Object bounced
            direction = (direction - 2) % 4
            self._state['moving'][index] = (y, x), direction
            self._move_object(index, bounced=True)


    def _manhattan_distance(self, p1, p2):
        y1, x1, y2, x2 = *p1, *p2
        return abs(y2 - y1) + abs(x2 - x1)


    def _get_reward(self):
        if self._state['self'] == self._state['fruit']:
            self._state['tail'].appendleft(None)
            positions = list(product(*map(range, SHAPE)))
            random.shuffle(positions)
            for position in positions:
                if self._state['walls'][position] == 0 and self._state['self'] != position and self._state['fruit'] != position:
                    self._state['fruit'] = position
                    self._state['last_distance'] = self._manhattan_distance(self._state['self'], self._state['fruit'])
                    return 10
            # The map is filled (will probably not happen ever)
            return 1000000
        else:
            distance = self._manhattan_distance(self._state['self'], self._state['fruit'])
            reward = 0.1/(self._state['last_distance'] - distance)
            self._state['last_distance'] = distance
            return reward


    def _get_states(self):
        # layers = (wall_layer, self_layer, fruit_layer)
        wall_layer, self_layer, fruit_layer = self._layers

        # Update self layer
        self_layer[self_layer == 1] = 0
        self_layer[self._state['self']] = 1

        # Update fruit layer
        fruit_layer[fruit_layer == 1] = 0
        fruit_layer[self._state['fruit']] = 1

        return wall_layer, self_layer, fruit_layer
