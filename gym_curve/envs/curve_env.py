import gym
import time
import os
import random
import itertools
import numpy as np
from enum import IntEnum
from collections import deque
from stable_baselines import PPO2
from maps import get_map
from keyhandler import KeyHandler

CHARACTERS = (' ', '#', 'S', 'A')

# Configure environment
SHAPE = (16, 16)
STICKY = 0.1
AVG_LATEST = 5
AGENTS = 1
MAP = 'empty'

class Direction(IntEnum):
    up    = 0
    right = 1
    down  = 2
    left  = 3

class CurveEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, training=True, play=False):
        super(CurveEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                shape=(SHAPE[0], SHAPE[1], 3), dtype=np.uint8)

        self._curr_episode = 0
        self._episode_reward = 0
        self._episode_rewards = deque([0]*AVG_LATEST, AVG_LATEST)
        self._training = training
        self._play = play
        self._agent_model = None

        # [up, right, down, left, failed_moves]
        self._move_count = [0, 0, 0, 0, 0]
        self._last_killing_move = (0, True)

        if self._play:
            self._key_handler = KeyHandler()


    def step(self, action):
        self._curr_step += 1

        # Move self and count the action
        self._move(action)
        self._move_count[action] += 1

        if self._play and AGENTS > 1:
            player_action = self._key_handler.get_action()
            self._move(player_action, agent=1)

        # Move AI agents
        if AGENTS > 1+self._play and self._agent_model is None:
            self.load_model()
        for i, agent in enumerate(self._state['agents'][1+self._play:], start=1+self._play):
            if agent is not None:
                agent_action, _ = self._agent_model.predict(np.stack(self._layers[i], axis=2))
                self._move(agent_action, agent=i)

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

        # Save last killing move
        if self._state['agents'][0] is None and self._last_killing_move[1]:
            self._last_killing_move = (action, False)

        # Update layers
        self._layers = self._get_states()

        # Calculate reward for this step
        reward = self._get_reward()
        self._episode_reward += reward

        return np.stack(self._layers[0], axis=2), reward, episode_over, {}


    def reset(self):
        # Set up new episode
        self._curr_episode += 1
        self._curr_step = 0
        self._state = self._set_state()
        self._episode_rewards.append(self._episode_reward)
        self._episode_reward = 0

        # Reset layers
        self._layers = [(self._state['walls'],
                         np.zeros(SHAPE),
                         np.zeros(SHAPE)) for _ in range(AGENTS)]
        self._layers = self._get_states()

        # Load new model
        if self._training and AGENTS > 2 and self._curr_episode % 1000 == 0:
            self.load_model()

        # Reset last killing move
        self._last_killing_move = (self._last_killing_move[0], True)

        # Return initial observations
        return np.stack(self._layers[0], axis=2)


    def render(self, mode='human', close=False):
        os.system('clear')
        print(f'Episode: {self._curr_episode}')
        print(f'Step: {self._curr_step}')
        print(f'Accumulated reward: {self._episode_reward}')
        print(f'Up: {self._move_count[Direction.up]}',
              f'\tRight: {self._move_count[Direction.right]}',
              f'\tDown: {self._move_count[Direction.down]}',
              f'\tLeft: {self._move_count[Direction.left]}',
              f'\tFailed: {self._move_count[-1]}')
        print(f'Last killing move: {Direction(self._last_killing_move[0]).name}')
        average = 0 if self._curr_episode < 2 else sum(self._episode_rewards) // min(AVG_LATEST, self._curr_episode-1)
        print(f'Last {AVG_LATEST} rewards: {list(self._episode_rewards)}, average: {average}, filled: {int(average*100/((SHAPE[0]-1)*(SHAPE[1]-1)))}%')

        layers = [np.copy(layer) for layer in self._layers[0]]
        state = [factor*layer for factor, layer in zip([1, -1, 3], layers)]
        state = np.sum(state, axis=0, dtype=np.uint8)
        for r in range(SHAPE[0]):
            for c in range(SHAPE[1]):
                print(CHARACTERS[state[r, c]], end=' ')
            print()


    def load_model(self):
        self._agent_model = PPO2.load('ppo_curve')


    def _set_state(self):
        state = dict()

        # Get map
        obstacles, state['walls'] = get_map(MAP, SHAPE)

        # Generate agents
        agents = set()
        while len(agents) < AGENTS:
            y, x = (random.randrange(SHAPE[0]),
                        random.randrange(SHAPE[1]))
            if (y, x) not in obstacles and not all(
                    [trap in obstacles for trap in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))]):
                agents.add(((y, x), None))
        agents = list(agents)
        random.shuffle(agents)
        state['agents'] = agents

        return state


    def _move(self, action, agent=0):
        if self._state['agents'][agent] is None: return

        # Unpack
        (y, x), direction = self._state['agents'][agent]

        if direction is None and action is None:
            action = random.randint(0, 3)
            direction = action
        elif direction is None:
            direction = action
        elif action is None or action == (direction - 2) % 4:
            # Ignore action if trying to go opposite direction (or no action)
            action = direction
            self._move_count[-1] += 1
        elif agent == 0 and np.random.uniform() < STICKY:
            # Sometimes, actions does not work (sticky actions)
            action = direction

        if action == Direction.up:
            new_position = (y-1, x)
        elif action == Direction.down:
            new_position = (y+1, x)
        elif action == Direction.left:
            new_position = (y, x-1)
        elif action == Direction.right:
            new_position = (y, x+1)

        # Create a wall at old position
        self._state['walls'][y, x] = 1

        # Place agent in new position
        self._state['agents'][agent] = (new_position, action)


    def _get_reward(self):
        return (len(self._state['agents'])
                - self._state['agents'].count(None))


    def _get_states(self):
        # layers = (wall_layer, self_layer, agents_layer)
        wall_layer, _, agents_layer = self._layers[0]

        # Update agent layer
        agents_layer[agents_layer == 1] = 0
        for agent in self._state['agents']:
            if agent is not None:
                agents_layer[agent[0]] = 1

        # Update self layer
        for i, (_, self_layer, _) in enumerate(self._layers):
            self_layer[self_layer == 1] = 0
            if self._state['agents'][i] is not None:
                self_layer[self._state['agents'][i][0]] = 1

        return [(wall_layer, self_layer, agents_layer)
                for _, self_layer, _ in self._layers]
