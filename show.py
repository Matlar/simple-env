import gym
import gym_curve
from spinup.utils.test_policy import load_policy, run_policy

_, get_action = load_policy('out')
env = gym.make('Curve-v0')
run_policy(env, get_action)
