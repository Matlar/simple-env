import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Simple-v0',
    entry_point='gym_simple.envs:SimpleEnv',
)
