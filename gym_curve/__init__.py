import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Curve-v0',
    entry_point='gym_curve.envs:CurveEnv',
)
