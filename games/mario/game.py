"""Super Mario Bros -- the whole game-specific surface, in one file.

Copy this file to add a game. Nothing in core/ needs to change.
"""

from gymnasium.wrappers import (
    GrayscaleObservation, MaxAndSkipObservation, ResizeObservation)
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros  # noqa: F401  -- registers the gym ids
from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT, SIMPLE_MOVEMENT

from rlforge import GameSpec, StallLimit
from rlforge.config import FRAME_SKIP, RESIZE, STALL_PATIENCE

from games.base_games import BaseGames


def preprocess(env):
    # JoypadSpace maps the 256 raw NES button combinations down to 5
    # (noop, right, right+A, right+B, right+A+B). Must match between
    # training and play or the loaded policy's action ids mean nothing.
    env = JoypadSpace(env, RIGHT_ONLY)
    env = MaxAndSkipObservation(env, skip=FRAME_SKIP)  # How often agents decide? (4x fewer)
    env = StallLimit(env, patience=STALL_PATIENCE)     # Outside skip, so patience counts agent steps
    env = ResizeObservation(env, RESIZE)               # How much detail? (9x fewer)
    return GrayscaleObservation(env, keep_dim=True)    # Does color matter? (3x less data)


SPEC = GameSpec(
    name="mario",
    env_id=BaseGames.SMB_L1_W1,
    preprocess=preprocess,
    progress_key="x_pos",
    report_keys=("x_pos", "flag_get"),
)
