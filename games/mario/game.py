"""Super Mario Bros -- the whole game-specific surface, in one file.

Copy this file to add a game. Nothing in core/ needs to change.
"""

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, COMPLEX_MOVEMENT, SIMPLE_MOVEMENT
from rlforge import GameSpec, StallLimit, pixel_pipeline
from rlforge.config import STALL_PATIENCE
from games.base_games import BaseGames
from games.vizdoom.play_option import PlayOptions


def preprocess(env):
    env = JoypadSpace(env, RIGHT_ONLY)             # 256 NES button combos -> 5 actions
    env = pixel_pipeline(env)                      # skip 4 frames, resize to 84x84, grayscale
    return StallLimit(env, patience=STALL_PATIENCE)  # outside skip, so patience counts agent steps

SPEC = GameSpec(
    name="mario",
    env_id=BaseGames.SMB_L1_W1,
    preprocess=preprocess,
    progress_key="x_pos",
    report_keys=("x_pos", "flag_get"),
)
