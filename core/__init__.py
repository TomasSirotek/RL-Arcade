"""Game-agnostic RL engine. Written once; every game plugs into it.

Nothing here imports a game -- a GameSpec is always passed in.
"""

from core.rollout import play
from core.spec import GameSpec
from core.trainer import train
from core.wrappers import StallLimit

__all__ = ["GameSpec", "train", "play", "StallLimit"]
