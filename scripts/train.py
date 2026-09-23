#!/usr/bin/env python
"""Train a policy: python scripts/train.py --game mario"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TrainConfig
from core import train
import games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="mario", choices=games.available())
    p.add_argument("--cpus", type=int)
    p.add_argument("--timesteps", type=int)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()

    cfg = TrainConfig()
    if a.cpus:
        cfg.num_cpu = a.cpus
    if a.timesteps:
        cfg.total_timesteps = a.timesteps
    if a.no_resume:
        cfg.resume = False

    train(games.load(a.game), cfg)


if __name__ == "__main__":
    main()
