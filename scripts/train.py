#!/usr/bin/env python
"""Train a policy: uv run scripts/train.py --game mario"""

import argparse

from rlforge import TrainConfig, train
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
