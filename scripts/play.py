#!/usr/bin/env python
"""Watch a trained policy: uv run scripts/play.py --game mario"""

import argparse

from rlforge import play
import games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="mario", choices=games.available())
    p.add_argument("--model", help="defaults to best_model.zip, else the newest final_model.zip, else random actions")
    p.add_argument("--fps", type=int, default=15)
    a = p.parse_args()

    play(games.load(a.game), a.model, a.fps)


if __name__ == "__main__":
    main()
