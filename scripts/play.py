#!/usr/bin/env python
"""Watch a trained policy: python scripts/play.py --game mario"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import play
import games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="mario", choices=games.available())
    p.add_argument("--model", help="defaults to logs/<game>/best_model.zip")
    p.add_argument("--fps", type=int, default=15)
    a = p.parse_args()

    play(games.load(a.game), a.model, a.fps)


if __name__ == "__main__":
    main()
