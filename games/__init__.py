"""Game registry. `load("mario")` -> games/mario/game.py's SPEC."""

from importlib import import_module


def load(name):
    """Import games/<name>/game.py and return its SPEC."""
    try:
        return import_module(f"games.{name}.game").SPEC
    except ModuleNotFoundError as e:
        raise SystemExit(f"unknown game '{name}': {e}\navailable: {', '.join(available())}")


def available():
    from pathlib import Path
    root = Path(__file__).parent
    return sorted(p.name for p in root.iterdir()
                  if (p / "game.py").exists())
