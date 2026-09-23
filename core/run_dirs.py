import os
import time


def make_run_dir(game_name):
    """logs/<game>/ plus a fresh timestamped subfolder for this run.

    Each run gets its own monitor folder, so restarting no longer wipes the
    previous run's episode history (Monitor opens its csv in write mode).
    """
    log_dir = os.path.join("logs", game_name)
    run_dir = os.path.join(log_dir, "run_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)  # makes log_dir too
    return log_dir, run_dir


def best_model_path(game_name):
    return os.path.join("logs", game_name, "best_model.zip")


def resume_from(game_name, resume=True):
    """Path of the checkpoint to continue from, or None to start fresh."""
    path = best_model_path(game_name)
    return path if resume and os.path.exists(path) else None


def board_dir(game_name):
    return os.path.join("board", game_name)
