import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from config import TrainConfig
from core.run_dirs import board_dir, make_run_dir, resume_from
from core.vec import build_vec_env


def build_model(spec, vec_env, cfg, resume_path):
    tb = board_dir(spec.name)
    if resume_path:
        return PPO.load(resume_path, env=vec_env, tensorboard_log=tb,
                        custom_objects=cfg.resume_overrides())
    return PPO(cfg.policy, vec_env, tensorboard_log=tb, **cfg.ppo_kwargs())


def train(spec, cfg=None):
    """Train a policy on `spec`. Imports no game -- the spec is the game."""
    cfg = cfg or TrainConfig()
    log_dir, run_dir = make_run_dir(spec.name)
    resume_path = resume_from(spec.name, cfg.resume)

    vec_env = build_vec_env(spec, cfg.num_cpu, os.path.join(run_dir, "train"))

    # Plays real episodes and saves only when they improve.
    callback = EvalCallback(
        eval_env=build_vec_env(spec, 1, os.path.join(run_dir, "eval")),
        best_model_save_path=log_dir, eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes, deterministic=False)

    model = build_model(spec, vec_env, cfg, resume_path)
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback,
                reset_num_timesteps=not resume_path)

    # EvalCallback only keeps the best-scoring checkpoint, so save the final
    # policy too -- otherwise everything learned since the last eval is lost.
    model.save(os.path.join(run_dir, "final_model"))

    # Close the emulator window and free the underlying NES process.
    vec_env.close()
    return model
