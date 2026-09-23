"""Vector-env assembly. Game-agnostic: the spec supplies the game half.

The wrapper chain here is part of the observation the policy learns on --
train and play MUST build it identically, so both go through this module.
"""

import gymnasium as gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecTransposeImage)


def make_env(spec, rank=0, seed=0, render_mode="rgb_array"):
    """One env, preprocessed. Returns a thunk -- SubprocVecEnv wants callables."""
    def _init():
        # render_mode='human' draws the game in a window instead of returning frames.
        env = gym.make(spec.env_id, render_mode=render_mode)
        # The spec owns the action remap + resize + grayscale + skip.
        env = spec.preprocess(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def _stack(venv, spec):
    # Frame stack and transpose are part of the learned observation,
    # not optional extras.
    return VecTransposeImage(
        VecFrameStack(venv, spec.frame_stack, channels_order="last"))


def build_vec_env(spec, num_cpu, monitor_path=None, seed=0):
    """Parallel envs for training."""
    venv = SubprocVecEnv([make_env(spec, i, seed) for i in range(num_cpu)])
    return _stack(VecMonitor(venv, monitor_path), spec)


def build_play_env(spec, seed=0):
    """A single on-screen env for watching a trained policy."""
    venv = DummyVecEnv([make_env(spec, 0, seed, render_mode="human")])
    return _stack(venv, spec)
