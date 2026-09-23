"""Hyperparameters, in one place and game-agnostic.

learning_rate and target_kl used to appear twice -- once in the PPO
constructor and again in the resume branch's custom_objects -- where they
could silently drift apart. RESUME_OVERRIDES is derived from PPO_KWARGS
below so that can no longer happen.
"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    num_cpu: int = 10
    total_timesteps: int = 1_500_000

    policy: str = "CnnPolicy"
    device: str = "cuda"

    learning_rate: float = 2.5e-4
    n_steps: int = 512
    ent_coef: float = 0.01
    target_kl: float = 0.05

    eval_freq: int = 5000
    n_eval_episodes: int = 3

    resume: bool = True

    def ppo_kwargs(self):
        return dict(
            learning_rate=self.learning_rate, n_steps=self.n_steps,
            ent_coef=self.ent_coef, target_kl=self.target_kl,
            device=self.device, verbose=1)

    def resume_overrides(self):
        # custom_objects overrides values baked into the .zip -- without it
        # these are ignored, because the resume branch skips the constructor.
        return dict(learning_rate=self.learning_rate, target_kl=self.target_kl)


# Env-side knobs games share unless their preprocess says otherwise.
STALL_PATIENCE = 80
FRAME_SKIP = 4
RESIZE = (84, 84)
