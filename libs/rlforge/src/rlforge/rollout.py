import time

import numpy as np
from stable_baselines3 import PPO

from rlforge.run_dirs import latest_model
from rlforge.vec import build_play_env


def play(spec, model_path=None, fps=15):
    """Watch a trained policy, or random actions if none exists. Ctrl-C to stop."""
    model_path = model_path or latest_model(spec.name)
    venv = build_play_env(spec)

    if model_path:
        print(f"Playing trained model: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"No trained model found in logs/{spec.name}/ -- playing RANDOM actions.\n"
              f"Train one with: uv run scripts/train.py --game {spec.name}")
        model = None

    obs = venv.reset()
    episode, steps, total = 1, 0, 0.0

    try:
        while True:
            if model:
                action, _ = model.predict(obs, deterministic=False)
            else:
                action = np.array([venv.action_space.sample()])
            obs, reward, dones, infos = venv.step(action)
            venv.envs[0].render()   # nes-py's step() does not draw
            steps += 1
            total += float(reward[0])
            time.sleep(1 / fps)

            if dones[0]:
                i = infos[0]
                parts = [f"{steps} steps", *(f"{k} {i.get(k)}" for k in spec.report_keys),
                         f"reward {total:.0f}"]
                print(f"episode {episode}: " + " | ".join(parts))
                episode, steps, total = episode + 1, 0, 0.0
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        venv.close()
