import time
from stable_baselines3 import PPO

from core.run_dirs import best_model_path
from core.vec import build_play_env


def play(spec, model_path=None, fps=15):
    """Watch a trained policy. Ctrl-C to stop."""
    venv = build_play_env(spec)
    model = PPO.load(model_path or best_model_path(spec.name))

    obs = venv.reset()
    episode, steps, total = 1, 0, 0.0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, dones, infos = venv.step(action)
            venv.envs[0].render()   # nes-py's step() does not draw
            steps += 1
            total += float(reward[0])
            time.sleep(1 / fps)

            if dones[0]:
                i = infos[0]
                extra = " | ".join(f"{k} {i.get(k)}" for k in spec.report_keys)
                print(f"episode {episode}: {steps} steps | {extra} "
                      f"| reward {total:.0f}")
                episode, steps, total = episode + 1, 0, 0.0
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        venv.close()
