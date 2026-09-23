![Image](docs/images/banner.jpg)

# RL-Arcade

Reinforcement learning agents trained to play classic arcade and platformer games (Super Mario, Sonic, and more) using deep RL algorithms. Python 3.14, Gymnasium-native, PPO via [stable-baselines3](https://stable-baselines3.readthedocs.io/).

I wanted to know a bit more about Reinforcement learning agents so I went to YT and most of the videos were super outdated and not clear for me to undestand plus
all the libraries were outdated as well so I decided to build it myself.

- Made it with proper architecture so that the games are plug&play.
- The training engine is its own library: [rl-forge](https://github.com/TomasSirotek/rl-forge).
- Currently ships [Super Mario Bros (NES)](docs/mario.md).
- In-progress [ViZDoom](https://vizdoom.farama.org/).

Hopefully this repo will help someone in same situation as Im/was. Thx. After clock project - SirTomas.

- Working on full YouTube video tutorial (Soon)
- If u do leave a star ⭐
- Issues and improvements are more than welcome.

---



## 1. Install

The only thing you install yourself is [uv](https://docs.astral.sh/uv/). It
fetches Python 3.14 and every dependency for you. A CUDA GPU makes it faast.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # skip if you have uv

git clone --recursive https://github.com/TomasSirotek/RL-Arcade.git rl-arcade
cd rl-arcade
uv sync
```

`--recursive` matters: the training engine is a separate library,
[rlforge](https://github.com/TomasSirotek/rl-forge), included as a git submodule
in `libs/rlforge/`. If you already cloned without it, run
`git submodule update --init`.

`uv sync` creates `.venv/` with the exact versions pinned in `uv.lock`, so every
machine gets the same setup. There's no venv to activate: `uv run` always uses
this project's environment.

Check it worked:

```bash
uv run python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

If that prints `False` you'll train on CPU — it works, just slowly. Set
`device="cpu"` in `TrainConfig` ([libs/rlforge/src/rlforge/config.py](libs/rlforge/src/rlforge/config.py)) to silence the warning. (On Linux the
PyPI torch wheel already includes CUDA. On Windows it's the CPU build.)

The Mario ROM ships legally inside `gym-super-mario-bros`. Nothing to download.

## 2. Run it

```bash
# 1. Train. Writes to logs/mario/ and board/mario/.
uv run scripts/train.py

# 2. Watch it play.
uv run scripts/play.py
```

`play.py` picks the best model it can find:

1. `logs/<game>/best_model.zip` — saved by the evaluator whenever the score improves
2. otherwise the newest `logs/<game>/run_*/final_model.zip`
3. otherwise **random actions**, so you can see the game before training anything

It prints which one it's using when it starts.

Watch it learn, in another terminal:

```bash
uv run tensorboard --logdir board/
```

The number to watch is `rollout/ep_rew_mean`. It should climb. If it's flat
after ~200k steps, something's wrong.

### Useful flags

```bash
uv run scripts/train.py --game mario       # pick a game (default: mario)
uv run scripts/train.py --cpus 4           # fewer parallel envs (less RAM)
uv run scripts/train.py --timesteps 50000  # short run, to test changes
uv run scripts/train.py --no-resume        # ignore the saved checkpoint, start fresh
uv run scripts/play.py --fps 30            # play back faster
uv run scripts/play.py --model logs/mario/run_20260101-120000/final_model.zip
```

`uv sync` installs `rlforge` and `games` into `.venv`, so they import from any
file, not just `scripts/`. Run things from the repo root, because `logs/` and
`board/` are relative paths.



### Resuming

`scripts/train.py` **resumes automatically** if `logs/mario/best_model.zip`
exists. That's usually what you want. Two things to know:

- Changing hyperparameters in `TrainConfig` and resuming does **not** fully apply
them — most values are baked into the `.zip`. Only `learning_rate` and
`target_kl` are overridden (see `resume_overrides()` in
[libs/rlforge/src/rlforge/config.py](libs/rlforge/src/rlforge/config.py)).
- To genuinely start over, use `--no-resume`, or delete `logs/mario/best_model.zip`.

Each run gets its own `logs/mario/run_<timestamp>/` folder, so restarting never
overwrites an old run's history.

## 3. How it's organised

```
libs/rlforge/            the engine, a separate repo (git submodule)
└── src/rlforge/
    ├── spec.py          GameSpec: the "socket" a game plugs into
    ├── config.py        TrainConfig + FRAME_SKIP, RESIZE, STALL_PATIENCE
    ├── vec.py           builds the parallel envs
    ├── wrappers.py      StallLimit (ends an episode when the agent stops progressing)
    ├── run_dirs.py      where logs and checkpoints go
    ├── trainer.py       the PPO loop
    └── rollout.py       the watch-it-play loop
games/                   the plugs — one folder per game
├── base_games.py        env ids of the games
└── mario/game.py        the ONLY Mario-specific file
scripts/                 thin CLIs you actually run
docs/                    README images and per-game docs
```

`libs/rlforge/` is [github.com/TomasSirotek/rl-forge](https://github.com/TomasSirotek/rl-forge).

The one rule: `rlforge` **never imports a game.** It receives a `GameSpec` as an
argument. That's what makes a second game cost one file instead of a fork.

### Working on rlforge

`libs/rlforge/` is its own git repo. Changes you make there take effect in
rl-arcade immediately. To save them:

```bash
cd libs/rlforge
git switch main                # submodules start on no branch
git commit -am "..." && git push
cd ../..
git add libs/rlforge && git commit -m "Bump rlforge"   # point rl-arcade at the new commit
```

## 4. Adding a new game

Say you want Sonic. Make one folder, one file:

```
games/sonic/
├── __init__.py     from games.sonic.game import SPEC
└── game.py
```

`games/sonic/game.py` needs exactly two things — a `preprocess()` function and
a `SPEC`:

```python
from gymnasium.wrappers import GrayscaleObservation, MaxAndSkipObservation, ResizeObservation
from rlforge import GameSpec, StallLimit
from rlforge.config import FRAME_SKIP, RESIZE, STALL_PATIENCE

def preprocess(env):
    # whatever your game needs: action-space remap, resize, grayscale, skip
    env = MaxAndSkipObservation(env, skip=FRAME_SKIP)
    env = StallLimit(env, patience=STALL_PATIENCE, progress_key="x")
    env = ResizeObservation(env, RESIZE)
    return GrayscaleObservation(env, keep_dim=True)

SPEC = GameSpec(
    name="sonic",              # -> logs/sonic/, board/sonic/
    env_id="SonicTheHedgehog-Genesis",
    preprocess=preprocess,
    progress_key="x",          # the info[] key that means "made progress"
    report_keys=("x", "rings"),
)
```

That's it. `--game sonic` now appears in both CLIs automatically:

```bash
uv run scripts/train.py --game sonic
uv run scripts/play.py --game sonic
```

You edit **nothing** in `rlforge`.

### What each GameSpec field means


| field          | what it's for                                                            |
| -------------- | ------------------------------------------------------------------------ |
| `name`         | slug for `logs/<name>/` and `board/<name>/`. Keeps games from colliding. |
| `env_id`       | the id the game is registered under in Gymnasium                         |
| `preprocess`   | env → wrapped env. Owns everything game-specific.                        |
| `progress_key` | `info[]` key `StallLimit` watches. Mario uses `x_pos`.                   |
| `report_keys`  | `info[]` keys printed after each episode during play                     |
| `frame_stack`  | frames stacked for motion. Must match between train and play.            |




### Things to be aware of

- `preprocess` **must be identical for training and play.** It is, because both
go through the same `SPEC` — but if you edit it after training, your saved
model sees a different observation than it learned on and plays like garbage.
Retrain after changing it.
- **The action-space remap belongs in** `preprocess`, not in `rlforge`. Mario uses
`JoypadSpace` from nes-py, which only exists for NES games.
- **Pick the right** `progress_key`**.** If the key never appears in `info`,
`StallLimit` sees no progress and truncates every episode at 80 steps.
- **A** `SPEC` **per level.** Mario is `SuperMarioBros-1-1-v0` — level 1-1 only.
Other levels are separate `env_id`s, so give each its own `name` too.



## 5. Games

- [Mario](games/mario/mario.md) — Super Mario Bros NES, reward function, preprocessing, gotchas.



## 6. Gotchas

- **Ignore pre-2023 tutorials** showing `obs = env.reset()` or a 4-value `step()`.
That API is dead — `reset()` returns `(obs, info)` and `step()` returns 5 values.
- **Ignore anything about gym-retro,** `shimmy`**, or** `apply_api_compatibility`**.**
Those were needed only on Python ≤3.12, where pip serves the 2021-era packages.
On 3.13+ you get the modern versions and none of it applies.
- **Requires Python 3.13+.** On older Python, pip silently installs
gym-super-mario-bros 7.4.0 instead of 9.1.0 — that's the old-gym version that
needs all the shims. If you see `import gym` anywhere, you're on the old stack.
- `--cpus 10` **is the default and it's hungry.** Each one is a full NES
emulator process. Drop to `--cpus 4` if you're short on RAM.



## Docs

- [Gymnasium](https://gymnasium.farama.org/) — the core API
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) — env ids, action sets
- [SB3 RL Tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) — why your agent isn't learning
- Best reference is local: `.venv/lib/python3.14/site-packages/gym_super_mario_bros/smb_env.py`

