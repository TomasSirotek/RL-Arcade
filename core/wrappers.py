import gymnasium as gym


class StallLimit(gym.Wrapper):
    """End the episode if `progress_key` hasn't improved in `patience` steps."""
    def __init__(self, env, patience=80, progress_key="x_pos"):
        super().__init__(env)
        self.patience = patience
        self.progress_key = progress_key

    def reset(self, **kwargs):
        self.best, self.stale = 0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get(self.progress_key, 0) > self.best:
            self.best, self.stale = info[self.progress_key], 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                truncated = True
        return obs, reward, terminated, truncated, info
