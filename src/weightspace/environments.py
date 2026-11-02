"""Small independent toy environment, plus optional official MinAtar adapter.

ToyBreakout is a cheap offline smoke test, NOT the published environment.
"""
import numpy as np


class ToyBreakout:
    """10x10 paddle/brick dynamics for installation-free validation."""

    def __init__(self, sticky: float = 0.1):
        self.sticky = sticky
        self.reset(0)

    def reset(self, seed: int) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.paddle_x = 4
        self.ball_x = int(self.rng.integers(2, 8))
        self.ball_y = 6
        self.vx = int(self.rng.choice([-1, 1]))
        self.vy = -1
        self.prev_x, self.prev_y = self.ball_x, self.ball_y
        self.bricks = np.zeros((10, 10), np.float32)
        self.bricks[1:4, :] = 1
        self.done = False
        self.last_action = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        obs = np.zeros((6, 10, 10), np.float32)
        obs[0, 9, self.paddle_x] = 1
        obs[1, self.ball_y, self.ball_x] = 1
        obs[2] = self.bricks
        obs[4, self.prev_y, self.prev_x] = 1
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if self.done:
            raise RuntimeError("Cannot step a terminated episode")
        if self.rng.random() < self.sticky:
            action = self.last_action
        self.last_action = action
        if action == 1:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == 3:
            self.paddle_x = min(9, self.paddle_x + 1)

        self.prev_x, self.prev_y = self.ball_x, self.ball_y
        nx, ny = self.ball_x + self.vx, self.ball_y + self.vy
        if not (0 <= nx <= 9):
            self.vx *= -1
            nx = self.ball_x + self.vx
        if ny < 0:
            self.vy *= -1
            ny = self.ball_y + self.vy
        reward = 0.0
        if ny < 9 and self.bricks[ny, nx] == 1:
            self.bricks[ny, nx] = 0
            reward = 1.0
            self.vy *= -1
            ny = self.ball_y
        elif ny >= 9:
            if abs(nx - self.paddle_x) <= 1:
                self.vy = -1
                ny = self.ball_y
            else:
                self.done = True
                ny = 9
        if not self.bricks.any():
            self.bricks[1:4, :] = 1
        self.ball_x, self.ball_y = nx, ny
        return self.observe(), reward, self.done


class MinatarBreakout:
    """Use MinAtar's real Breakout with all six native actions."""

    def __init__(self, sticky: float = 0.1):
        try:
            from minatar import Environment
        except ImportError as exc:
            raise ImportError("Install official environment: pip install '.[minatar]'") from exc
        self.env = Environment("breakout", sticky_action_prob=sticky)

    def reset(self, seed: int) -> np.ndarray:
        self.env.seed(seed)
        self.env.last_action = 0
        self.env.reset()
        return self.observe()

    def observe(self) -> np.ndarray:
        raw = self.env.state()
        if raw.shape != (10, 10, 4):
            raise ValueError(f"Unexpected upstream Breakout observation: {raw.shape}")
        canonical = np.zeros((6, 10, 10), dtype=np.float32)
        canonical[0] = raw[:, :, 0]
        canonical[1] = raw[:, :, 1]
        canonical[2] = raw[:, :, 3]
        canonical[4] = raw[:, :, 2]
        return canonical

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        reward, done = self.env.act(action)
        return self.observe(), float(reward), bool(done)


def make_env(backend: str, sticky: float = 0.1):
    if backend == "toy":
        return ToyBreakout(sticky=sticky)
    if backend == "minatar":
        return MinatarBreakout(sticky=sticky)
    raise ValueError(f"Unknown environment: {backend}")
