"""Fixed MLP matching the blog's 600 -> 32 -> 32 -> 6 architecture."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

GRID = 10
INPUT_CHANNELS = (
    "player", "moving_object", "reward_target", "obstacle",
    "previous_frame", "gauge",
)
# Original MinAtar API uses n/l/u/r/d/f; not a generalized action order.
ACTIONS = ("no_op", "left", "up", "right", "down", "fire")
N_INPUTS, N_HIDDEN, N_ACTIONS = 600, 32, 6


def index(channel: int, row: int, col: int) -> int:
    """Channel-first flattening: [6, 10, 10]."""
    if not (0 <= channel < 6 and 0 <= row < GRID and 0 <= col < GRID):
        raise ValueError("Grid/channel index out of bounds")
    return channel * 100 + row * 10 + col


def encode(obs: np.ndarray) -> np.ndarray:
    """Canonical [6,10,10] observation to 600 float32 features."""
    if obs.shape != (6, 10, 10):
        raise ValueError(f"Expected (6,10,10), received {obs.shape}")
    return np.asarray(obs, dtype=np.float32).reshape(600)


@dataclass
class MLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray

    @classmethod
    def zeros(cls) -> "MLP":
        return cls(
            np.zeros((32, 600), np.float32),
            np.zeros(32, np.float32),
            np.zeros((32, 32), np.float32),
            np.zeros(32, np.float32),
            np.zeros((6, 32), np.float32),
            np.zeros(6, np.float32),
        )

    @property
    def parameter_count(self) -> int:
        return sum(x.size for x in self.arrays().values())

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ("w1", "b1", "w2", "b2", "w3", "b3")}

    def logits(self, observation: np.ndarray) -> np.ndarray:
        """Inference only; no gradient tape, training, or optimizer."""
        x = encode(observation)
        h1 = np.tanh(self.w1 @ x + self.b1)
        h2 = np.tanh(self.w2 @ h1 + self.b2)
        return self.w3 @ h2 + self.b3

    def act(self, observation: np.ndarray, rng: np.random.Generator,
            temperature: float = 1.0) -> int:
        logits = self.logits(observation)
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if temperature == 0:
            return int(np.argmax(logits))
        z = (logits - logits.max()) / temperature
        p = np.exp(z)
        p /= p.sum()
        return int(rng.choice(6, p=p))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        with np.load(path, allow_pickle=False) as data:
            names = ("w1", "b1", "w2", "b2", "w3", "b3")
            arrays = [np.array(data[name], dtype=np.float32) for name in names]
        model = cls(*arrays)
        expected = cls.zeros()
        for name in names:
            if model.arrays()[name].shape != expected.arrays()[name].shape:
                raise ValueError(f"Invalid shape for {name}")
        return model
