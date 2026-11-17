"""Train-only RMS normalization for ten-dimensional moments."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class MomentNormalizer:
    scales: np.ndarray
    count: int

    def __post_init__(self):
        if self.scales.shape != (10,):
            raise ValueError("scales must have shape [10]")
        if not np.isfinite(self.scales).all():
            raise ValueError("nonfinite normalization scales")
        if (self.scales <= 0).any():
            raise ValueError("normalization scales must be positive")

    @classmethod
    def fit(cls, corpus):
        total_square = np.zeros(10, dtype=np.float64)
        count = 0

        for _, _, labels in corpus.iter_split("train"):
            moments = np.asarray(
                labels["moments"], dtype=np.float64
            )
            total_square += np.square(moments).sum(axis=0)
            count += len(moments)

        if count == 0:
            raise ValueError("empty training set")

        rms = np.sqrt(total_square / count)

        # Numerical floor only; do not center physical moments.
        rms = np.maximum(rms, 1e-12)

        return cls(rms, count)

    def tensor(self, device, dtype=torch.float64):
        return torch.as_tensor(
            self.scales.copy(),
            device=device,
            dtype=dtype,
        )

    def transform(self, values):
        return values / self.scales

    def state_dict(self):
        return {
            "scales": self.scales.tolist(),
            "count": self.count,
            "method": "train-only-rms-no-centering",
        }

    @classmethod
    def from_state_dict(cls, state):
        if state["method"] != "train-only-rms-no-centering":
            raise ValueError("unknown normalization method")
        return cls(
            np.asarray(state["scales"], dtype=np.float64),
            int(state["count"]),
        )
