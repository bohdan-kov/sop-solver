"""Структури даних для задачі SOP."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SOPInstance:
    """Індивідуальна задача SOP (sequential ordering problem).

    Зберігає опис задачі оптимізації послідовності пайки на платі
    з обмеженнями порядку.
    """

    n: int
    coords: np.ndarray            # shape (n, 2)
    s: np.ndarray                 # shape (2,) — стартова позиція
    chains: List[List[int]]       # ланцюжки передування (1-індексовані)
    p: float = 0.0                # щільність обмежень (для відтворюваності)
    X_max: float = 100.0
    Y_max: float = 100.0
    seed: Optional[int] = None
    _D: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def D(self) -> np.ndarray:
        """Матриця часів переміщень розмірності (n+1) x (n+1).

        Індекс 0 відповідає стартовій позиції s, індекси 1..n —
        позиціям пайки. Швидкість головки v = 10 мм/с (фіксована).
        """
        if self._D is None:
            self._D = self._build_D()
        return self._D

    def _build_D(self, v: float = 10.0) -> np.ndarray:
        all_pts = np.vstack([self.s.reshape(1, 2), self.coords])
        diff = all_pts[:, None, :] - all_pts[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        return dist / v

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "X_max": self.X_max,
            "Y_max": self.Y_max,
            "p": self.p,
            "seed": self.seed,
            "start": self.s.tolist(),
            "coords": self.coords.tolist(),
            "chains": self.chains,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SOPInstance":
        return cls(
            n=int(data["n"]),
            coords=np.array(data["coords"], dtype=float),
            s=np.array(data["start"], dtype=float),
            chains=[list(c) for c in data["chains"]],
            p=float(data.get("p", 0.0)),
            X_max=float(data.get("X_max", 100.0)),
            Y_max=float(data.get("Y_max", 100.0)),
            seed=data.get("seed"),
        )


@dataclass
class SOPResult:
    """Результат роботи алгоритму на одній індивідуальній задачі."""

    route: List[int]              # послідовність вершин у маршруті
    T: float                      # значення цільової функції (час маршруту)
    t_ms: float                   # час роботи алгоритму, мс
    algorithm: str                # "greedy" | "aco" | "brute"

    def to_dict(self) -> dict:
        return {
            "route": list(self.route),
            "T": float(self.T),
            "t_ms": float(self.t_ms),
            "algorithm": self.algorithm,
        }
