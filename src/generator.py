"""Генератор індивідуальних задач SOP."""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from .data_types import SOPInstance


def generate_instance(
    n: int,
    X_max: float = 100.0,
    Y_max: float = 100.0,
    p: float = 0.4,
    L_min: int = 2,
    L_max: int = 4,
    seed: Optional[int] = None,
) -> SOPInstance:
    """Згенерувати індивідуальну задачу SOP.

    Параметри відповідають специфікації з підрозділу 3.2.

    n        — кількість позицій пайки;
    X_max    — ширина прямокутної зони;
    Y_max    — висота прямокутної зони;
    p        — щільність обмежень передування, p in [0; 1];
    L_min    — мінімальна довжина ланцюжка передування;
    L_max    — максимальна довжина ланцюжка передування;
    seed     — зерно генератора випадкових чисел.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p має бути у [0; 1]")
    if L_min < 2 or L_max < L_min:
        raise ValueError("L_min >= 2 і L_max >= L_min")
    if n < 2:
        raise ValueError("n >= 2")

    rng = np.random.default_rng(seed)
    py_rnd = random.Random(seed)

    coords = rng.uniform(low=[0.0, 0.0], high=[X_max, Y_max], size=(n, 2))
    s = np.array([0.0, 0.0])

    n_chained = int(p * n)
    chains: List[List[int]] = []
    if n_chained >= 2:
        all_indices = list(range(1, n + 1))   # 1-indexed позиції
        py_rnd.shuffle(all_indices)
        pool = all_indices[:n_chained]
        while pool:
            L = py_rnd.randint(L_min, L_max)
            L = min(L, len(pool))
            if L < 2:
                break
            chain = pool[:L]
            pool = pool[L:]
            chains.append(chain)

    return SOPInstance(
        n=n,
        coords=coords,
        s=s,
        chains=chains,
        p=p,
        X_max=X_max,
        Y_max=Y_max,
        seed=seed,
    )
