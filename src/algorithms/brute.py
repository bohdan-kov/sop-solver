"""Точний розв'язок задачі SOP повним перебором (для малих n).

Використовується в експерименті 4 для обчислення gap = (T-T*)/T*.
"""
from __future__ import annotations

import time
from itertools import permutations
from typing import List

import numpy as np

from ..data_types import SOPInstance, SOPResult


def brute_force_solve(inst: SOPInstance) -> SOPResult:
    """Перебрати всі n! перестановок позицій 1..n; обрати ту, що
    задовольняє обмеження передування і має мінімальний T.

    Працює тільки для n <= 10. Для n > 10 викине ValueError.
    """
    if inst.n > 10:
        raise ValueError(
            f"Повний перебір недоступний для n = {inst.n} (n > 10)"
        )

    start = time.perf_counter()
    n = inst.n
    D = inst.D

    # індекси позицій передування у вигляді словника: для пози v —
    # множина всіх позицій, які мають іти ДО неї.
    must_before: dict[int, set[int]] = {v: set() for v in range(1, n + 1)}
    for chain in inst.chains:
        for k in range(len(chain)):
            for j in range(k):
                must_before[chain[k]].add(chain[j])

    best_T = float("inf")
    best_route: List[int] = []

    for perm in permutations(range(1, n + 1)):
        # перевірка передування
        seen: set[int] = set()
        feasible = True
        for v in perm:
            if not must_before[v].issubset(seen):
                feasible = False
                break
            seen.add(v)
        if not feasible:
            continue

        # обчислюємо T
        T = float(D[0, perm[0]])
        for k in range(len(perm) - 1):
            T += float(D[perm[k], perm[k + 1]])
        T += float(D[perm[-1], 0])

        if T < best_T:
            best_T = T
            best_route = [0, *perm, 0]

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if best_T == float("inf"):
        raise RuntimeError("Не існує допустимого маршруту")
    return SOPResult(
        route=best_route,
        T=best_T,
        t_ms=elapsed_ms,
        algorithm="brute",
    )
