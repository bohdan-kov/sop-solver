"""Жадібний алгоритм розв'язання задачі SOP."""
from __future__ import annotations

import time
from typing import List

import numpy as np

from ..data_types import SOPInstance, SOPResult


def _build_predecessor_lookup(inst: SOPInstance):
    """Побудувати rem_pred та succ_list для перевірки допустимості за O(1)."""
    n = inst.n
    rem_pred = np.zeros(n + 1, dtype=int)        # індекс 1..n
    succ_list: List[List[int]] = [[] for _ in range(n + 1)]
    for chain in inst.chains:
        for k in range(1, len(chain)):
            rem_pred[chain[k]] += 1
            succ_list[chain[k - 1]].append(chain[k])
    return rem_pred, succ_list


def greedy_solve(inst: SOPInstance) -> SOPResult:
    """Жадібний алгоритм для SOP.

    На кожному кроці обирає найближчу допустиму позицію (з урахуванням
    обмежень передування). Обчислювальна складність O(n^2).
    """
    start = time.perf_counter()

    n = inst.n
    D = inst.D
    rem_pred, succ_list = _build_predecessor_lookup(inst)

    visited = np.zeros(n + 1, dtype=bool)
    route: List[int] = [0]                       # 0 — стартова позиція s
    current = 0
    T = 0.0

    while int(visited[1:].sum()) < n:
        best_v = -1
        best_d = float("inf")
        for v in range(1, n + 1):
            if visited[v]:
                continue
            if rem_pred[v] != 0:
                continue
            if D[current, v] < best_d:
                best_d = D[current, v]
                best_v = v
        if best_v == -1:
            raise RuntimeError("Не знайдено допустимого кандидата (цикл у ланцюжках?)")
        visited[best_v] = True
        for w in succ_list[best_v]:
            rem_pred[w] -= 1
        route.append(best_v)
        T += best_d
        current = best_v

    # повернення до стартової позиції
    T += D[current, 0]
    route.append(0)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SOPResult(route=route, T=T, t_ms=elapsed_ms, algorithm="greedy")
