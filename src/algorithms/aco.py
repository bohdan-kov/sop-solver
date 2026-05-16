"""Алгоритм мурашиних колоній (АМК) для задачі SOP."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data_types import SOPInstance, SOPResult
from .greedy import _build_predecessor_lookup


@dataclass
class ACOParams:
    """Специфічні параметри АМК."""

    alpha: float = 1.0          # ступінь впливу феромону
    beta: float = 2.0           # ступінь "жадібності" (евристика)
    rho: float = 0.5            # коефіцієнт випаровування, 0 < rho < 1
    m_a: Optional[int] = None   # кількість мурах; за замовчуванням m_a = n
    N_iter: int = 10000         # максимальна кількість ітерацій (страхувальний ліміт)
    N_stag: int = 30            # ітерацій без поліпшення → стоп

    def resolve(self, n: int) -> "ACOParams":
        """Заповнити m_a, якщо не задано."""
        if self.m_a is None:
            return ACOParams(self.alpha, self.beta, self.rho, n,
                             self.N_iter, self.N_stag)
        return self


def _build_one_route(
    D: np.ndarray,
    tau: np.ndarray,
    rem_pred_init: np.ndarray,
    succ_list: List[List[int]],
    n: int,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> tuple[List[int], float]:
    """Побудувати один маршрут однією мурахою."""
    rem_pred = rem_pred_init.copy()
    visited = np.zeros(n + 1, dtype=bool)
    route: List[int] = [0]
    current = 0
    T = 0.0

    for _ in range(n):
        # формуємо множину допустимих кандидатів
        candidates: List[int] = []
        for v in range(1, n + 1):
            if not visited[v] and rem_pred[v] == 0:
                candidates.append(v)
        if not candidates:
            raise RuntimeError("Немає допустимих кандидатів")

        # обчислюємо ваги переходів
        cand = np.array(candidates, dtype=int)
        d_curr = D[current, cand]
        # уникаємо ділення на нуль (зазвичай d_curr > 0)
        eta = 1.0 / np.where(d_curr > 0, d_curr, 1e-9)
        tau_curr = tau[current, cand]
        weights = (tau_curr ** alpha) * (eta ** beta)
        s = weights.sum()
        if s <= 0 or not np.isfinite(s):
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / s

        nxt = int(rng.choice(cand, p=probs))
        visited[nxt] = True
        for w in succ_list[nxt]:
            rem_pred[w] -= 1
        route.append(nxt)
        T += float(D[current, nxt])
        current = nxt

    T += float(D[current, 0])
    route.append(0)
    return route, T


def aco_solve(
    inst: SOPInstance,
    params: ACOParams,
    seed: Optional[int] = None,
) -> SOPResult:
    """Один прогін АМК (R_aco = 1) на одній індивідуальній задачі.

    Запускає метаевристичний пошук з параметрами params; повертає
    найкращий знайдений маршрут.
    """
    start = time.perf_counter()

    p = params.resolve(inst.n)
    n = inst.n
    D = inst.D
    rem_pred_init, succ_list = _build_predecessor_lookup(inst)

    # нижня межа T_LB та калібрування tau_0, Q за формулами (2.4)-(2.6)
    D_off = D.copy()
    np.fill_diagonal(D_off, np.inf)
    T_LB = float(D_off.min(axis=1).sum())
    tau0 = 1.0 / (n * T_LB) if T_LB > 0 else 1.0
    Q = T_LB if T_LB > 0 else 1.0

    tau = np.full((n + 1, n + 1), tau0, dtype=float)
    rng = np.random.default_rng(seed)

    best_T = float("inf")
    best_route: List[int] = []
    stag = 0

    for it in range(p.N_iter):
        if stag >= p.N_stag:
            break

        # m мурах будують маршрути
        ant_routes: List[tuple[List[int], float]] = []
        for _ in range(p.m_a):
            r, t = _build_one_route(
                D, tau, rem_pred_init, succ_list, n,
                p.alpha, p.beta, rng,
            )
            ant_routes.append((r, t))

        # випаровування феромону
        tau *= (1.0 - p.rho)

        # відкладення феромону
        for r, t in ant_routes:
            if t <= 0:
                continue
            delta = Q / t
            for k in range(len(r) - 1):
                u, v = r[k], r[k + 1]
                tau[u, v] += delta
                tau[v, u] += delta  # симетрична задача

        # оновлення рекорду
        iter_best_t = min((t for _, t in ant_routes), default=float("inf"))
        improved = False
        for r, t in ant_routes:
            if t < best_T:
                best_T = t
                best_route = r
                improved = True
        stag = 0 if improved else stag + 1

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return SOPResult(route=best_route, T=best_T, t_ms=elapsed_ms,
                     algorithm="aco")


def aco_solve_average(
    inst: SOPInstance,
    params: ACOParams,
    R_aco: int = 30,
    base_seed: int = 0,
) -> tuple[SOPResult, dict]:
    """Виконати R_aco незалежних запусків АМК; повернути найкращий
    та агрегатну статистику (T_min, T_avg, D)."""
    Ts: List[float] = []
    ts: List[float] = []
    best: Optional[SOPResult] = None
    for r in range(R_aco):
        res = aco_solve(inst, params, seed=base_seed + r)
        Ts.append(res.T)
        ts.append(res.t_ms)
        if best is None or res.T < best.T:
            best = res
    Ts_arr = np.array(Ts)
    stats = {
        "T_min": float(Ts_arr.min()),
        "T_avg": float(Ts_arr.mean()),
        "D": float(Ts_arr.var(ddof=0)),
        "t_avg_ms": float(np.mean(ts)),
    }
    assert best is not None
    return best, stats
