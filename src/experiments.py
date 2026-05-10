"""Серії експериментів та калібрування параметрів АМК.

Реалізовано всі 4 експерименти, описані у підрозділах 3.3.2 — 3.3.5.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .algorithms.aco import ACOParams, aco_solve_average
from .algorithms.brute import brute_force_solve
from .algorithms.greedy import greedy_solve
from .data_types import SOPInstance
from .generator import generate_instance


# ---------- метрики (формули (3.1) — (3.3) з розділу 3) -------------------

def _compute_pair_metrics(T_g: float, T_a: float) -> tuple[float, int]:
    """Повертає (delta, win_flag) для одної задачі."""
    if T_g > 0:
        delta = (T_g - T_a) / T_g
    else:
        delta = 0.0
    win = 1 if T_a < T_g else 0
    return delta, win


# ---------- експеримент 1: вплив N_stag на якість АМК --------------------

def experiment_n_stag(
    pi_values: List[int],
    R: int = 30,
    R_aco: int = 30,
    n: int = 30,
    p: float = 0.4,
    L_min: int = 2,
    L_max: int = 4,
    aco_params: Optional[ACOParams] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Експеримент 1: знайти ефективне N_stag.

    Для кожного значення pi генеруються R задач і запускається АМК
    R_aco разів на кожній задачі; результати усереднюються.
    """
    if aco_params is None:
        aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                               m_a=None, N_iter=100, N_stag=30)

    rows: List[dict] = []
    for pi in pi_values:
        if progress_cb:
            progress_cb(f"  π = {pi}: запуск {R} задач × {R_aco} прогонів АМК...")
        T_sum = 0.0
        t_sum = 0.0
        for i in range(R):
            inst = generate_instance(n, p=p, L_min=L_min, L_max=L_max,
                                     seed=i)
            params = ACOParams(aco_params.alpha, aco_params.beta,
                               aco_params.rho, aco_params.m_a,
                               aco_params.N_iter, pi)
            best, stats = aco_solve_average(inst, params, R_aco=R_aco,
                                            base_seed=1000 * i)
            T_sum += stats["T_avg"]
            t_sum += stats["t_avg_ms"]
        rows.append({
            "pi": pi,
            "T_avg": T_sum / R,
            "t_avg_ms": t_sum / R,
        })
    return rows


# ---------- експеримент 2: вплив β на якість АМК -------------------------

def experiment_beta(
    beta_values: List[float],
    R: int = 30,
    R_aco: int = 30,
    n: int = 30,
    p: float = 0.4,
    L_min: int = 2,
    L_max: int = 4,
    aco_params: Optional[ACOParams] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Експеримент 2: знайти ефективне β при α = 1."""
    if aco_params is None:
        aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                               m_a=None, N_iter=100, N_stag=30)

    rows: List[dict] = []
    for beta in beta_values:
        if progress_cb:
            progress_cb(f"  β = {beta}: запуск {R} задач × {R_aco} прогонів АМК...")
        T_sum = 0.0
        t_sum = 0.0
        for i in range(R):
            inst = generate_instance(n, p=p, L_min=L_min, L_max=L_max,
                                     seed=i)
            params = ACOParams(aco_params.alpha, beta, aco_params.rho,
                               aco_params.m_a, aco_params.N_iter,
                               aco_params.N_stag)
            best, stats = aco_solve_average(inst, params, R_aco=R_aco,
                                            base_seed=2000 * i)
            T_sum += best.T  # використовуємо рекордне T_min
            t_sum += stats["t_avg_ms"]
        rows.append({
            "beta": beta,
            "T_avg": T_sum / R,
            "t_avg_ms": t_sum / R,
        })
    return rows


# ---------- експеримент 3: вплив параметрів задачі (p, L) ---------------

def experiment_param_series(
    param_name: str,
    param_values: List,
    R: int = 30,
    R_aco: int = 30,
    n: int = 30,
    p: float = 0.4,
    L_min: int = 2,
    L_max: int = 4,
    aco_params: Optional[ACOParams] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Експеримент 3: вплив параметра задачі (p або L) на ефективність.

    Для p: param_values — список значень щільності.
    Для L: param_values — список середніх довжин ланцюжка
           (внутрішньо L_min = L-1, L_max = L+1, з обмеженнями).
    """
    if aco_params is None:
        aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                               m_a=None, N_iter=100, N_stag=30)

    rows: List[dict] = []
    for v in param_values:
        if progress_cb:
            progress_cb(f"  {param_name} = {v}: запуск {R} задач × {R_aco} прогонів...")

        T_g_sum = T_a_sum = 0.0
        t_g_sum = t_a_sum = 0.0
        delta_sum = 0.0
        wins = 0

        # підставляємо значення параметра
        if param_name == "p":
            cur_p = float(v)
            cur_Lmin, cur_Lmax = L_min, L_max
        elif param_name == "L":
            cur_p = p
            L = int(v)
            cur_Lmin = max(2, L - 1)
            cur_Lmax = max(cur_Lmin, L + 1)
        else:
            raise ValueError(f"невідомий параметр: {param_name}")

        for i in range(R):
            inst = generate_instance(n, p=cur_p,
                                     L_min=cur_Lmin, L_max=cur_Lmax,
                                     seed=i)
            g_res = greedy_solve(inst)
            best, stats = aco_solve_average(inst, aco_params,
                                            R_aco=R_aco,
                                            base_seed=3000 * i)
            T_g_sum += g_res.T
            T_a_sum += best.T
            t_g_sum += g_res.t_ms
            t_a_sum += stats["t_avg_ms"]
            delta, win = _compute_pair_metrics(g_res.T, best.T)
            delta_sum += delta
            wins += win

        rows.append({
            param_name: v,
            "T_g_avg": T_g_sum / R,
            "T_a_avg": T_a_sum / R,
            "t_g_avg_ms": t_g_sum / R,
            "t_a_avg_ms": t_a_sum / R,
            "delta_avg": delta_sum / R,
            "w": wins / R,
        })
    return rows


# ---------- експеримент 4: вплив n на точність та час -------------------

def experiment_n(
    n_values: List[int],
    R: int = 20,
    R_aco: int = 30,
    p: float = 0.4,
    L_min: int = 2,
    L_max: int = 4,
    aco_params: Optional[ACOParams] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Експеримент 4 (2-в-1): вплив n на якість і час.

    Для n <= 10 додатково обчислюється gap = (T_a - T_opt) / T_opt
    через повний перебор.
    """
    if aco_params is None:
        aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                               m_a=None, N_iter=100, N_stag=30)

    rows: List[dict] = []
    for n in n_values:
        if progress_cb:
            progress_cb(f"  n = {n}: запуск {R} задач × {R_aco} прогонів АМК...")

        T_g_sum = T_a_sum = 0.0
        t_g_sum = t_a_sum = 0.0
        delta_sum = 0.0
        gap_sum = 0.0
        gap_count = 0
        wins = 0

        for i in range(R):
            inst = generate_instance(n, p=p, L_min=L_min, L_max=L_max,
                                     seed=i)
            g_res = greedy_solve(inst)
            best, stats = aco_solve_average(inst, aco_params,
                                            R_aco=R_aco,
                                            base_seed=4000 * i)

            T_g_sum += g_res.T
            T_a_sum += best.T
            t_g_sum += g_res.t_ms
            t_a_sum += stats["t_avg_ms"]
            delta, win = _compute_pair_metrics(g_res.T, best.T)
            delta_sum += delta
            wins += win

            if n <= 10:
                opt = brute_force_solve(inst)
                if opt.T > 0:
                    gap_sum += (best.T - opt.T) / opt.T
                    gap_count += 1

        row = {
            "n": n,
            "T_g_avg": T_g_sum / R,
            "T_a_avg": T_a_sum / R,
            "t_g_avg_ms": t_g_sum / R,
            "t_a_avg_ms": t_a_sum / R,
            "delta_avg": delta_sum / R,
            "w": wins / R,
        }
        if gap_count > 0:
            row["gap_avg"] = gap_sum / gap_count
        else:
            row["gap_avg"] = None
        rows.append(row)
    return rows
