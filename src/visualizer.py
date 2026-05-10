"""Побудова графіків matplotlib для маршрутів та результатів експериментів."""
from __future__ import annotations

import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_types import SOPInstance, SOPResult


def visualize_route(inst: SOPInstance, result: SOPResult,
                    fname: str, title: Optional[str] = None) -> None:
    """Побудувати маршрут на координатній площині та зберегти у файл."""
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    # позиції пайки
    coords = inst.coords
    s = inst.s
    ax.scatter(coords[:, 0], coords[:, 1], c="steelblue", s=80,
               zorder=3, label="Позиція пайки")
    ax.scatter(s[0], s[1], c="green", marker="s", s=120,
               zorder=4, label="Стартова позиція")

    for i, (x, y) in enumerate(coords, start=1):
        ax.annotate(str(i), (x, y), xytext=(6, 6),
                    textcoords="offset points", fontsize=10)

    # маршрут
    route = result.route
    all_pts = np.vstack([s.reshape(1, 2), coords])
    xs = [all_pts[v, 0] for v in route]
    ys = [all_pts[v, 1] for v in route]
    ax.plot(xs, ys, "-", color="crimson", linewidth=1.5, zorder=2,
            label="Маршрут")

    # обмеження передування — підпис у верхньому лівому куті
    if inst.chains:
        chain_text = "Обмеження передування:\n" + "\n".join(
            " → ".join(str(v) for v in c) for c in inst.chains
        )
        ax.text(0.02, 0.98, chain_text, transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow",
                          alpha=0.8))

    if title is None:
        title = f"{result.algorithm.upper()} — Розв'язок (Час: {result.T:.2f} с)"
    ax.set_title(title)
    ax.set_xlabel("x (мм)")
    ax.set_ylabel("y (мм)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_calibration_n_stag(rows: list[dict], fname: str) -> None:
    """Графік для експерименту 1: T̄ як функція π."""
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    pis = [r["pi"] for r in rows]
    Ts = [r["T_avg"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pis, Ts, "o-", color="darkorange", linewidth=2)
    ax.set_xlabel("Значення параметра N_stag (π)")
    ax.set_ylabel("Середнє значення цільової функції T̄")
    ax.set_title("Експеримент 1: вплив N_stag на якість АМК")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_calibration_beta(rows: list[dict], fname: str) -> None:
    """Графік для експерименту 2: T̄ як функція β."""
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    bs = [r["beta"] for r in rows]
    Ts = [r["T_avg"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bs, Ts, "s-", color="teal", linewidth=2)
    ax.set_xlabel("Значення параметра β")
    ax.set_ylabel("Середнє значення цільової функції T̄")
    ax.set_title("Експеримент 2: вплив β на якість АМК (α=1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_param_series(rows: list[dict], param_name: str, fname: str) -> None:
    """Графік для експерименту 3: T̄_g, T̄_a і δ ̄ від параметра задачі."""
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    xs = [r[param_name] for r in rows]
    Tg = [r["T_g_avg"] for r in rows]
    Ta = [r["T_a_avg"] for r in rows]
    delta = [r["delta_avg"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(xs, Tg, "o-", label="Жадібний (T̄_g)", color="steelblue")
    ax1.plot(xs, Ta, "s--", label="АМК (T̄_a)", color="crimson")
    ax1.fill_between(xs, Ta, Tg, alpha=0.15, color="seagreen",
                     label="Виграш АМК")
    ax1.set_xlabel(f"Параметр задачі: {param_name}")
    ax1.set_ylabel("Середній час маршруту T̄ (од.)")
    ax1.set_title(f"T̄ від {param_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(xs, [d * 100 for d in delta], "^-", color="darkorange",
             linewidth=2)
    ax2.set_xlabel(f"Параметр задачі: {param_name}")
    ax2.set_ylabel("Середня парна різниця δ ̄, %")
    ax2.set_title(f"δ ̄ від {param_name}")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linewidth=0.8, linestyle=":")

    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot_n_series(rows: list[dict], fname_quality: str,
                  fname_time: str, fname_gap: Optional[str] = None) -> None:
    """Три графіки для експерименту 4."""
    os.makedirs(os.path.dirname(fname_quality) or ".", exist_ok=True)
    ns = [r["n"] for r in rows]
    Tg = [r["T_g_avg"] for r in rows]
    Ta = [r["T_a_avg"] for r in rows]
    tg = [r["t_g_avg_ms"] for r in rows]
    ta = [r["t_a_avg_ms"] for r in rows]

    # 1. Якість
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, Tg, "o-", label="Жадібний (T̄_g)", color="steelblue")
    ax.plot(ns, Ta, "s--", label="АМК (T̄_a)", color="crimson")
    ax.fill_between(ns, Ta, Tg, alpha=0.15, color="seagreen",
                    label="Виграш АМК")
    ax.set_xlabel("Кількість позицій n")
    ax.set_ylabel("Середній час маршруту T̄ (од.)")
    ax.set_title("Експеримент 4: T̄_g та T̄_a залежно від n")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname_quality, dpi=120)
    plt.close(fig)

    # 2. Час (логарифмічна шкала)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, tg, "o-", label="Жадібний (t̄_g)", color="steelblue")
    ax.plot(ns, ta, "s--", label="АМК (t̄_a)", color="crimson")
    ax.set_yscale("log")
    ax.set_xlabel("Кількість позицій n")
    ax.set_ylabel("Час роботи, мс (логарифмічна шкала)")
    ax.set_title("Експеримент 4: час роботи алгоритмів від n")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fname_time, dpi=120)
    plt.close(fig)

    # 3. gap (якщо є)
    if fname_gap is not None:
        ns_gap = [r["n"] for r in rows if r.get("gap_avg") is not None]
        gaps = [r["gap_avg"] * 100 for r in rows if r.get("gap_avg") is not None]
        if ns_gap:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(ns_gap, gaps, "^-", color="darkorange", linewidth=2)
            ax.set_xlabel("Кількість позицій n")
            ax.set_ylabel("Відносне відхилення від оптимуму gap̄, %")
            ax.set_title("Експеримент 4: gap̄ як функція n (для n ≤ 10)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(fname_gap, dpi=120)
            plt.close(fig)
