"""Генерація PNG-маршрутів для 4 індивідуальних задач з курсової роботи.

Координати, ланцюжки передування та номер задачі взяті безпосередньо
з підрозділів 1.2.1 — 1.2.4 пояснювальної записки.

Кожна задача розв'язується обома алгоритмами (жадібний та АМК);
по 2 PNG на учасника — разом 8 файлів у results/0_individual_tasks/.

Запуск:
    python3 make_individual_routes.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.algorithms.aco import ACOParams, aco_solve_average
from src.algorithms.greedy import greedy_solve
from src.data_types import SOPInstance
from src.visualizer import visualize_route


OUT_DIR = os.path.join(HERE, "results", "0_individual_tasks")


# --- 4 індивідуальні задачі за документом ---------------------------------

TASKS = [
    {
        "owner": "koval",       # 1.2.4, n=6
        "n": 6,
        "coords": [(10, 15), (25, 8), (20, 25), (35, 12), (30, 30), (45, 20)],
        "chains": [[1, 3, 5], [2, 4]],
        "title_name": "Коваль",
    },
    {
        "owner": "bakunets",    # 1.2.3, n=7
        "n": 7,
        "coords": [(8, 12), (22, 6), (18, 28), (32, 15), (28, 32),
                   (42, 18), (36, 25)],
        "chains": [[1, 3], [2, 4, 6]],
        "title_name": "Бакунець",
    },
    {
        "owner": "mykhailova",  # 1.2.1, n=8
        "n": 8,
        "coords": [(12, 18), (28, 10), (15, 30), (38, 20), (25, 35),
                   (48, 15), (40, 28), (52, 22)],
        "chains": [[1, 3, 5], [2, 4, 7]],
        "title_name": "Михайлова",
    },
    {
        "owner": "harmash",     # 1.2.2, n=9
        "n": 9,
        "coords": [(15, 20), (30, 12), (18, 35), (42, 18), (28, 38),
                   (50, 22), (45, 30), (55, 25), (60, 20)],
        "chains": [[1, 3, 5], [2, 4, 6, 8], [7, 9]],
        "title_name": "Гармаш",
    },
]


def build_instance(task: dict) -> SOPInstance:
    return SOPInstance(
        n=task["n"],
        coords=np.array(task["coords"], dtype=float),
        s=np.array([0.0, 0.0]),
        chains=task["chains"],
        p=len({v for c in task["chains"] for v in c}) / task["n"],
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Генерація PNG-маршрутів для 4 індивідуальних задач...\n")

    for task in TASKS:
        inst = build_instance(task)
        owner = task["owner"]
        name = task["title_name"]
        n = task["n"]

        # --- жадібний ---
        g_res = greedy_solve(inst)
        fname_g = os.path.join(OUT_DIR, f"{owner}_n{n}_greedy.png")
        visualize_route(
            inst, g_res, fname_g,
            title=f"Жадібний — {name} (n={n}), T = {g_res.T:.2f} с",
        )
        print(f"  {os.path.basename(fname_g):28s}  T = {g_res.T:6.2f} с")

        # --- АМК (30 незалежних запусків, рекордний) ---
        a_best, stats = aco_solve_average(
            inst,
            ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                      m_a=None, N_iter=100, N_stag=30),
            R_aco=30,
            base_seed=0,
        )
        fname_a = os.path.join(OUT_DIR, f"{owner}_n{n}_aco.png")
        visualize_route(
            inst, a_best, fname_a,
            title=f"АМК — {name} (n={n}), T = {a_best.T:.2f} с "
                  f"(T_avg={stats['T_avg']:.2f})",
        )
        print(f"  {os.path.basename(fname_a):28s}  T = {a_best.T:6.2f} с "
              f"(T_g − T_a = {g_res.T - a_best.T:+.2f} с)")
        print()

    print(f"Готово. {len(TASKS) * 2} файлів у: {OUT_DIR}")


if __name__ == "__main__":
    main()
