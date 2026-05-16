"""Допоміжний скрипт для масової генерації рисунків маршрутів.

Створює 4 PNG у директорії results/:
  - route_greedy_n8.png   (жадібний, мала задача)
  - route_aco_n8.png      (АМК, мала задача)
  - route_greedy_n30.png  (жадібний, середня задача)
  - route_aco_n30.png     (АМК, середня задача)

Запуск:
    python3 make_route_pictures.py
"""
from __future__ import annotations

import os

from src.algorithms.aco import ACOParams, aco_solve_average
from src.algorithms.greedy import greedy_solve
from src.generator import generate_instance
from src.visualizer import visualize_route


RESULTS_DIR = "results"


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Генерація маршрутів для рисунків Розділу 5...")

    # ---- мала задача n = 8 ------------------------------------------------
    inst8 = generate_instance(n=8, p=0.25, L_min=2, L_max=3, seed=11)
    g8 = greedy_solve(inst8)
    visualize_route(
        inst8, g8,
        os.path.join(RESULTS_DIR, "route_greedy_n8.png"),
        title=f"Жадібний — n=8, T={g8.T:.2f}",
    )
    print(f"  route_greedy_n8.png  (T = {g8.T:.2f})")

    a8, _ = aco_solve_average(
        inst8, ACOParams(N_iter=10000, N_stag=30),
        R_aco=30, base_seed=0,
    )
    visualize_route(
        inst8, a8,
        os.path.join(RESULTS_DIR, "route_aco_n8.png"),
        title=f"АМК — n=8, T={a8.T:.2f}",
    )
    print(f"  route_aco_n8.png     (T = {a8.T:.2f})")

    # ---- середня задача n = 30 -------------------------------------------
    inst30 = generate_instance(n=30, p=0.4, L_min=2, L_max=4, seed=42)
    g30 = greedy_solve(inst30)
    visualize_route(
        inst30, g30,
        os.path.join(RESULTS_DIR, "route_greedy_n30.png"),
        title=f"Жадібний — n=30, T={g30.T:.2f}",
    )
    print(f"  route_greedy_n30.png (T = {g30.T:.2f})")

    a30, _ = aco_solve_average(
        inst30, ACOParams(N_iter=10000, N_stag=30),
        R_aco=30, base_seed=0,
    )
    visualize_route(
        inst30, a30,
        os.path.join(RESULTS_DIR, "route_aco_n30.png"),
        title=f"АМК — n=30, T={a30.T:.2f}",
    )
    print(f"  route_aco_n30.png    (T = {a30.T:.2f})")

    print("\nГотово. Файли у директорії results/.")


if __name__ == "__main__":
    main()
