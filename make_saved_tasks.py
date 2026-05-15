"""Підготувати готові JSON-задачі для швидкого зчитування на захисті.

Створює у saved_tasks/:
  - 4 індивідуальні задачі команди (n=6, 7, 8, 9 з координатами з документа);
  - 1 задачу n=30 (середня, на якій калібрувались параметри АМК);
  - 1 задачу n=50 (велика, для демонстрації різниці алгоритмів).

Запуск:
    python3 make_saved_tasks.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.data_types import SOPInstance
from src.file_io import save_instance
from src.generator import generate_instance

OUT_DIR = os.path.join(HERE, "saved_tasks")


# 4 індивідуальні задачі — точно як у документі ----------------------------

INDIVIDUAL_TASKS = [
    ("koval_n6.json",      6,
     [(10, 15), (25, 8), (20, 25), (35, 12), (30, 30), (45, 20)],
     [[1, 3, 5], [2, 4]]),
    ("bakunets_n7.json",   7,
     [(8, 12), (22, 6), (18, 28), (32, 15), (28, 32), (42, 18), (36, 25)],
     [[1, 3], [2, 4, 6]]),
    ("mykhailova_n8.json", 8,
     [(12, 18), (28, 10), (15, 30), (38, 20), (25, 35), (48, 15),
      (40, 28), (52, 22)],
     [[1, 3, 5], [2, 4, 7]]),
    ("harmash_n9.json",    9,
     [(15, 20), (30, 12), (18, 35), (42, 18), (28, 38), (50, 22),
      (45, 30), (55, 25), (60, 20)],
     [[1, 3, 5], [2, 4, 6, 8], [7, 9]]),
]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Збереження готових задач для захисту...\n")

    # індивідуальні задачі команди
    for fname, n, coords, chains in INDIVIDUAL_TASKS:
        inst = SOPInstance(
            n=n,
            coords=np.array(coords, dtype=float),
            s=np.array([0.0, 0.0]),
            chains=chains,
            p=len({v for c in chains for v in c}) / n,
        )
        path = os.path.join(OUT_DIR, fname)
        save_instance(inst, path)
        print(f"  {fname:24s}  n={n}, ланцюжків={len(chains)}")

    # середня задача n=30 (калібрування експ. 1, 2, 3)
    inst30 = generate_instance(n=30, p=0.4, L_min=2, L_max=4, seed=42)
    save_instance(inst30, os.path.join(OUT_DIR, "task_n30.json"))
    print(f"  {'task_n30.json':24s}  n=30, p=0.4, seed=42  (для калібрування)")

    # велика задача n=50 (експ. 4)
    inst50 = generate_instance(n=50, p=0.4, L_min=2, L_max=4, seed=42)
    save_instance(inst50, os.path.join(OUT_DIR, "task_n50.json"))
    print(f"  {'task_n50.json':24s}  n=50, p=0.4, seed=42  (для демо різниці)")

    print(f"\nГотово. Файлів у {OUT_DIR}: 6")


if __name__ == "__main__":
    main()
