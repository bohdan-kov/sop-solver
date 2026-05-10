"""Виведення результатів та таблиць у консоль."""
from __future__ import annotations

from typing import List, Optional

from .data_types import SOPResult


def format_route(route: List[int]) -> str:
    if len(route) <= 12:
        return " → ".join(str(v) if v != 0 else "s" for v in route)
    head = " → ".join(str(v) if v != 0 else "s" for v in route[:6])
    tail = " → ".join(str(v) if v != 0 else "s" for v in route[-3:])
    return f"{head} → ... → {tail}"


def print_solve_results(greedy_res: SOPResult,
                        aco_res: SOPResult,
                        aco_stats: Optional[dict] = None) -> None:
    print("\nРезультати алгоритмів:")
    print(f"  Жадібний:  T = {greedy_res.T:.4f},  t = {greedy_res.t_ms:.2f} мс")
    print(f"             Маршрут: {format_route(greedy_res.route)}")
    print(f"  АМК:       T_min = {aco_res.T:.4f},  t = {aco_res.t_ms:.2f} мс")
    print(f"             Маршрут: {format_route(aco_res.route)}")

    if aco_stats is not None:
        print(f"  АМК (статистика за серією): "
              f"T_avg = {aco_stats['T_avg']:.4f}, "
              f"D = {aco_stats['D']:.4f}")

    if greedy_res.T > 0:
        delta = (greedy_res.T - aco_res.T) / greedy_res.T
    else:
        delta = 0.0
    sign = "+" if delta > 0 else ""
    print()
    print(f"Порівняння:")
    print(f"  delta = (T_g - T_a) / T_g = {sign}{delta:.4f}  "
          f"({sign}{delta * 100:.2f} %)")
    if delta > 0:
        print("  АМК переміг: ТАК")
    elif delta < 0:
        print("  АМК переміг: НІ (жадібний дав кращий результат)")
    else:
        print("  АМК переміг: РІВНО (співпали)")


def print_table(rows: list[dict], header: list[str]) -> None:
    col_widths = [max(len(h), 10) for h in header]
    print()
    print("  ".join(h.ljust(w) for h, w in zip(header, col_widths)))
    print("-" * (sum(col_widths) + 2 * (len(header) - 1)))
    for row in rows:
        cells = []
        for h, w in zip(header, col_widths):
            v = row.get(h, "")
            if isinstance(v, float):
                cells.append(f"{v:>10.4f}")
            else:
                cells.append(str(v).ljust(w))
        print("  ".join(cells))
