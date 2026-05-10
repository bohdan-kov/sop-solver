"""Серіалізація / десеріалізація даних задачі та результатів."""
from __future__ import annotations

import json
import os
from typing import Optional

from .data_types import SOPInstance, SOPResult


def save_instance(inst: SOPInstance, filepath: str,
                  greedy: Optional[SOPResult] = None,
                  aco: Optional[SOPResult] = None) -> None:
    data = inst.to_dict()
    if greedy is not None:
        data["greedy"] = greedy.to_dict()
    if aco is not None:
        data["aco"] = aco.to_dict()
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_instance(filepath: str) -> SOPInstance:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SOPInstance.from_dict(data)


def save_results_table(rows: list[dict], filepath: str,
                       header: list[str]) -> None:
    """Записати результати серії експериментів у текстовий файл-таблицю."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    col_widths = [max(len(h), 10) for h in header]
    lines = []
    lines.append("  ".join(h.ljust(w) for h, w in zip(header, col_widths)))
    lines.append("-" * (sum(col_widths) + 2 * (len(header) - 1)))
    for row in rows:
        cells = []
        for h, w in zip(header, col_widths):
            v = row.get(h, "")
            if isinstance(v, float):
                cells.append(f"{v:>10.4f}")
            else:
                cells.append(str(v).ljust(w))
        lines.append("  ".join(cells))
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
