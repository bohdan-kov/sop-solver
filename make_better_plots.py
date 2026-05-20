"""Покращені графіки для всіх 4 експериментів.

Будує нові версії графіків з анотаціями, виділенням обраних значень,
другою віссю часу, теоретичними кривими.

Запуск:
    cd sop-solver
    python3 make_better_plots.py
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# ============================================================
# Загальний стиль
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_GREEDY = "#2E5C8A"      # темно-синій
COLOR_ACO = "#C23B22"          # темно-червоний
COLOR_TIME = "#7F8C8D"         # сірий для другої осі
COLOR_HIGHLIGHT = "#27AE60"    # зелений для виділення оптимуму
COLOR_THEORY = "#95A5A6"       # світло-сірий для теоретичних кривих
COLOR_WIN_BAND = "#A8D5BA"     # пастельний зелений для «зони виграшу»

RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "_better_plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Парсинг таблиць результатів
# ============================================================

def parse_results_table(path: str) -> List[dict]:
    """Розпарсити exp*_results.txt у список словників."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]
    if len(lines) < 3:
        return []
    header = lines[0].split()
    rows = []
    for ln in lines[2:]:
        parts = ln.split()
        if not parts:
            continue
        row = {}
        for k, v in zip(header, parts):
            if v == "None":
                row[k] = None
            else:
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
        rows.append(row)
    return rows


# ============================================================
# Експ. 1 — Калібрування N_stag
# ============================================================

def plot_exp1():
    rows = parse_results_table(
        os.path.join(RESULTS_DIR, "1_mykhailova_n_stag",
                     "exp1_n_stag_results.txt")
    )
    pi = np.array([r["pi"] for r in rows])
    T = np.array([r["T_avg"] for r in rows])
    t = np.array([r["t_avg_ms"] for r in rows]) / 1000.0  # у секунди

    fig, ax1 = plt.subplots(figsize=(10, 7.2))

    # Якість маршруту (T)
    ln1 = ax1.plot(pi, T, "o-", color=COLOR_ACO, lw=2, ms=9,
                   label="Середнє T (якість маршруту)", zorder=3)
    for x, y in zip(pi, T):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9,
                     color=COLOR_ACO)

    # Горизонтальна лінія плато
    T_plateau = T[-1]
    ax1.axhline(T_plateau, color=COLOR_HIGHLIGHT, ls=":", lw=1.5,
                alpha=0.7, label=f"Плато T ≈ {T_plateau:.3f}")

    # Виділення обраного π* = 5·n = 150
    pi_star = 150
    T_star = T[0]
    ax1.scatter([pi_star], [T_star], s=300, marker="*",
                color=COLOR_HIGHLIGHT, ec="black", lw=1.5, zorder=5,
                label=f"Обране N*_stag = 5·n = {pi_star}")

    ax1.set_xlabel("Значення параметра N_stag (π = κ·n при n=30)")
    ax1.set_ylabel("Середнє значення цільової функції T",
                   color=COLOR_ACO)
    ax1.tick_params(axis="y", labelcolor=COLOR_ACO)

    # Друга вісь — час роботи
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ln2 = ax2.plot(pi, t, "s--", color=COLOR_TIME, lw=1.8, ms=8,
                   alpha=0.85, label="Середній час прогону (с)",
                   zorder=2)
    for x, y in zip(pi, t):
        ax2.annotate(f"{y:.1f}с", (x, y), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9,
                     color=COLOR_TIME)
    ax2.set_ylabel("Середній час одного прогону АМК, секунд",
                   color=COLOR_TIME)
    ax2.tick_params(axis="y", labelcolor=COLOR_TIME)
    ax2.grid(False)

    # Об'єднана легенда — знизу під графіком у виділеній області
    lines = ln1 + ln2 + [ax1.lines[1]] + [ax1.collections[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", framealpha=0.95,
               bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)

    plt.title("Експеримент 1: калібрування N_stag\n"
              "якість виходить на плато при π = 5·n, "
              "збільшення π тільки витрачає час",
              pad=15)
    plt.subplots_adjust(bottom=0.22)
    out = os.path.join(OUT_DIR, "exp1_n_stag_better.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ============================================================
# Експ. 2 — Калібрування β
# ============================================================

def plot_exp2():
    rows = parse_results_table(
        os.path.join(RESULTS_DIR, "2_harmash_beta",
                     "exp2_beta_results.txt")
    )
    beta = np.array([r["beta"] for r in rows])
    T = np.array([r["T_avg"] for r in rows])
    t = np.array([r["t_avg_ms"] for r in rows]) / 1000.0

    fig, ax1 = plt.subplots(figsize=(10, 7.2))

    ln1 = ax1.plot(beta, T, "o-", color=COLOR_ACO, lw=2, ms=9,
                   label="Середнє T (якість маршруту)", zorder=3)
    for x, y in zip(beta, T):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=9,
                     color=COLOR_ACO)

    # Виділення обраного значення β* = 2 (як у курсовій)
    beta_star = 2.0
    idx_star = int(np.where(beta == beta_star)[0][0])
    T_star = T[idx_star]
    ax1.scatter([beta_star], [T_star], s=350, marker="*",
                color=COLOR_HIGHLIGHT, ec="black", lw=1.5, zorder=5,
                label=f"Обране β* = {beta_star:.0f} (T = {T_star:.2f})")

    # Зони пояснення країв (без накладання на легенду — знижено)
    ax1.axvspan(0, 0.75, alpha=0.08, color="blue")
    ax1.text(0.5, 64,
             "β/α мале:\nрівноймовірний\nперехід",
             fontsize=9, ha="center", style="italic", color="#444")
    ax1.axvspan(4.25, 5.5, alpha=0.08, color="orange")
    ax1.text(4.75, 64,
             "β/α велике:\nАМК ≈ багатократний\nжадібний",
             fontsize=9, ha="center", style="italic", color="#444")

    ax1.set_xlabel("Значення параметра β (при α = 1)")
    ax1.set_ylabel("Середнє значення цільової функції T",
                   color=COLOR_ACO)
    ax1.tick_params(axis="y", labelcolor=COLOR_ACO)

    # Друга вісь — час роботи
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ln2 = ax2.plot(beta, t, "s--", color=COLOR_TIME, lw=1.8, ms=8,
                   alpha=0.85, label="Середній час прогону (с)",
                   zorder=2)
    ax2.set_ylabel("Середній час одного прогону АМК, секунд",
                   color=COLOR_TIME)
    ax2.tick_params(axis="y", labelcolor=COLOR_TIME)
    ax2.grid(False)

    lines = ln1 + ln2 + [ax1.collections[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", framealpha=0.95,
               bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True)

    plt.title("Експеримент 2: калібрування β при α = 1\n"
              "мінімум T у точці β = 2…3, далі крива виходить на плато",
              pad=15)
    plt.subplots_adjust(bottom=0.22)
    out = os.path.join(OUT_DIR, "exp2_beta_better.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ============================================================
# Експ. 3 — Вплив p та L
# ============================================================

def plot_exp3_param(param_name: str, fname: str):
    rows = parse_results_table(
        os.path.join(RESULTS_DIR, "3_bakunets_p_and_L",
                     f"exp3_{param_name}_results.txt")
    )
    x = np.array([r[param_name] for r in rows])
    Tg = np.array([r["T_g_avg"] for r in rows])
    Ta = np.array([r["T_a_avg"] for r in rows])
    delta = np.array([r["delta_avg"] for r in rows]) * 100
    w = np.array([r["w"] for r in rows]) * 100

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6.2))

    # Панель 1: T_g та T_a з зоною виграшу
    ax1.fill_between(x, Ta, Tg, color=COLOR_WIN_BAND, alpha=0.6,
                     label="Виграш АМК")
    ax1.plot(x, Tg, "o-", color=COLOR_GREEDY, lw=2, ms=9,
             label="Жадібний T̄_g")
    ax1.plot(x, Ta, "s--", color=COLOR_ACO, lw=2, ms=9,
             label="АМК T̄_a")
    for xi, yi in zip(x, Tg):
        ax1.annotate(f"{yi:.1f}", (xi, yi),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color=COLOR_GREEDY)
    for xi, yi in zip(x, Ta):
        ax1.annotate(f"{yi:.1f}", (xi, yi),
                     textcoords="offset points", xytext=(0, -16),
                     ha="center", fontsize=8, color=COLOR_ACO)
    ax1.set_xlabel(f"Параметр задачі: {param_name}")
    ax1.set_ylabel("Середній час маршруту T")
    ax1.set_title("Якість маршруту T̄", fontsize=12)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=3, framealpha=0.95, frameon=True, fontsize=9)
    # маленький запас знизу, щоб підписи не виходили за рамку
    y_lo = min(Ta.min(), Tg.min()) - 2
    y_hi = max(Ta.max(), Tg.max()) + 2
    ax1.set_ylim(y_lo, y_hi)

    # Панель 2: відносна різниця δ
    ax2.plot(x, delta, "^-", color=COLOR_HIGHLIGHT, lw=2, ms=10)
    ax2.fill_between(x, 0, delta, color=COLOR_HIGHLIGHT, alpha=0.15)
    for xi, yi in zip(x, delta):
        ax2.annotate(f"{yi:.1f}%", (xi, yi),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, fontweight="bold",
                     color="#1F7D45")
    ax2.set_xlabel(f"Параметр задачі: {param_name}")
    ax2.set_ylabel("Середня парна різниця δ̄, %")
    ax2.set_title("Виграш АМК над жадібним (δ̄)", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.axhline(0, color="gray", lw=0.8, ls=":")

    # Панель 3: частка перемог w
    bars = ax3.bar(x, w, width=(x[1]-x[0])*0.6 if len(x) > 1 else 0.5,
                   color=COLOR_ACO, alpha=0.75, edgecolor="black",
                   linewidth=0.8)
    for xi, yi in zip(x, w):
        ax3.annotate(f"{yi:.0f}%", (xi, yi),
                     textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=10, fontweight="bold")
    ax3.axhline(100, color=COLOR_HIGHLIGHT, ls=":", lw=1.5,
                label="100% — АМК виграє на всіх задачах")
    ax3.set_xlabel(f"Параметр задачі: {param_name}")
    ax3.set_ylabel("Частка перемог АМК w, %")
    ax3.set_title("Стабільність переваги (w)", fontsize=12)
    ax3.set_ylim(0, 110)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               framealpha=0.95, frameon=True, fontsize=9)

    fig.suptitle(f"Експеримент 3: вплив параметра {param_name} "
                 f"на ефективність обох алгоритмів",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(bottom=0.22, top=0.85, wspace=0.4)
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_exp3():
    plot_exp3_param("p", "exp3_p_better.png")
    plot_exp3_param("L", "exp3_L_better.png")


# ============================================================
# Експ. 4 — Вплив розмірності n
# ============================================================

def plot_exp4():
    rows = parse_results_table(
        os.path.join(RESULTS_DIR, "4_koval_n", "exp4_n_results.txt")
    )
    n = np.array([r["n"] for r in rows])
    Tg = np.array([r["T_g_avg"] for r in rows])
    Ta = np.array([r["T_a_avg"] for r in rows])
    tg = np.array([r["t_g_avg_ms"] for r in rows])
    ta = np.array([r["t_a_avg_ms"] for r in rows])
    delta = np.array([r["delta_avg"] for r in rows]) * 100
    gap = np.array([r["gap_avg"] if r["gap_avg"] is not None else np.nan
                    for r in rows]) * 100

    # ---------- Графік 1: якість ----------
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.fill_between(n, Ta, Tg, color=COLOR_WIN_BAND, alpha=0.6,
                    label="Виграш АМК")
    ax.plot(n, Tg, "o-", color=COLOR_GREEDY, lw=2, ms=9,
            label="Жадібний T̄_g")
    ax.plot(n, Ta, "s--", color=COLOR_ACO, lw=2, ms=9,
            label="АМК T̄_a")
    # Підписи виграшу δ% над зоною
    for ni, tg_i, ta_i, d in zip(n, Tg, Ta, delta):
        mid_y = (tg_i + ta_i) / 2
        ax.annotate(f"δ={d:.1f}%", (ni, mid_y),
                    textcoords="offset points", xytext=(0, 0),
                    ha="center", fontsize=9, fontweight="bold",
                    color="#1F7D45",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="#27AE60", alpha=0.85))
    ax.set_xlabel("Розмірність задачі n (кількість позицій)")
    ax.set_ylabel("Середній час маршруту T")
    ax.set_title("Експеримент 4: якість маршруту залежно від n\n"
                 "АМК стабільно виграє 11–17% незалежно від розмірності",
                 pad=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, framealpha=0.95, frameon=True)
    plt.subplots_adjust(bottom=0.20)
    out = os.path.join(OUT_DIR, "exp4_n_quality_better.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")

    # ---------- Графік 2: час з теоретичними кривими ----------
    fig, ax = plt.subplots(figsize=(10, 7.2))

    # Теоретичні криві (нормуємо так, щоб збігалися з даними при n=30)
    n_smooth = np.linspace(n.min(), n.max(), 100)
    # O(n²) для жадібного: підбираємо константу так, щоб збігалося з tg[n=30]
    idx30 = int(np.where(n == 30)[0][0])
    K_g = tg[idx30] / (30 ** 2)
    theory_g = K_g * n_smooth ** 2
    # O(n² · N_iter · m_a) для АМК — m_a = n, N_iter спрацьовує реально
    # на N_stag = 5n; візьмемо ефективну константу
    K_a = ta[idx30] / (30 ** 3)
    theory_a = K_a * n_smooth ** 3

    ax.plot(n_smooth, theory_g, ":", color=COLOR_GREEDY, lw=2,
            alpha=0.6, label="Теорія жадібного: O(n²)")
    ax.plot(n_smooth, theory_a, ":", color=COLOR_ACO, lw=2,
            alpha=0.6, label="Теорія АМК: O(n²·m_a)")
    ax.plot(n, tg, "o-", color=COLOR_GREEDY, lw=2.2, ms=10,
            label="Жадібний — практика")
    ax.plot(n, ta, "s-", color=COLOR_ACO, lw=2.2, ms=10,
            label="АМК — практика")

    # Підписи реальних значень — праворуч від точок, щоб не перекривати маркери
    for ni, ti in zip(n, tg):
        ax.annotate(f"{ti:.2f} мс", (ni, ti),
                    textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=COLOR_GREEDY, ha="left")
    for ni, ti in zip(n, ta):
        unit = "мс" if ti < 1000 else "с"
        val = ti if ti < 1000 else ti / 1000
        ax.annotate(f"{val:.1f} {unit}", (ni, ti),
                    textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=COLOR_ACO, ha="left")

    ax.set_yscale("log")
    ax.set_xlabel("Розмірність задачі n")
    ax.set_ylabel("Час роботи, мс (логарифмічна шкала)")
    ax.set_title("Експеримент 4: час роботи vs теоретична складність\n"
                 "практика збігається з теорією O(n²) та O(n²·m_a)",
                 pad=15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, framealpha=0.95, frameon=True)
    ax.grid(True, which="both", alpha=0.25)
    plt.subplots_adjust(bottom=0.22)
    out = os.path.join(OUT_DIR, "exp4_n_time_better.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")

    # ---------- Графік 3: gap як barplot ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = ~np.isnan(gap)
    n_valid = n[valid]
    gap_valid = gap[valid]
    bars = ax.bar(n_valid.astype(int).astype(str), gap_valid,
                  color=COLOR_HIGHLIGHT, alpha=0.8, edgecolor="black",
                  linewidth=1.0, width=0.5)
    for ni, gi in zip(n_valid, gap_valid):
        ax.annotate(f"{gi:.3f}%", (str(int(ni)), gi),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=11, fontweight="bold",
                    color="#1F7D45")
    ax.set_xlabel("Розмірність задачі n")
    ax.set_ylabel("Відхилення АМК від оптимуму, %")
    ax.set_title("Експеримент 4: gap АМК від справжнього оптимуму\n"
                 "(порівняння з повним перебором для n ≤ 10)",
                 pad=15)
    ax.set_ylim(0, max(gap_valid) * 1.4 if len(gap_valid) > 0 else 1)
    ax.axhline(0, color="gray", lw=0.8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "exp4_n_gap_better.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    print("Будуємо покращені графіки в", OUT_DIR)
    print()
    print("Експеримент 1 — N_stag:")
    plot_exp1()
    print()
    print("Експеримент 2 — β:")
    plot_exp2()
    print()
    print("Експеримент 3 — p та L:")
    plot_exp3()
    print()
    print("Експеримент 4 — n:")
    plot_exp4()
    print()
    print("Готово. Усі файли в", OUT_DIR)
