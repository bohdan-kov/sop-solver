"""SOP Solver — головна точка входу та CLI-меню.

Програмний продукт для задачі оптимізації послідовності пайки на платі
з обмеженнями порядку (Sequential Ordering Problem, SOP).

Запуск:
    python3 main.py
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from src.algorithms.aco import ACOParams, aco_solve, aco_solve_average
from src.algorithms.brute import brute_force_solve
from src.algorithms.greedy import greedy_solve
from src.data_types import SOPInstance, SOPResult
from src.experiments import (experiment_beta, experiment_n,
                             experiment_n_stag,
                             experiment_param_series)
from src.file_io import load_instance, save_instance, save_results_table
from src.generator import generate_instance
from src.report import format_route, print_solve_results, print_table
from src.visualizer import (plot_calibration_beta,
                            plot_calibration_n_stag,
                            plot_n_series, plot_param_series,
                            visualize_route)


SAVED_DIR = "saved_tasks"
RESULTS_DIR = "results"


# ============================================================
#                      ВВЕДЕННЯ ДАНИХ
# ============================================================

def menu_input_instance(state: dict) -> None:
    print("\n" + "─" * 50)
    print("Підменю для внесення даних задачі.\n")
    print("Доступні опції:")
    print("1. Внести дані вручну")
    print("2. Згенерувати дані випадковим чином")
    print("3. Зчитати дані з файлу")
    print("0. Повернутись у головне меню\n")
    choice = input("Введіть число: ").strip()

    if choice == "1":
        _input_manual(state)
    elif choice == "2":
        _input_generate(state)
    elif choice == "3":
        _input_from_file(state)


def _input_manual(state: dict) -> None:
    print("\nВнесення даних вручну.\n")
    try:
        n = int(input("Введіть кількість позицій пайки n: "))
        coords = []
        for i in range(1, n + 1):
            line = input(f"  Координати позиції {i} (x y): ").split()
            coords.append([float(line[0]), float(line[1])])
        s_line = input("Координати стартової позиції s (x y) [за замовч. 0 0]: ").strip()
        if s_line:
            sx, sy = map(float, s_line.split())
            s = np.array([sx, sy])
        else:
            s = np.array([0.0, 0.0])
        chains_input = input(
            "Ланцюжки передування у форматі '1 3 5; 2 4' "
            "(порожнє — без обмежень): "
        ).strip()
        chains = []
        if chains_input:
            for c_str in chains_input.split(";"):
                chain = [int(x) for x in c_str.split()]
                if len(chain) >= 2:
                    chains.append(chain)
        inst = SOPInstance(
            n=n,
            coords=np.array(coords, dtype=float),
            s=s,
            chains=chains,
            p=len({v for c in chains for v in c}) / n if chains else 0.0,
        )
        state["instance"] = inst
        print("\nНові дані задачі збережено успішно!")
    except (ValueError, IndexError) as e:
        print(f"Помилка введення: {e}")


def _input_generate(state: dict) -> None:
    print("\nГенерація даних задачі.")
    try:
        n = int(input("Введіть кількість позицій пайки (n, рекомендовано 30): "))
        X_max = float(input("Введіть X_max (рекомендовано 100): "))
        Y_max = float(input("Введіть Y_max (рекомендовано 100): "))
        p_str = input("Введіть щільність обмежень (0..1, рекомендовано 0.4): ")
        p = float(p_str)
        L_min = int(input("Введіть мінімальну довжину ланцюжка L_min (рекомендовано 2): "))
        L_max = int(input("Введіть максимальну довжину ланцюжка L_max (рекомендовано 4): "))
        seed_str = input("Введіть зерно генератора seed (рекомендовано 42): ")
        seed = int(seed_str) if seed_str.strip() else None

        inst = generate_instance(
            n=n, X_max=X_max, Y_max=Y_max, p=p,
            L_min=L_min, L_max=L_max, seed=seed,
        )
        state["instance"] = inst

        print(f"\nЗгенеровано задачу: n={n}, p={p}. З ланцюжками.")
        save_q = input("\nЗберегти задачу у файл? (так/ні): ").strip().lower()
        if save_q in ("так", "yes", "y", "т"):
            default = f"task_n{n}.json"
            fname = input(f"Ім'я файлу [{default}]: ").strip() or default
            path = os.path.join(SAVED_DIR, fname)
            save_instance(inst, path)
            print(f"Збережено: {path}")
    except (ValueError, IndexError) as e:
        print(f"Помилка введення: {e}")


def _input_from_file(state: dict) -> None:
    path = input("\nВведіть шлях до файлу JSON: ").strip()
    if not os.path.exists(path):
        print(f"Файл не знайдено: {path}")
        return
    try:
        inst = load_instance(path)
        state["instance"] = inst
        print(f"\nЗавантажено: n={inst.n}, p={inst.p}. Ланцюжків: {len(inst.chains)}.")
        print("Нові дані задачі збережено успішно!")
    except Exception as e:
        print(f"Помилка завантаження: {e}")


# ============================================================
#                      РОЗВ'ЯЗАННЯ ЗАДАЧІ
# ============================================================

def menu_solve(state: dict) -> None:
    inst: Optional[SOPInstance] = state.get("instance")
    if inst is None:
        print("\nСпочатку задайте дані задачі (опція 1).")
        return

    print("\n" + "─" * 50)
    print("Розв'язання задачі всіма розробленими методами.\n")
    print(f"Задача: n={inst.n}, кількість ланцюжків={len(inst.chains)}.")

    # параметри АМК — типові, з можливістю замінити
    R_aco_str = input("Кількість незалежних запусків АМК R_aco "
                      "[за замовч. 30]: ").strip()
    R_aco = int(R_aco_str) if R_aco_str else 30

    print("\nРозв'язуємо жадібним алгоритмом...")
    g_res = greedy_solve(inst)
    print(f"Маршрут: {format_route(g_res.route)}")
    print(f"Час маршруту T_g = {g_res.T:.4f}")
    print(f"Час виконання t = {g_res.t_ms:.4f} мс")

    print(f"\nРозв'язуємо АМК ({R_aco} незалежних запусків)...")
    params = ACOParams()
    best, stats = aco_solve_average(inst, params, R_aco=R_aco, base_seed=0)
    print(f"Маршрут (рекордний): {format_route(best.route)}")
    print(f"Час маршруту T_a = {best.T:.4f}")
    print(f"Час виконання середній t = {stats['t_avg_ms']:.4f} мс")

    print_solve_results(g_res, best, aco_stats=stats)

    save_q = input("\nЗберегти результати у файл? (так/ні): ").strip().lower()
    if save_q in ("так", "yes", "y", "т"):
        default = f"results_n{inst.n}.txt"
        fname = input(f"Введіть назву файлу [{default}]: ").strip() or default
        path = os.path.join(RESULTS_DIR, fname)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Задача: n={inst.n}, p={inst.p}\n")
            f.write(f"Жадібний:  T_g = {g_res.T:.4f},  t = {g_res.t_ms:.4f} мс\n")
            f.write(f"АМК:       T_a = {best.T:.4f},  t_avg = {stats['t_avg_ms']:.4f} мс\n")
            f.write(f"           T_avg = {stats['T_avg']:.4f}, D = {stats['D']:.4f}\n")
            if g_res.T > 0:
                d = (g_res.T - best.T) / g_res.T
                f.write(f"delta = {d:.4f}\n")
            f.write(f"Маршрут жадібного: {g_res.route}\n")
            f.write(f"Маршрут АМК:       {best.route}\n")
        print(f"Дані записано: {path}")

    viz_q = input("\nЗберегти візуалізацію маршрутів (PNG)? (так/ні): ").strip().lower()
    if viz_q in ("так", "yes", "y", "т"):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        f1 = os.path.join(RESULTS_DIR, f"route_greedy_n{inst.n}.png")
        f2 = os.path.join(RESULTS_DIR, f"route_aco_n{inst.n}.png")
        visualize_route(inst, g_res, f1,
                        title=f"Жадібний — n={inst.n}, T={g_res.T:.2f}")
        visualize_route(inst, best, f2,
                        title=f"АМК — n={inst.n}, T={best.T:.2f}")
        print(f"Збережено: {f1}")
        print(f"Збережено: {f2}")


# ============================================================
#                       ЕКСПЕРИМЕНТИ
# ============================================================

def menu_experiments(state: dict) -> None:
    print("\n" + "─" * 50)
    print("Підменю для проведення експериментів.\n")
    print("Доступні дослідження:")
    print("1. Калібрування N_stag (експеримент 1)")
    print("2. Калібрування β (експеримент 2)")
    print("3. Вплив параметрів задачі p, L (експеримент 3)")
    print("4. Вплив розмірності n (експеримент 4 — 2 в 1)")
    print("0. Повернутись у головне меню\n")
    choice = input("Введіть число: ").strip()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if choice == "1":
        _exp_n_stag()
    elif choice == "2":
        _exp_beta()
    elif choice == "3":
        _exp_param_series()
    elif choice == "4":
        _exp_n()


def _parse_int_list(prompt: str, default: list[int]) -> list[int]:
    s = input(prompt).strip()
    if not s:
        return default
    return [int(x) for x in s.replace(",", " ").split()]


def _parse_float_list(prompt: str, default: list[float]) -> list[float]:
    s = input(prompt).strip()
    if not s:
        return default
    return [float(x) for x in s.replace(",", " ").split()]


def _exp_n_stag() -> None:
    print("\nЕксперимент 1. Калібрування N_stag.")
    n = int(input("Розмірність задачі n [30]: ").strip() or 30)
    pis = _parse_int_list(
        f"Множина значень π (через пробіл) [за замовч.: {[5*n,10*n,20*n,50*n,100*n]}]: ",
        [5 * n, 10 * n, 20 * n, 50 * n, 100 * n],
    )
    R = int(input("Кількість задач R [10]: ").strip() or 10)
    R_aco = int(input("Кількість запусків АМК R_aco [10]: ").strip() or 10)

    print(f"\nЗапуск експерименту 1 (R={R}, R_aco={R_aco})...")
    rows = experiment_n_stag(pi_values=pis, R=R, R_aco=R_aco, n=n,
                             progress_cb=print)
    print_table(rows, header=["pi", "T_avg", "t_avg_ms"])
    save_results_table(rows,
                       os.path.join(RESULTS_DIR, "exp1_n_stag_results.txt"),
                       header=["pi", "T_avg", "t_avg_ms"])
    plot_calibration_n_stag(rows,
                            os.path.join(RESULTS_DIR,
                                         "exp1_n_stag_graph.png"))
    print(f"\nТаблиця збережена: results/exp1_n_stag_results.txt")
    print(f"Графік збережено:  results/exp1_n_stag_graph.png")


def _exp_beta() -> None:
    print("\nЕксперимент 2. Калібрування β.")
    n = int(input("Розмірність задачі n [30]: ").strip() or 30)
    betas = _parse_float_list(
        "Множина значень β [за замовч.: 0.5 1 2 3 4 5]: ",
        [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    R = int(input("Кількість задач R [10]: ").strip() or 10)
    R_aco = int(input("Кількість запусків АМК R_aco [10]: ").strip() or 10)

    print(f"\nЗапуск експерименту 2 (R={R}, R_aco={R_aco})...")
    rows = experiment_beta(beta_values=betas, R=R, R_aco=R_aco, n=n,
                           progress_cb=print)
    print_table(rows, header=["beta", "T_avg", "t_avg_ms"])
    save_results_table(rows,
                       os.path.join(RESULTS_DIR, "exp2_beta_results.txt"),
                       header=["beta", "T_avg", "t_avg_ms"])
    plot_calibration_beta(rows,
                          os.path.join(RESULTS_DIR, "exp2_beta_graph.png"))
    print(f"\nТаблиця збережена: results/exp2_beta_results.txt")
    print(f"Графік збережено:  results/exp2_beta_graph.png")


def _exp_param_series() -> None:
    print("\nЕксперимент 3. Вплив параметрів задачі.")
    print("  1) p — щільність обмежень")
    print("  2) L — середня довжина ланцюжка")
    sub = input("Який параметр досліджувати? [1]: ").strip() or "1"
    n = int(input("Розмірність задачі n [30]: ").strip() or 30)
    R = int(input("Кількість задач R [15]: ").strip() or 15)
    R_aco = int(input("Кількість запусків АМК R_aco [10]: ").strip() or 10)

    if sub == "1":
        param_name = "p"
        values = _parse_float_list("Значення p [0 0.2 0.4 0.6 0.8]: ",
                                   [0.0, 0.2, 0.4, 0.6, 0.8])
    else:
        param_name = "L"
        values = _parse_int_list("Значення L [2 3 5]: ", [2, 3, 5])

    print(f"\nЗапуск експерименту 3 ({param_name}, R={R}, R_aco={R_aco})...")
    rows = experiment_param_series(param_name=param_name,
                                    param_values=values,
                                    R=R, R_aco=R_aco, n=n,
                                    progress_cb=print)
    header = [param_name, "T_g_avg", "T_a_avg", "t_g_avg_ms",
              "t_a_avg_ms", "delta_avg", "w"]
    print_table(rows, header=header)
    save_results_table(rows,
                       os.path.join(RESULTS_DIR,
                                    f"exp3_{param_name}_results.txt"),
                       header=header)
    plot_param_series(rows, param_name,
                      os.path.join(RESULTS_DIR,
                                   f"exp3_{param_name}_graph.png"))
    print(f"\nТаблиця збережена: results/exp3_{param_name}_results.txt")
    print(f"Графік збережено:  results/exp3_{param_name}_graph.png")


def _exp_n() -> None:
    print("\nЕксперимент 4. Вплив розмірності n (точність + час).")
    ns = _parse_int_list("Значення n [6 10 15 20 30]: ",
                         [6, 10, 15, 20, 30])
    R = int(input("Кількість задач R [10]: ").strip() or 10)
    R_aco = int(input("Кількість запусків АМК R_aco [10]: ").strip() or 10)

    print(f"\nЗапуск експерименту 4 (R={R}, R_aco={R_aco})...")
    print("Це може зайняти кілька хвилин...")
    rows = experiment_n(n_values=ns, R=R, R_aco=R_aco, progress_cb=print)
    header = ["n", "T_g_avg", "T_a_avg", "t_g_avg_ms", "t_a_avg_ms",
              "delta_avg", "w", "gap_avg"]
    print_table(rows, header=header)
    save_results_table(rows,
                       os.path.join(RESULTS_DIR, "exp4_n_results.txt"),
                       header=header)
    plot_n_series(
        rows,
        fname_quality=os.path.join(RESULTS_DIR, "exp4_n_quality.png"),
        fname_time=os.path.join(RESULTS_DIR, "exp4_n_time.png"),
        fname_gap=os.path.join(RESULTS_DIR, "exp4_n_gap.png"),
    )
    print(f"\nТаблиця збережена: results/exp4_n_results.txt")
    print(f"Графіки збережено: results/exp4_n_quality.png,")
    print(f"                   results/exp4_n_time.png,")
    print(f"                   results/exp4_n_gap.png")


# ============================================================
#                      ВИВЕДЕННЯ ДАНИХ
# ============================================================

def menu_output(state: dict) -> None:
    inst: Optional[SOPInstance] = state.get("instance")
    if inst is None:
        print("\nСпочатку задайте дані задачі (опція 1).")
        return

    print("\n" + "─" * 50)
    print("Підменю для виведення даних задачі.\n")
    print("Доступні опції:")
    print("1. Вивести дані на екран")
    print("2. Записати дані до файлу")
    print("0. Повернутись у головне меню\n")
    choice = input("Введіть число: ").strip()

    if choice == "1":
        _print_instance(inst)
    elif choice == "2":
        _save_instance_dialog(inst)


def _print_instance(inst: SOPInstance) -> None:
    print("\nВиводимо дані на екран...\n")
    print(f"Кількість позицій пайки n = {inst.n}")
    print(f"Стартова позиція s = ({inst.s[0]}, {inst.s[1]})")
    print(f"Щільність обмежень p = {inst.p}")
    print(f"Ланцюжків передування: {len(inst.chains)}")
    for i, c in enumerate(inst.chains, 1):
        arrow = " → ".join(str(v) for v in c)
        print(f"  P{i}: {arrow}")
    print("Координати позицій:")
    show_n = min(inst.n, 30)
    for i in range(show_n):
        x, y = inst.coords[i]
        print(f"  {i+1:>2}: x={x:6.2f}  y={y:6.2f}")
    if inst.n > show_n:
        print(f"  ... (показано перші {show_n} з {inst.n})")


def _save_instance_dialog(inst: SOPInstance) -> None:
    default = f"task_n{inst.n}.json"
    fname = input(f"\nВведіть назву файлу [{default}]: ").strip() or default
    path = os.path.join(SAVED_DIR, fname)
    if os.path.exists(path):
        ans = input(f"Файл вже існує. Перезаписати? (так/ні): ").strip().lower()
        if ans not in ("так", "yes", "y", "т"):
            print("Скасовано.")
            return
    save_instance(inst, path)
    print(f"Записуємо дані до файлу... Записано!")
    print(f"Шлях: {path}")


# ============================================================
#                       ГОЛОВНЕ МЕНЮ
# ============================================================

def main() -> None:
    state: dict = {"instance": None}
    while True:
        inst = state.get("instance")
        status = "Задачу не задано." if inst is None else (
            f"Задача n={inst.n}, ланцюжків={len(inst.chains)}."
        )
        print("\n" + "═" * 50)
        print("Головне меню.")
        print(f"Статус задачі: {status}\n")
        print("Доступні опції:")
        print("1. Внести дані задачі")
        print("2. Розв'язати задачу всіма розробленими методами")
        print("3. Провести експерименти")
        print("4. Вивести дані задачі")
        print("0. Завершити роботу\n")

        choice = input("Введіть число: ").strip()

        if choice == "1":
            menu_input_instance(state)
        elif choice == "2":
            menu_solve(state)
        elif choice == "3":
            menu_experiments(state)
        elif choice == "4":
            menu_output(state)
        elif choice == "0":
            print("\nЗавершення роботи. До побачення!")
            sys.exit(0)
        else:
            print("Невідома опція.")


if __name__ == "__main__":
    main()
