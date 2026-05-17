"""Запуск усіх 4 експериментів послідовно з параметрами з розділу 3.

Параметри відповідають плану в курсовій:
  Експ.1-3: R = 30, R_aco = 30;
  Експ.4:   R = 20, R_aco = 30 (для великих n зменшено через час обчислень,
            стор. 53);
  Експ.4:   n ∈ {6, 10, 15, 20, 30, 50, 80, 100} (повна сітка за планом).

Використання:
    python3 run_all_experiments.py            # повний прогін за планом
    python3 run_all_experiments.py --parallel # 4 експерименти паралельно
    python3 run_all_experiments.py --quick    # швидкий smoke-тест (~3 хв)
    python3 run_all_experiments.py --only 1   # тільки експ. 1 (або 2, 3, 4)
    python3 run_all_experiments.py --quick-n  # Експ.4 без n=80, 100
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# гарантуємо, що можемо імпортувати з src/ незалежно від cwd
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.algorithms.aco import ACOParams
from src.experiments import (experiment_beta, experiment_n,
                             experiment_n_stag, experiment_param_series)
from src.file_io import save_results_table
from src.report import print_table
from src.visualizer import (plot_calibration_beta, plot_calibration_n_stag,
                            plot_n_series, plot_param_series)

RESULTS_DIR = os.path.join(HERE, "results")

# Префікс для логів, який встановлюється у дочірньому процесі при
# паралельному запуску (напр. "[EXP1] "). У головному процесі — порожній.
_LOG_PREFIX = ""


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_stamp()}] {_LOG_PREFIX}{msg}", flush=True)


def _section(title: str) -> None:
    print("\n" + "=" * 70, flush=True)
    print(f"[{_stamp()}] {_LOG_PREFIX}{title}", flush=True)
    print("=" * 70, flush=True)


def run_exp1(R: int, R_aco: int) -> None:
    """Експеримент 1. Калібрування N_stag.

    План (підрозділ 3.3.2): n=30, p=0.4, L_min=2, L_max=4,
    π ∈ {5n, 10n, 20n, 50n, 100n} = {150, 300, 600, 1500, 3000}.

    Зауваження: N_iter=10000 виступає як страхувальний ліміт, щоб саме
    умова "Nstag без поліпшення" керувала зупинкою — тоді крива T̄(π)
    реально показує насичення замість плоскої лінії на N_iter.
    """
    _section("ЕКСПЕРИМЕНТ 1: калібрування N_stag")
    n = 30
    pis = [5 * n, 10 * n, 20 * n, 50 * n, 100 * n]   # 150, 300, 600, 1500, 3000

    # N_iter велике, щоб умова Nstag реально визначала зупинку
    aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                           m_a=None, N_iter=10000, N_stag=30)

    _log(f"n={n}, p=0.4, L_min=2, L_max=4, R={R}, R_aco={R_aco}")
    _log(f"π values = {pis}")
    t0 = time.perf_counter()
    rows = experiment_n_stag(pi_values=pis, R=R, R_aco=R_aco, n=n,
                             p=0.4, L_min=2, L_max=4,
                             aco_params=aco_params,
                             progress_cb=_log)
    _log(f"Експеримент 1 завершено за {time.perf_counter() - t0:.1f} с")

    header = ["pi", "T_avg", "t_avg_ms"]
    print_table(rows, header=header)
    save_results_table(rows,
                       os.path.join(RESULTS_DIR, "exp1_n_stag_results.txt"),
                       header=header)
    plot_calibration_n_stag(
        rows, os.path.join(RESULTS_DIR, "exp1_n_stag_graph.png"))
    _log("Збережено: exp1_n_stag_results.txt, exp1_n_stag_graph.png")


def run_exp2(R: int, R_aco: int) -> None:
    """Експеримент 2. Калібрування β при α=1.

    План (підрозділ 3.3.3): n=30, p=0.4, L_min=2, L_max=4,
    β ∈ {0.5, 1, 2, 3, 4, 5}.
    """
    _section("ЕКСПЕРИМЕНТ 2: калібрування β")
    n = 30
    betas = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                           m_a=None, N_iter=10000, N_stag=30)

    _log(f"n={n}, p=0.4, L_min=2, L_max=4, R={R}, R_aco={R_aco}")
    _log(f"β values = {betas}")
    t0 = time.perf_counter()
    rows = experiment_beta(beta_values=betas, R=R, R_aco=R_aco, n=n,
                           p=0.4, L_min=2, L_max=4,
                           aco_params=aco_params,
                           progress_cb=_log)
    _log(f"Експеримент 2 завершено за {time.perf_counter() - t0:.1f} с")

    header = ["beta", "T_avg", "t_avg_ms"]
    print_table(rows, header=header)
    save_results_table(rows,
                       os.path.join(RESULTS_DIR, "exp2_beta_results.txt"),
                       header=header)
    plot_calibration_beta(rows,
                          os.path.join(RESULTS_DIR, "exp2_beta_graph.png"))
    _log("Збережено: exp2_beta_results.txt, exp2_beta_graph.png")


def run_exp3(R: int, R_aco: int) -> None:
    """Експеримент 3. Вплив параметрів задачі (p та L).

    План (підрозділ 3.3.4):
      p ∈ {0; 0.2; 0.4; 0.6; 0.8} при фіксованому L (Lmin=2, Lmax=4),
      L ∈ {2; 3; 5} при фіксованому p=0.4.
    """
    _section("ЕКСПЕРИМЕНТ 3a: вплив p (щільність обмежень)")
    n = 30
    ps = [0.0, 0.2, 0.4, 0.6, 0.8]
    aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                           m_a=None, N_iter=10000, N_stag=30)

    _log(f"n={n}, p={ps}, R={R}, R_aco={R_aco}")
    t0 = time.perf_counter()
    rows_p = experiment_param_series(
        param_name="p", param_values=ps,
        R=R, R_aco=R_aco, n=n,
        L_min=2, L_max=4,
        aco_params=aco_params,
        progress_cb=_log,
    )
    _log(f"Експеримент 3a завершено за {time.perf_counter() - t0:.1f} с")

    header = ["p", "T_g_avg", "T_a_avg", "t_g_avg_ms", "t_a_avg_ms",
              "delta_avg", "w"]
    print_table(rows_p, header=header)
    save_results_table(rows_p,
                       os.path.join(RESULTS_DIR, "exp3_p_results.txt"),
                       header=header)
    plot_param_series(rows_p, "p",
                      os.path.join(RESULTS_DIR, "exp3_p_graph.png"))
    _log("Збережено: exp3_p_results.txt, exp3_p_graph.png")

    _section("ЕКСПЕРИМЕНТ 3b: вплив L (середня довжина ланцюжка)")
    Ls = [2, 3, 5]
    _log(f"n={n}, p=0.4, L={Ls}, R={R}, R_aco={R_aco}")
    t0 = time.perf_counter()
    rows_L = experiment_param_series(
        param_name="L", param_values=Ls,
        R=R, R_aco=R_aco, n=n,
        p=0.4,
        aco_params=aco_params,
        progress_cb=_log,
    )
    _log(f"Експеримент 3b завершено за {time.perf_counter() - t0:.1f} с")

    header = ["L", "T_g_avg", "T_a_avg", "t_g_avg_ms", "t_a_avg_ms",
              "delta_avg", "w"]
    print_table(rows_L, header=header)
    save_results_table(rows_L,
                       os.path.join(RESULTS_DIR, "exp3_L_results.txt"),
                       header=header)
    plot_param_series(rows_L, "L",
                      os.path.join(RESULTS_DIR, "exp3_L_graph.png"))
    _log("Збережено: exp3_L_results.txt, exp3_L_graph.png")


def run_exp4(R: int, R_aco: int, full: bool) -> None:
    """Експеримент 4 (2-в-1). Вплив n на точність та час.

    План (підрозділ 3.3.5): n ∈ {6, 10, 15, 20, 30, 50, 80, 100}, R = 20.
    Для n ≤ 10 додатково обчислюється gap відносно brute force.

    full=False (швидкий) обрізає до {6, 10, 15, 20, 30, 50}, бо
    n=80,100 коштують десятки хвилин.
    """
    _section("ЕКСПЕРИМЕНТ 4: вплив розмірності n")
    # Курсова (стор. 53): для Експ.4 R = 20 (для великих n зменшено через
    # час обчислень). Якщо переданий R більший — приводимо до плану.
    if R > 20:
        _log(f"Експ.4: R={R} → 20 (як у плані курсової)")
        R = 20
    if full:
        ns = [6, 10, 15, 20, 30, 50, 80, 100]
    else:
        ns = [6, 10, 15, 20, 30, 50]
    aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                           m_a=None, N_iter=10000, N_stag=30)

    _log(f"n={ns}, p=0.4, L_min=2, L_max=4, R={R}, R_aco={R_aco}")
    t0 = time.perf_counter()
    rows = experiment_n(n_values=ns, R=R, R_aco=R_aco,
                        p=0.4, L_min=2, L_max=4,
                        aco_params=aco_params,
                        progress_cb=_log)
    _log(f"Експеримент 4 завершено за {time.perf_counter() - t0:.1f} с")

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
    _log("Збережено: exp4_n_results.txt, exp4_n_quality.png, "
         "exp4_n_time.png, exp4_n_gap.png")


def _run_in_process(exp_num: int, R: int, R_aco: int, full_n: bool) -> None:
    """Точка входу дочірнього процесу — запускає один експеримент.

    Виставляє _LOG_PREFIX = "[EXPn] ", щоб у спільному stdout було видно,
    рядок якого експерименту куди належить. Оскільки усі експерименти
    використовують різні бази seed (1000·i, 2000·i, 3000·i, 4000·i) і не
    діляться станом, результат паралельного запуску ідентичний
    послідовному до останнього біта.
    """
    global _LOG_PREFIX
    _LOG_PREFIX = f"[EXP{exp_num}] "
    if exp_num == 1:
        run_exp1(R, R_aco)
    elif exp_num == 2:
        run_exp2(R, R_aco)
    elif exp_num == 3:
        run_exp3(R, R_aco)
    elif exp_num == 4:
        run_exp4(R, R_aco, full=full_n)


def _run_parallel(targets: list[int], R: int, R_aco: int,
                  full_n: bool) -> None:
    """Запускає вибрані експерименти у паралельних процесах."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs: list[tuple[int, mp.Process]] = []
    for exp_num in targets:
        p = ctx.Process(
            target=_run_in_process,
            args=(exp_num, R, R_aco, full_n),
            name=f"exp{exp_num}",
        )
        p.start()
        procs.append((exp_num, p))
        _log(f"Запущено EXP{exp_num} у процесі PID={p.pid}")

    failed: list[int] = []
    try:
        for exp_num, p in procs:
            p.join()
            if p.exitcode == 0:
                _log(f"EXP{exp_num} завершено успішно")
            else:
                failed.append(exp_num)
                _log(f"EXP{exp_num} впав з exitcode={p.exitcode}")
    except KeyboardInterrupt:
        _log("Перервано користувачем — зупиняю дочірні процеси...")
        for _, p in procs:
            if p.is_alive():
                p.terminate()
        for _, p in procs:
            p.join(timeout=5)
        sys.exit(130)

    if failed:
        _log(f"Падіння у експериментах: {failed}")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Запуск усіх експериментів")
    ap.add_argument("--quick", action="store_true",
                    help="швидкий smoke-тест з мінімальними R, R_aco")
    ap.add_argument("--only", type=int, choices=[1, 2, 3, 4],
                    help="запустити тільки один експеримент")
    ap.add_argument("--quick-n", action="store_true",
                    help="для експ. 4 виключити n=80, 100 (швидше)")
    ap.add_argument("--R", type=int, default=30,
                    help="кількість задач у наборі (за замовч. 30; "
                         "для Експ.4 автоматично обмежується до 20)")
    ap.add_argument("--R-aco", type=int, default=30,
                    help="кількість прогонів АМК на задачі (за замовч. 30)")
    ap.add_argument("--parallel", action="store_true",
                    help="запустити усі 4 експерименти паралельно у "
                         "окремих процесах (результати ідентичні "
                         "послідовним; ефективно при 4+ ядрах CPU)")
    args = ap.parse_args()

    if args.quick:
        R, R_aco = 3, 3
        full_n = False
    else:
        R, R_aco = args.R, args.R_aco
        full_n = not args.quick_n

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _log(f"Старт запуску експериментів. R={R}, R_aco={R_aco}, "
         f"full_n={full_n}, only={args.only}, parallel={args.parallel}")
    t_total = time.perf_counter()

    targets = [args.only] if args.only else [1, 2, 3, 4]

    if args.parallel and len(targets) > 1:
        _run_parallel(targets, R, R_aco, full_n)
    else:
        try:
            for exp_num in targets:
                _run_in_process(exp_num, R, R_aco, full_n)
        except KeyboardInterrupt:
            _log("Перервано користувачем.")
            sys.exit(130)

    _log(f"УСІ ЕКСПЕРИМЕНТИ ЗАВЕРШЕНО за "
         f"{(time.perf_counter() - t_total) / 60:.1f} хв")


if __name__ == "__main__":
    main()
