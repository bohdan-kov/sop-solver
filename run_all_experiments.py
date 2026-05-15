"""Запуск усіх 4 експериментів послідовно з параметрами з розділу 3.

Параметри відповідають плану в курсовій з адаптацією під реалістичний
час виконання (R та R_aco зменшено з 30 до 15 для прийнятної тривалості;
це не міняє якісного характеру висновків).

Використання:
    python3 run_all_experiments.py            # повний прогін (~30-60 хв)
    python3 run_all_experiments.py --quick    # швидкий smoke-тест (~3 хв)
    python3 run_all_experiments.py --only 1   # тільки експ. 1 (або 2, 3, 4)
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


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_stamp()}] {msg}", flush=True)


def _section(title: str) -> None:
    print("\n" + "=" * 70, flush=True)
    print(f"[{_stamp()}] {title}", flush=True)
    print("=" * 70, flush=True)


def run_exp1(R: int, R_aco: int) -> None:
    """Експеримент 1. Калібрування N_stag.

    План (підрозділ 3.3.2): n=30, p=0.4, L_min=2, L_max=4,
    π ∈ {5n, 10n, 20n, 50n, 100n} = {150, 300, 600, 1500, 3000}.

    Зауваження: типовий N_iter=100 жорстко обмежить АМК незалежно від Nstag.
    Для коректного дослідження плато по Nstag тимчасово піднімаємо
    N_iter до 10000, щоб саме умова "Nstag без поліпшення" керувала
    зупинкою — тоді крива T̄(π) реально показує насичення.
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
                           m_a=None, N_iter=100, N_stag=30)

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
                           m_a=None, N_iter=100, N_stag=30)

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

    План (підрозділ 3.3.5): n ∈ {6, 10, 15, 20, 30, 50, 80, 100}.
    Для n ≤ 10 додатково обчислюється gap відносно brute force.

    full=False (швидкий) обрізає до {6, 10, 15, 20, 30, 50}, бо
    n=80,100 коштують десятки хвилин.
    """
    _section("ЕКСПЕРИМЕНТ 4: вплив розмірності n")
    if full:
        ns = [6, 10, 15, 20, 30, 50, 80, 100]
    else:
        ns = [6, 10, 15, 20, 30, 50]
    aco_params = ACOParams(alpha=1.0, beta=2.0, rho=0.5,
                           m_a=None, N_iter=100, N_stag=30)

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Запуск усіх експериментів")
    ap.add_argument("--quick", action="store_true",
                    help="швидкий smoke-тест з мінімальними R, R_aco")
    ap.add_argument("--only", type=int, choices=[1, 2, 3, 4],
                    help="запустити тільки один експеримент")
    ap.add_argument("--full-n", action="store_true",
                    help="для експ. 4 використати n до 100 (повільно)")
    ap.add_argument("--R", type=int, default=15,
                    help="кількість задач у наборі (за замовч. 15)")
    ap.add_argument("--R-aco", type=int, default=15,
                    help="кількість прогонів АМК на задачі (за замовч. 15)")
    args = ap.parse_args()

    if args.quick:
        R, R_aco = 3, 3
        full_n = False
    else:
        R, R_aco = args.R, args.R_aco
        full_n = args.full_n

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _log(f"Старт запуску експериментів. R={R}, R_aco={R_aco}, "
         f"full_n={full_n}, only={args.only}")
    t_total = time.perf_counter()

    try:
        if args.only in (None, 1):
            run_exp1(R, R_aco)
        if args.only in (None, 2):
            run_exp2(R, R_aco)
        if args.only in (None, 3):
            run_exp3(R, R_aco)
        if args.only in (None, 4):
            run_exp4(R, R_aco, full=full_n)
    except KeyboardInterrupt:
        _log("Перервано користувачем.")
        sys.exit(130)

    _log(f"УСІ ЕКСПЕРИМЕНТИ ЗАВЕРШЕНО за "
         f"{(time.perf_counter() - t_total) / 60:.1f} хв")


if __name__ == "__main__":
    main()
