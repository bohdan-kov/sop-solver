"""Генерація UML-діаграм для Розділу 4 (Рис 4.1 та Рис 4.2).

Скрипт:
  1) формує два PlantUML-описи (компонентну та класову діаграми);
  2) зберігає їх у `diagrams/*.puml`;
  3) рендерить кожну у PNG через онлайн-сервіс kroki.io.

Запуск:
    python3 make_diagrams.py

Сервіс kroki.io (https://kroki.io) — публічний рендерер діаграм,
який підтримує PlantUML, Mermaid, Graphviz, BlockDiag тощо. Не
потребує локальної установки Java, Graphviz чи PlantUML — вистачає
інтернет-з'єднання.

Якщо kroki.io недоступний, .puml-файли можна вставити в:
  • https://www.plantuml.com/plantuml/uml/ (офіційний онлайн-рендерер);
  • будь-який локальний `plantuml.jar`.
"""
from __future__ import annotations

import base64
import os
import sys
import urllib.error
import urllib.request
import zlib


DIAGRAMS_DIR = "diagrams"
KROKI_URL = "https://kroki.io/plantuml/png"
PLANTUML_URL_TEMPLATE = "https://www.plantuml.com/plantuml/png/~1{encoded}"
USER_AGENT = "Mozilla/5.0 (compatible; sop-solver/1.0)"


# ============================================================
#  РИС 4.1 — Схема залежностей модулів програмного продукту
# ============================================================

DIAGRAM_4_1 = r"""@startuml
title Рисунок 4.1 — Схема залежностей модулів програмного продукту

skinparam backgroundColor #FAFAFA
skinparam shadowing false
skinparam componentStyle rectangle
skinparam component {
    FontName "DejaVu Sans"
    BorderColor #1F4E79
    BackgroundColor<<entry>> #BDD7EE
    BackgroundColor<<io>>    #D9E2F3
    BackgroundColor<<logic>> #C6E0B4
    BackgroundColor<<data>>  #FFF2CC
}

package "Шар вводу-виведення" #EAF1FA {
  [main.py\nТочка входу / CLI]   as main   <<entry>>
  [report.py\nВиведення таблиць та логів] as report <<io>>
  [visualizer.py\nГрафіки matplotlib] as viz <<io>>
  [file_io.py\nЧитання-запис JSON, PNG] as fio <<io>>
  [data_types.py\nSOPInstance / SOPResult] as dt <<io>>
}

package "Шар бізнес-логіки" #EBF7E2 {
  [generator.py\nГенератор\nіндивідуальних задач] as gen <<logic>>
  [greedy.py\nЖадібний\nалгоритм]               as greedy <<logic>>
  [aco.py\nАлгоритм\nмурашиних колоній]         as aco    <<logic>>
  [brute.py\nПовний перебір\n(n ≤ 10)]          as brute  <<logic>>
  [experiments.py\nСерії експериментів,\nкалібрування α, β, ρ, L]  as exp <<logic>>
}

package "Спільні структури даних" #FFF6DA {
  [numpy.ndarray\nМатриця D, феромон τ,\nrem_pred, succ_list]  as np <<data>>
}

main   --> report
main   --> viz
main   --> fio
main   --> gen
main   --> greedy
main   --> aco
main   --> exp
main   --> dt

exp    --> gen
exp    --> greedy
exp    --> aco
exp    --> brute

aco    --> greedy : ініціалізація\nτ₀ та Q
aco    --> dt
greedy --> dt
brute  --> dt
gen    --> dt
fio    --> dt
viz    --> dt

greedy --> np : D, rem_pred,\nsucc_list
aco    --> np : τ, η, D
brute  --> np
gen    --> np : coords

@enduml
"""


# ============================================================
#  РИС 4.2 — Діаграма класів програмного продукту
# ============================================================

DIAGRAM_4_2 = r"""@startuml
title Рисунок 4.2 — Діаграма класів програмного продукту

skinparam backgroundColor #FAFAFA
skinparam shadowing false
skinparam classFontName "DejaVu Sans"
skinparam class {
    BorderColor #1F4E79
    BackgroundColor #DCE6F1
    HeaderBackgroundColor #BDD7EE
    AttributeFontColor #1F1F1F
}

class SOPInstance <<dataclass>> {
    + n : int
    + coords : ndarray[n,2]
    + s : ndarray[2]
    + chains : List[List[int]]
    + p : float
    + X_max : float
    + Y_max : float
    + seed : Optional[int]
    --
    + D : ndarray[n+1, n+1]
    + to_dict() : dict
    + from_dict(d) : SOPInstance
    - _build_D(v=10.0) : ndarray
}

class SOPResult <<dataclass>> {
    + route : List[int]
    + T : float
    + t_ms : float
    + algorithm : str
    --
    + to_dict() : dict
}

class ACOParams <<dataclass>> {
    + alpha : float
    + beta : float
    + rho : float
    + m_a : Optional[int]
    + N_iter : int
    + N_stag : int
    --
    + resolve(n) : ACOParams
}

class GreedySolver <<module: greedy.py>> {
    + greedy_solve(inst : SOPInstance) : SOPResult
    - _build_predecessor_lookup(inst)\n     : (rem_pred, succ_list)
}

class AntColonySolver <<module: aco.py>> {
    + aco_solve(inst, params, seed) : SOPResult
    + aco_solve_average(inst, params,\n     R_aco, base_seed) : (best, stats)
    - _build_one_route(D, tau, ...) : (route, T)
}

class BruteForceSolver <<module: brute.py>> {
    + brute_force_solve(inst) : SOPResult
}

class TaskGenerator <<module: generator.py>> {
    + generate_instance(n, X_max, Y_max,\n     p, L_min, L_max, seed) : SOPInstance
}

class ExperimentRunner <<module: experiments.py>> {
    + experiment_n_stag(...) : List[dict]
    + experiment_beta(...) : List[dict]
    + experiment_param_series(name,\n     values, ...) : List[dict]
    + experiment_n(n_values, ...) : List[dict]
    - _compute_pair_metrics(T_g, T_a)\n     : (delta, win)
}

class Visualizer <<module: visualizer.py>> {
    + visualize_route(inst, result, fname)
    + plot_calibration_n_stag(rows, fname)
    + plot_calibration_beta(rows, fname)
    + plot_param_series(rows, name, fname)
    + plot_n_series(rows, q_fname,\n     t_fname, gap_fname)
}

class FileIO <<module: file_io.py>> {
    + save_instance(inst, path,\n     greedy?, aco?)
    + load_instance(path) : SOPInstance
    + save_results_table(rows, path,\n     header)
}

class CLI <<module: main.py>> {
    + main()
    - menu_input_instance(state)
    - menu_solve(state)
    - menu_experiments(state)
    - menu_output(state)
}

' --- залежності ---
GreedySolver        ..> SOPInstance       : uses
GreedySolver        ..> SOPResult         : returns

AntColonySolver     ..> SOPInstance       : uses
AntColonySolver     ..> ACOParams         : uses
AntColonySolver     ..> SOPResult         : returns
AntColonySolver     ..> GreedySolver      : ініціалізація\nτ₀, Q

BruteForceSolver    ..> SOPInstance       : uses
BruteForceSolver    ..> SOPResult         : returns

TaskGenerator       ..> SOPInstance       : creates

ExperimentRunner    ..> TaskGenerator     : uses
ExperimentRunner    ..> GreedySolver      : uses
ExperimentRunner    ..> AntColonySolver   : uses
ExperimentRunner    ..> BruteForceSolver  : uses

Visualizer          ..> SOPInstance       : uses
Visualizer          ..> SOPResult         : uses

FileIO              ..> SOPInstance       : (de)serialize

CLI                 ..> TaskGenerator
CLI                 ..> GreedySolver
CLI                 ..> AntColonySolver
CLI                 ..> ExperimentRunner
CLI                 ..> Visualizer
CLI                 ..> FileIO

@enduml
"""


# ============================================================
#                   РЕНДЕРИНГ ЧЕРЕЗ KROKI.IO
# ============================================================

def render_via_kroki(plantuml_source: str, output_path: str,
                     timeout: float = 30.0) -> None:
    """Відіслати PlantUML-код на kroki.io і зберегти отриманий PNG."""
    data = plantuml_source.encode("utf-8")
    req = urllib.request.Request(
        KROKI_URL,
        data=data,
        headers={
            "Content-Type": "text/plain",
            "Accept": "image/png",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        png = resp.read()
    with open(output_path, "wb") as f:
        f.write(png)


# --- Резервний спосіб: офіційний PlantUML-сервер -----------------------

_PLANTUML_ALPHABET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "-_"
)


def _plantuml_encode(source: str) -> str:
    """Кодування PlantUML-тексту для GET-запиту до plantuml.com.

    Алгоритм: DEFLATE → base64 з кастомним алфавітом '0-9A-Za-z-_'.
    Деталі: https://plantuml.com/text-encoding
    """
    raw = zlib.compress(source.encode("utf-8"), 9)
    # zlib додає 2-байтовий заголовок і 4-байтовий Adler-32; PlantUML
    # очікує "сирий" deflate-потік.
    if raw[:2] == b"\x78\x9c":
        raw = raw[2:-4]
    std_b64 = base64.b64encode(raw).decode("ascii")
    table = str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
        _PLANTUML_ALPHABET,
    )
    return std_b64.rstrip("=").translate(table)


def render_via_plantuml(plantuml_source: str, output_path: str,
                        timeout: float = 30.0) -> None:
    """Відрендерити через https://www.plantuml.com/plantuml/png/<...>."""
    encoded = _plantuml_encode(plantuml_source)
    url = PLANTUML_URL_TEMPLATE.format(encoded=encoded)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        png = resp.read()
    with open(output_path, "wb") as f:
        f.write(png)


def render_with_fallback(source: str, output_path: str) -> str:
    """Спробувати kroki.io, потім plantuml.com. Повертає назву сервісу."""
    try:
        render_via_kroki(source, output_path)
        return "kroki.io"
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e1:
        print(f"    ⚠ kroki.io: {e1}; пробую plantuml.com ...")
        render_via_plantuml(source, output_path)
        return "plantuml.com"


def save_source(source: str, fname: str) -> None:
    with open(fname, "w", encoding="utf-8") as f:
        f.write(source)


def main() -> None:
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)

    plans = [
        ("Рис_4_1_Схема_модулів",   DIAGRAM_4_1),
        ("Рис_4_2_Діаграма_класів", DIAGRAM_4_2),
    ]

    for name, src in plans:
        puml_path = os.path.join(DIAGRAMS_DIR, f"{name}.puml")
        png_path = os.path.join(DIAGRAMS_DIR, f"{name}.png")
        save_source(src, puml_path)
        print(f"  → {puml_path}")
        try:
            service = render_with_fallback(src, png_path)
            print(f"  → {png_path}  (через {service})")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"    ⚠ Не вдалося відрендерити автоматично: {e}")
            print(f"      Вставте {puml_path} вручну на")
            print(f"      https://www.plantuml.com/plantuml/uml/")

    print("\nГотово. Файли у директорії diagrams/.")
    print("Якщо потрібно заново відрендерити — повторно запустіть скрипт.")


if __name__ == "__main__":
    sys.exit(main() or 0)
