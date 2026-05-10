# SOP Solver

Програмний продукт для розв'язання задачі **оптимізації послідовності
пайки на платі з обмеженнями порядку** (Sequential Ordering Problem,
SOP). Реалізовано жадібний алгоритм та алгоритм мурашиних колоній
(АМК), а також 4 експерименти за планом з розділу 3 курсової роботи.

## Запуск

```bash
pip install numpy matplotlib
python3 main.py
```

## Структура

```
sop-solver/
├── main.py                  # точка входу, CLI-меню
├── saved_tasks/             # JSON-файли збережених ІЗ
├── results/                 # таблиці експериментів та графіки
└── src/
    ├── data_types.py        # SOPInstance / SOPResult
    ├── generator.py         # генератор індивідуальних задач
    ├── algorithms/
    │   ├── greedy.py        # жадібний алгоритм
    │   ├── aco.py           # алгоритм мурашиних колоній (АМК)
    │   └── brute.py         # повний перебор (для n ≤ 10)
    ├── experiments.py       # 4 експерименти + калібрування
    ├── visualizer.py        # графіки matplotlib
    ├── report.py            # виведення таблиць та логів
    └── file_io.py           # JSON-серіалізація
```
