"""Walk-forward splitting without look-ahead bias.

Generates expanding/rolling train-test folds with optional purge & embargo gaps so
that no test observation can leak information into training (and vice versa).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (валидация без заглядывания вперёд).
НАЗНАЧЕНИЕ:  генерирует расширяющиеся/скользящие фолды train-test с опциональными
             разрывами purge и embargo, чтобы тестовые наблюдения не «протекали»
             в обучение и наоборот (предотвращение look-ahead bias).
ИМПОРТИРУЕТ: dataclass; numpy.
ЭКСПОРТИРУЕТ: Fold (один фолд), walk_forward_splits (генератор фолдов).
КЕМ ИСПОЛЬЗУЕТСЯ: агент backtesting (walk-forward валидация).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass                        # для Fold
from typing import Iterator, List                        # аннотации

import numpy as np                                       # индексные массивы


@dataclass(frozen=True)
class Fold:                                               # один фолд walk-forward
    index: int                                          # порядковый номер фолда
    train: np.ndarray                                   # индексы обучения
    test: np.ndarray                                    # индексы теста


def walk_forward_splits(
    n: int,                                             # общее число наблюдений
    train_window: int,                                 # размер окна обучения
    test_window: int,                                  # размер окна теста
    step: int = None,                                  # шаг сдвига (по умолч. = test_window)
    purge: int = 0,                                    # разрыв между train и test
    embargo: int = 0,                                  # дополнительный отступ после теста
    expanding: bool = False,                           # расширяющееся окно обучения
) -> List[Fold]:
    """Return rolling (or expanding) walk-forward folds over ``range(n)``."""
    step = step or test_window                          # шаг по умолчанию = размер теста
    folds: List[Fold] = []                             # результирующие фолды
    start = 0                                          # начало текущего окна
    fold_idx = 0                                       # счётчик фолдов
    while True:                                        # пока влезает следующий фолд:
        train_end = start + train_window               #   конец обучения
        test_start = train_end + purge                 #   начало теста (с purge-разрывом)
        test_end = test_start + test_window            #   конец теста
        if test_end > n:                               #   вышли за пределы данных…
            break                                      #     стоп
        train_start = 0 if expanding else start        #   расширяющееся или скользящее окно
        train = np.arange(train_start, train_end)      #   индексы обучения
        test = np.arange(test_start, test_end)         #   индексы теста
        folds.append(Fold(fold_idx, train, test))      #   добавляем фолд
        fold_idx += 1                                  #   следующий номер
        start += step + embargo                        #   сдвигаем окно (с embargo)
    return folds                                       # список фолдов
