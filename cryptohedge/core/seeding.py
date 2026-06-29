"""Centralised seeding for full reproducibility.

Every stochastic component draws from generators initialised here from a single
configuration value, so identical inputs always produce identical outputs.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас приложения, поперечный слой).
НАЗНАЧЕНИЕ:  единая точка инициализации ВСЕХ генераторов случайных чисел (RNG)
             из одного числа `seed`. Это фундамент воспроизводимости проекта:
             синтетические данные, калибровка Хестона, Монте-Карло, веса
             портфеля и PnL получаются бит-в-бит одинаковыми при одном seed.
ИМПОРТИРУЕТ:
  - os      : чтобы выставить переменную окружения PYTHONHASHSEED.
  - random  : стандартный генератор Python (его тоже нужно засидить).
  - numpy   : основной численный генератор всего проекта.
  - torch   : опционально (если установлен) — для нейросетевых частей.
ЭКСПОРТИРУЕТ (используется снаружи):
  - set_global_seed(seed) -> np.random.Generator   — засидить всё и вернуть RNG.
  - spawn_rng(seed, stream) -> np.random.Generator — независимый под-поток RNG.
КЕМ ИСПОЛЬЗУЕТСЯ:
  - core/context.py : AgentContext в конструкторе зовёт set_global_seed(config.seed)
                      и spawn_rng() для отдельных RNG-потоков каждому агенту.
КОНСТАНТЫ:        своих нет; значение seed приходит из config/system.yaml (seed: 90909090).
=============================================================================
"""

from __future__ import annotations  # отложенное вычисление аннотаций типов (PEP 563)

import os                           # доступ к переменным окружения (PYTHONHASHSEED)
import random                       # стандартный RNG Python — его тоже сидим
from typing import Optional         # тип Optional для необязательного аргумента stream

import numpy as np                  # NumPy — основной источник случайности в проекте


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed all known RNG sources and return a fresh NumPy ``Generator``.

    Seeds Python's ``random``, NumPy's legacy global RNG, the ``PYTHONHASHSEED``
    environment variable and (if importable) PyTorch. Returns a dedicated
    :class:`numpy.random.Generator` that callers should thread through their code
    instead of relying on global state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)   # фиксируем хеш-сид Python (порядок set/dict, hash())
    random.seed(seed)                          # сидим стандартный модуль random
    np.random.seed(seed % (2**32 - 1))         # сидим legacy-глобальный RNG NumPy (диапазон uint32)

    try:  # torch is optional for the analytical pipeline   # PyTorch может быть не установлен — не падаем
        import torch                                         # пробуем импортировать torch лениво

        torch.manual_seed(seed)                              # сидим CPU-генератор torch
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent   # если есть GPU…
            torch.cuda.manual_seed_all(seed)                 # …сидим все GPU-генераторы
        torch.use_deterministic_algorithms(True, warn_only=True)  # требуем детерминированные алгоритмы
    except Exception:  # pragma: no cover - torch absent or partial   # torch нет или частично — игнор
        pass                                                 # тихо продолжаем без torch

    return np.random.default_rng(seed)         # возвращаем выделенный современный Generator (его и используем)


def spawn_rng(seed: int, stream: Optional[int] = None) -> np.random.Generator:
    """Return an independent generator for a named sub-stream.

    Using :class:`numpy.random.SeedSequence` guarantees statistically independent
    yet deterministic streams for different agents / components.
    """
    # SeedSequence + spawn_key даёт статистически НЕЗАВИСИМЫЕ, но детерминированные
    # потоки: разным агентам — разный stream, но при том же seed всё воспроизводимо.
    seq = np.random.SeedSequence(entropy=seed, spawn_key=() if stream is None else (stream,))
    return np.random.default_rng(seq)          # строим Generator из этой последовательности
