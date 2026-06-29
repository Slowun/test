"""Shared execution context (blackboard) for the agent pipeline.

The :class:`AgentContext` is the single object threaded through every agent. It
exposes the validated configuration, a per-agent logger factory, the checkpoint
manager, deterministic RNG streams and a shared *blackboard* dictionary that
agents use to publish and consume intermediate artifacts. This keeps agents
decoupled: they depend on the context contract, not on each other.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас).
НАЗНАЧЕНИЕ:  ЦЕНТРАЛЬНЫЙ объект, который протягивается через все агенты. Содержит:
             конфиг, фабрику логгеров, менеджер чекпойнтов, детерминированные
             RNG-потоки и общий словарь-«доску» (blackboard) для обмена артефактами.
             Именно через blackboard агенты передают данные друг другу, не зная
             друг о друге напрямую.
ИМПОРТИРУЕТ:
  - pathlib.Path        : работа с путями файловой системы.
  - typing              : аннотации Any/Dict/Optional.
  - numpy               : тип np.random.Generator для RNG-потоков.
  - core.checkpoint     : CheckpointManager (сохранение/восстановление этапов).
  - core.config         : SystemConfig (валидированная конфигурация).
  - core.logging        : LoggerFactory/StructuredLogger (логирование).
  - core.seeding        : set_global_seed/spawn_rng (воспроизводимость RNG).
ЭКСПОРТИРУЕТ:
  - AgentContext : blackboard (put/get/require/has), logger(), rng(), пути к артефактам.
КЕМ ИСПОЛЬЗУЕТСЯ:
  - cli.py, agents/* (каждый execute(context, message)), orchestrator.py.
КОНСТАНТЫ:   2**31 — модуль для нормировки хеша имени в id RNG-потока.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from pathlib import Path                                  # пути к каталогам/файлам артефактов
from typing import Any, Dict, Optional                    # аннотации blackboard и значений

import numpy as np                                        # тип генератора RNG

from cryptohedge.core.checkpoint import CheckpointManager # менеджер чекпойнтов
from cryptohedge.core.config import SystemConfig          # валидированная конфигурация
from cryptohedge.core.logging import LoggerFactory, StructuredLogger  # логирование
from cryptohedge.core.seeding import set_global_seed, spawn_rng       # сидинг RNG


class AgentContext:
    def __init__(self, config: SystemConfig, root: str | Path = ".") -> None:
        self.config = config                              # сохраняем конфиг (все настройки фонда)
        self.root = Path(root).resolve()                  # корень проекта (абсолютный путь)
        config.paths.ensure(self.root)                    # создаём все нужные каталоги (artifacts, logs…)

        self.master_rng = set_global_seed(config.seed)    # сидим ВСЕ RNG из config.seed и берём мастер-генератор
        self._logger_factory = LoggerFactory(             # создаём фабрику логгеров (консоль + JSONL)
            log_dir=self.root / config.paths.log_dir,     #   каталог логов
            level=config.logging.level,                   #   уровень логирования (INFO…)
            console=config.logging.console,               #   выводить ли в консоль
            jsonl=config.logging.jsonl,                   #   писать ли JSONL-файл
            file_name=config.logging.file_name,           #   имя файла лога
            timing=config.logging.timing,                 #   логировать ли тайминги операций
        )
        self.checkpoints = CheckpointManager(             # менеджер чекпойнтов (resume после сбоя)
            checkpoint_dir=self.root / config.paths.checkpoint_dir,  #   каталог чекпойнтов
            run_id=config.run_id,                         #   идентификатор прогона (подкаталог)
            enabled=config.runtime.checkpointing,         #   включены ли чекпойнты
        )
        self.blackboard: Dict[str, Any] = {}              # ОБЩАЯ ДОСКА: {ключ: артефакт} для обмена между агентами
        self._rng_streams: Dict[str, np.random.Generator] = {}  # кэш именованных RNG-потоков

    # ------------------------------------------------------------------ services
    def logger(self, agent_name: str) -> StructuredLogger:
        return self._logger_factory.get(agent_name)       # выдать структурированный логгер для агента

    def rng(self, name: str) -> np.random.Generator:
        """Return a deterministic, independent RNG stream for the named consumer."""
        if name not in self._rng_streams:                 # если поток для этого имени ещё не создан…
            stream_id = abs(hash(name)) % (2**31)         # …превращаем имя в стабильный id потока
            self._rng_streams[name] = spawn_rng(self.config.seed, stream_id)  # создаём независимый поток
        return self._rng_streams[name]                    # возвращаем (кэшированный) поток

    # --------------------------------------------------------------- blackboard
    def put(self, key: str, value: Any) -> None:
        self.blackboard[key] = value                      # положить артефакт на доску под ключом

    def get(self, key: str, default: Any = None) -> Any:
        return self.blackboard.get(key, default)          # взять артефакт (или default, если нет)

    def require(self, key: str) -> Any:
        if key not in self.blackboard:                    # обязательный артефакт обязан существовать…
            raise KeyError(f"Required artifact '{key}' is missing from the blackboard")  # …иначе ошибка
        return self.blackboard[key]                       # вернуть артефакт

    def has(self, key: str) -> bool:
        return key in self.blackboard                     # есть ли такой ключ на доске

    # ---------------------------------------------------------------- file paths
    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)                    # собрать путь относительно корня проекта
        return p                                          # вернуть Path

    def artifact_path(self, relative: str) -> Path:
        p = self.root / self.config.paths.artifacts_dir / relative  # путь внутри каталога artifacts/
        p.parent.mkdir(parents=True, exist_ok=True)       # создать родительские каталоги при необходимости
        return p                                          # вернуть путь к артефакту

    def results_path(self, name: str) -> Path:
        p = self.root / self.config.paths.results_dir / name  # путь внутри artifacts/results/
        p.parent.mkdir(parents=True, exist_ok=True)       # гарантировать существование каталога
        return p                                          # вернуть путь к файлу результата

    def calibration_path(self, name: str) -> Path:
        p = self.root / self.config.paths.calibration_dir / name  # путь внутри artifacts/calibration/
        p.parent.mkdir(parents=True, exist_ok=True)       # гарантировать существование каталога
        return p                                          # вернуть путь к файлу калибровки
