"""Application framework for the multi-agent hedging system.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.core (каркас) — публичный фасад пакета core.
НАЗНАЧЕНИЕ:  собирает и реэкспортирует все ключевые сущности каркаса в одном
             месте, чтобы остальной код мог писать, например,
             `from cryptohedge.core import AgentContext, Orchestrator`.
РЕЭКСПОРТИРУЕТ (из подмодулей):
  - config.py       : SystemConfig, load_config — конфигурация.
  - context.py      : AgentContext — доска/логгеры/чекпойнты/RNG.
  - agent.py        : BaseAgent, AgentResult — контракт агента.
  - message.py      : Message, MessageType — протокол общения.
  - bus.py          : MessageBus — маршрутизация сообщений.
  - orchestrator.py : Orchestrator, PipelineStage — сборка/запуск пайплайна.
  - checkpoint.py   : CheckpointManager — чекпойнты.
  - seeding.py      : set_global_seed — воспроизводимость.
КЕМ ИСПОЛЬЗУЕТСЯ:  agents/*, cli.py, tests/*, solution.ipynb.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from cryptohedge.core.config import SystemConfig, load_config       # конфигурация и её загрузчик
from cryptohedge.core.context import AgentContext                   # центральный контекст прогона
from cryptohedge.core.agent import BaseAgent, AgentResult           # контракт агента и результат
from cryptohedge.core.message import Message, MessageType           # протокол сообщений
from cryptohedge.core.bus import MessageBus                         # шина сообщений
from cryptohedge.core.orchestrator import Orchestrator, PipelineStage  # оркестратор и этап
from cryptohedge.core.checkpoint import CheckpointManager           # менеджер чекпойнтов
from cryptohedge.core.seeding import set_global_seed                # глобальный сидинг RNG

__all__ = [                                              # явный публичный API пакета core
    "SystemConfig",
    "load_config",
    "AgentContext",
    "BaseAgent",
    "AgentResult",
    "Message",
    "MessageType",
    "MessageBus",
    "Orchestrator",
    "PipelineStage",
    "CheckpointManager",
    "set_global_seed",
]
