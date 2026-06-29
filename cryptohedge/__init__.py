"""cryptohedge: a multi-agent system for hedging crypto FX (volatility) risk.

The package follows a Clean Architecture layering:

* :mod:`cryptohedge.domain`   - pure, dependency-free domain entities / value objects.
* :mod:`cryptohedge.core`     - application framework: config, logging, messaging,
  checkpointing, the base agent contract and the orchestrator.
* :mod:`cryptohedge.services` - computational use-cases (volatility, correlation,
  calibration, optimization, risk metrics, stop-loss, data providers).
* :mod:`cryptohedge.agents`   - the eleven autonomous agents, each an independent
  module exposing the unified :class:`cryptohedge.core.agent.BaseAgent` interface.

Every random process is seeded from a single configuration value to guarantee
full reproducibility (see :func:`cryptohedge.core.seeding.set_global_seed`).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        корень пакета cryptohedge (точка «знакомства» с проектом).
НАЗНАЧЕНИЕ:  описывает слоистую архитектуру (domain/core/services/agents) и
             хранит версию пакета. Это первый файл для чтения новичком.
ЭКСПОРТИРУЕТ: __version__ — строка версии пакета.
=============================================================================
"""

from __future__ import annotations            # отложенные аннотации типов

__version__ = "1.0.0"                          # версия пакета cryptohedge

__all__ = ["__version__"]                      # публичный API на уровне пакета — только версия
