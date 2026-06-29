"""The eleven autonomous agents of the hedging system.

Each agent lives in its own module and implements the unified
:class:`cryptohedge.core.agent.BaseAgent` interface. :func:`build_pipeline` wires
them into an :class:`cryptohedge.core.Orchestrator` in dependency order.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.agents (внешний слой) — публичный фасад пакета агентов.
НАЗНАЧЕНИЕ:  собирает все 11 агентов и предоставляет two helper'а:
             all_agents() — экземпляры в каноническом порядке пайплайна;
             build_pipeline(context) — готовый Orchestrator со всеми агентами.
             ЭТО точка входа для понимания пайплайна целиком (читать после core/).
ИМПОРТИРУЕТ:
  - typing.List              : аннотация списка агентов.
  - core.agent.BaseAgent     : общий тип агентов.
  - core.context.AgentContext: контекст для оркестратора.
  - core.orchestrator        : Orchestrator (сборка пайплайна).
  - agents.* (11 модулей)    : сами классы агентов.
ЭКСПОРТИРУЕТ:
  - 11 классов агентов + all_agents() + build_pipeline().
КЕМ ИСПОЛЬЗУЕТСЯ:
  - cli.py::run_pipeline() и solution.ipynb вызывают build_pipeline(...).run().
ПОРЯДОК АГЕНТОВ = ПОРЯДОК ПАЙПЛАЙНА (данные→…→дашборд).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import List                                  # аннотация списка агентов

from cryptohedge.core.agent import BaseAgent             # базовый тип агента
from cryptohedge.core.context import AgentContext        # контекст прогона
from cryptohedge.core.orchestrator import Orchestrator   # оркестратор пайплайна

from cryptohedge.agents.data_acquisition import DataAcquisitionAgent          # 1) сбор данных
from cryptohedge.agents.market_analysis import MarketAnalysisAgent            # 2) анализ рынка
from cryptohedge.agents.heston_calibration import HestonCalibrationAgent      # 3) калибровка Хестона
from cryptohedge.agents.greeks_calculation import GreeksCalculationAgent      # 4) греки
from cryptohedge.agents.hedging_decision import HedgingDecisionAgent          # 5) решение/исполнение хеджа
from cryptohedge.agents.portfolio_optimization import PortfolioOptimizationAgent  # 6) портфель
from cryptohedge.agents.risk_management import RiskManagementAgent            # 7) риск-менеджмент
from cryptohedge.agents.backtesting import BacktestingAgent                   # 8) бэктест/стресс
from cryptohedge.agents.self_diagnostic import SelfDiagnosticAgent            # 9) самодиагностика
from cryptohedge.agents.explainability import ExplainabilityAgent            # 10) объяснимость
from cryptohedge.agents.dashboard import DashboardAgent                       # 11) дашборд

__all__ = [                                              # публичный API пакета agents
    "DataAcquisitionAgent",
    "MarketAnalysisAgent",
    "HestonCalibrationAgent",
    "GreeksCalculationAgent",
    "HedgingDecisionAgent",
    "PortfolioOptimizationAgent",
    "RiskManagementAgent",
    "BacktestingAgent",
    "SelfDiagnosticAgent",
    "ExplainabilityAgent",
    "DashboardAgent",
    "all_agents",
    "build_pipeline",
]


def all_agents() -> List[BaseAgent]:
    """Instantiate the eleven agents in canonical pipeline order."""
    return [                                             # порядок важен — это и есть последовательность пайплайна
        DataAcquisitionAgent(),                          # 1
        MarketAnalysisAgent(),                           # 2
        HestonCalibrationAgent(),                        # 3
        GreeksCalculationAgent(),                        # 4
        HedgingDecisionAgent(),                          # 5
        PortfolioOptimizationAgent(),                    # 6
        RiskManagementAgent(),                           # 7
        BacktestingAgent(),                              # 8
        SelfDiagnosticAgent(),                           # 9
        ExplainabilityAgent(),                           # 10
        DashboardAgent(),                                # 11
    ]


def build_pipeline(context: AgentContext, fail_fast: bool = True) -> Orchestrator:
    """Create and wire the full agent pipeline."""
    orch = Orchestrator(context, fail_fast=fail_fast)    # создаём оркестратор с заданной политикой ошибок
    for agent in all_agents():                           # перебираем все 11 агентов по порядку…
        orch.register(agent)                             # …регистрируем каждого как этап пайплайна
    return orch                                          # возвращаем готовый к запуску оркестратор
