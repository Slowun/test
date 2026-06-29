"""Decision-domain value objects produced by the agents.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.domain (внутренний слой).
НАЗНАЧЕНИЕ:  структуры-РЕШЕНИЯ, которые выпускают агенты: решение о хедже,
             решение о ребалансировке портфеля, уровень стоп-лосса, оценка риска.
             Каждое решение несёт не только число, но и его обоснование (rationale,
             metrics, components) — основа объяснимости (агент 10).
ИМПОРТИРУЕТ: dataclasses (frozen dataclass, asdict, field), typing (Literal/Dict/List).
ЭКСПОРТИРУЕТ: HedgeDecision, RebalanceDecision, StopLevel, RiskAssessment.
КЕМ СОЗДАЁТСЯ:
  - HedgeDecision      : services/hedging_engine, агент 5.
  - RebalanceDecision  : services/optimization, агент 6.
  - StopLevel          : services/stops, агент 7.
  - RiskAssessment     : services/metrics, агент 7.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass, asdict, field         # frozen dataclass + поля по умолчанию
from typing import Dict, List, Literal                   # Literal ограничивает допустимые значения


@dataclass(frozen=True)
class HedgeDecision:
    """A delta/vega hedging instruction with quantitative justification."""

    timestamp: int                                       # момент решения (наносекунды)
    instrument: Literal["spot", "vega_option"]           # чем хеджируем: спот (Δ) или опцион (ν)
    side: Literal["buy", "sell", "hold"]                 # действие: купить/продать/держать
    quantity: float                                      # объём хеджа
    target_greek: str  # 'delta' | 'vega'                # какой грек нейтрализуем
    pre_hedge_value: float                               # значение грека ДО хеджа
    post_hedge_value: float                              # значение грека ПОСЛЕ хеджа
    rationale: str = ""                                  # текстовое обоснование решения
    metrics: Dict[str, float] = field(default_factory=dict)  # числовые метрики решения

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)                              # сериализация в словарь


@dataclass(frozen=True)
class RebalanceDecision:
    """A portfolio rebalancing instruction from the optimization agent."""

    method: str                                          # выбранный метод оптимизации
    target_weights: Dict[str, float]                     # целевые веса активов
    current_weights: Dict[str, float]                    # текущие веса активов
    turnover: float                                      # оборот ребалансировки (сумма |Δвес|)
    expected_return: float                               # ожидаемая доходность портфеля
    expected_risk: float                                 # ожидаемый риск портфеля
    transaction_cost: float                              # издержки ребалансировки
    triggered: bool                                      # сработала ли ребалансировка
    rationale: str = ""                                  # текстовое обоснование

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)                              # сериализация в словарь


@dataclass(frozen=True)
class StopLevel:
    """An adaptive stop-loss / trailing-stop level."""

    instrument: str                                      # инструмент, к которому относится стоп
    side: Literal["long", "short"]                       # сторона позиции
    stop_price: float                                    # цена срабатывания стопа
    reference_price: float                               # опорная цена (вход/референс)
    distance_pct: float                                  # дистанция стопа в процентах
    method: str                                          # метод расчёта (atr+var+heston и т.п.)
    components: Dict[str, float] = field(default_factory=dict)  # вклад компонент (ATR/VaR/Heston)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)                              # сериализация в словарь


@dataclass(frozen=True)
class RiskAssessment:
    """Output of the risk-management agent for a single evaluation point."""

    var: float                                           # Value at Risk
    cvar: float                                          # Conditional VaR
    expected_shortfall: float                            # ожидаемые потери в хвосте (ES)
    max_drawdown: float                                  # максимальная просадка
    within_limits: bool                                  # все ли лимиты соблюдены
    breached_limits: List[str] = field(default_factory=list)  # список нарушенных лимитов
    utilization: Dict[str, float] = field(default_factory=dict)  # утилизация лимитов (доля от лимита)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)                              # сериализация в словарь
