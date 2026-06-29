"""Pure domain entities and value objects (no framework dependencies).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.domain (внутренний слой) — публичный фасад домена.
НАЗНАЧЕНИЕ:  реэкспортирует все доменные сущности (структуры данных, которыми
             обмениваются агенты). Домен НЕ зависит ни от чего внутри проекта —
             это просто типы данных (dataclass), без бизнес-логики и I/O.
РЕЭКСПОРТИРУЕТ:
  - greeks.py    : Greeks, PortfolioGreeks — чувствительности.
  - market.py    : HestonParameters, InstrumentRanking, OptionContract, VolatilityEstimate.
  - portfolio.py : Position, Trade — позиции и сделки.
  - decisions.py : HedgeDecision, RebalanceDecision, RiskAssessment, StopLevel.
КЕМ ИСПОЛЬЗУЕТСЯ:  services/* и agents/* строят и читают эти объекты.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from cryptohedge.domain.greeks import Greeks, PortfolioGreeks  # греки инструмента и портфеля
from cryptohedge.domain.market import (                  # рыночные сущности:
    HestonParameters,                                    #   параметры модели Хестона
    InstrumentRanking,                                   #   рейтинг инструмента хеджа
    OptionContract,                                      #   опционный контракт
    VolatilityEstimate,                                  #   оценка волатильности
)
from cryptohedge.domain.portfolio import Position, Trade  # позиция и сделка
from cryptohedge.domain.decisions import (               # объекты-решения агентов:
    HedgeDecision,                                       #   решение о хедже
    RebalanceDecision,                                   #   решение о ребалансировке
    RiskAssessment,                                      #   оценка риска
    StopLevel,                                           #   уровень стоп-лосса
)

__all__ = [                                              # публичный API домена
    "Greeks",
    "PortfolioGreeks",
    "HestonParameters",
    "InstrumentRanking",
    "OptionContract",
    "VolatilityEstimate",
    "Position",
    "Trade",
    "HedgeDecision",
    "RebalanceDecision",
    "RiskAssessment",
    "StopLevel",
]
