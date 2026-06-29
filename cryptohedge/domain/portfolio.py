"""Portfolio-domain value objects.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.domain (внутренний слой).
НАЗНАЧЕНИЕ:  структуры ПОЗИЦИИ и СДЕЛКИ. Position — удерживаемое количество
             инструмента; Trade — исполненная сделка с комиссией. Содержат
             вычисляемые свойства (market_value, signed_quantity, cash_flow).
ИМПОРТИРУЕТ: dataclasses (frozen dataclass, asdict), typing (Literal для side).
ЭКСПОРТИРУЕТ: Position, Trade.
КЕМ ИСПОЛЬЗУЕТСЯ: services/hedging_engine.py (сделки хеджа), агент 5; портфельные сервисы.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass, asdict                # frozen dataclass + сериализация
from typing import Dict, Literal                         # Literal ограничивает значение side


@dataclass(frozen=True)
class Position:
    """A held quantity of an instrument."""

    instrument: str                                      # тикер инструмента
    quantity: float                                      # количество (может быть отрицательным = шорт)
    price: float                                         # цена инструмента

    @property
    def market_value(self) -> float:
        return self.quantity * self.price                # рыночная стоимость позиции

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)                              # сериализация в словарь


@dataclass(frozen=True)
class Trade:
    """An executed trade and the fee it incurred."""

    instrument: str                                      # тикер торгуемого инструмента
    side: Literal["buy", "sell"]                         # сторона сделки: покупка/продажа
    quantity: float                                      # объём сделки (положительный)
    price: float                                         # цена исполнения
    fee: float                                           # уплаченная комиссия
    reason: str = ""                                     # причина сделки (для объяснимости/логов)

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == "buy" else -self.quantity  # знаковый объём (+покупка/−продажа)

    @property
    def cash_flow(self) -> float:
        """Cash impact of the trade including fees (negative = cash out)."""
        return -self.signed_quantity * self.price - self.fee  # денежный поток с учётом комиссии (−=отток)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)                              # сериализация в словарь
