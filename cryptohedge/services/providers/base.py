"""Provider abstractions shared by all data sources.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services.providers (источники данных).
НАЗНАЧЕНИЕ:  определяет ОБЩИЙ контракт всех провайдеров данных: структуру
             MarketDataBundle (что должен вернуть любой источник) и абстрактный
             класс MarketDataProvider. Также задаёт КОНСТАНТЫ кодов инструментов
             и схему колонок опционных данных.
ИМПОРТИРУЕТ: abc (ABC/abstractmethod), dataclasses, typing, pandas.
ЭКСПОРТИРУЕТ:
  - INSTR_ASSET/INSTR_CALL/INSTR_PUT : числовые коды типов инструментов.
  - MARKET_DATA_COLUMNS              : ожидаемая схема колонок опционных данных.
  - MarketDataBundle                 : контейнер всех данных для агентов.
  - MarketDataProvider               : абстрактный провайдер (метод load()).
КЕМ ИСПОЛЬЗУЕТСЯ:  bundled/synthetic/binance провайдеры наследуют контракт;
                   агенты 1/3/4/8 читают коды INSTR_* и схему.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from abc import ABC, abstractmethod                      # абстрактный базовый класс провайдера
from dataclasses import dataclass, field                 # @dataclass для MarketDataBundle
from typing import Dict, List                            # аннотации

import pandas as pd                                       # таблицы данных

# Instrument-type codes matching the bundled dataset schema (see *.parquet descriptions).
INSTR_ASSET = 7                                          # код «базовый актив» (спот)
INSTR_CALL = 5                                           # код «опцион call»
INSTR_PUT = 6                                            # код «опцион put»

MARKET_DATA_COLUMNS = [                                  # обязательная схема колонок опционных данных
    "sample_idx",                                        #   индекс временно́го среза
    "timestamp",                                         #   временна́я метка
    "instrument_type",                                   #   тип инструмента (INSTR_*)
    "strike",                                            #   страйк
    "expiry_ts",                                         #   время экспирации
    "time_to_maturity",                                  #   срок до экспирации
    "price",                                             #   цена
    "best_bid_price",                                    #   лучшая цена покупки
    "best_ask_price",                                    #   лучшая цена продажи
    "bid_amount_total",                                  #   суммарный объём на покупку
    "ask_amount_total",                                  #   суммарный объём на продажу
    "bid_vwap",                                          #   VWAP покупки
    "ask_vwap",                                          #   VWAP продажи
]


@dataclass
class MarketDataBundle:
    """Everything the downstream agents need from a data source.

    Attributes:
        spot_bars: Long OHLCV frame with columns
            ``[timestamp, symbol, open, high, low, close, volume]``.
        spot_close: Wide close-price frame ``[timestamp x symbol]``.
        option_market_data: Option + spot quotes for the primary symbol following
            the bundled ``market_data`` schema (see :data:`MARKET_DATA_COLUMNS`).
        symbols: The instrument universe.
        primary_symbol: The asset whose risk is hedged (e.g. ``BTCUSDT``).
        meta: Free-form provenance metadata.
    """

    spot_bars: pd.DataFrame                              # «длинные» OHLCV-бары по всем символам
    spot_close: pd.DataFrame                             # «широкая» таблица цен закрытия [время × символ]
    option_market_data: pd.DataFrame                    # опционные котировки первичного актива
    symbols: List[str]                                   # вселенная инструментов
    primary_symbol: str                                  # хеджируемый актив (BTCUSDT)
    meta: Dict[str, object] = field(default_factory=dict)  # произвольные метаданные происхождения

    def validate(self) -> "MarketDataBundle":
        if self.spot_close.isna().all().any():           # есть полностью пустая колонка цен?
            raise ValueError("spot_close contains an all-NaN column")  #   → ошибка
        missing = set(MARKET_DATA_COLUMNS) - set(self.option_market_data.columns)  # каких колонок не хватает
        if missing:                                      # если чего-то не хватает…
            raise ValueError(f"option_market_data missing columns: {sorted(missing)}")  #   → ошибка
        if self.primary_symbol not in self.symbols:      # первичный актив обязан быть во вселенной…
            raise ValueError("primary_symbol not present in symbols")  #   → ошибка
        return self                                      # данные валидны — возвращаем себя


class MarketDataProvider(ABC):
    """Contract for a market-data source."""

    name: str = "base"                                  # имя провайдера (переопределяется потомками)

    @abstractmethod
    def load(self) -> MarketDataBundle:                  # КАЖДЫЙ провайдер обязан реализовать load()
        """Return a fully populated, validated :class:`MarketDataBundle`."""
