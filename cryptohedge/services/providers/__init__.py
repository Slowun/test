"""Market-data providers (pluggable data sources).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services.providers — фасад источников данных.
НАЗНАЧЕНИЕ:  реэкспортирует контракт и провайдеры + ФАБРИКА build_provider,
             которая по строке config.data.provider возвращает нужный источник
             (bundled/synthetic/binance). Реализует принцип Open-Closed: добавить
             новый источник можно, не меняя агентов.
ИМПОРТИРУЕТ: core.config.DataConfig; base/bundled/synthetic провайдеры.
ЭКСПОРТИРУЕТ: MarketDataBundle, MarketDataProvider, BundledProvider,
             SyntheticProvider, build_provider.
КЕМ ИСПОЛЬЗУЕТСЯ: агент 1 (DataAcquisition) зовёт build_provider(...).
ПРИМЕЧАНИЕ: BinanceProvider импортируется ЛЕНИВО (только при provider=binance),
           чтобы не требовать сеть/зависимости при оффлайн-прогоне.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from cryptohedge.core.config import DataConfig           # типизированный конфиг данных
from cryptohedge.services.providers.base import MarketDataBundle, MarketDataProvider  # контракт
from cryptohedge.services.providers.bundled import BundledProvider      # оффлайн-датасет
from cryptohedge.services.providers.synthetic import SyntheticProvider  # синтетика из seed

__all__ = [                                              # публичный API пакета провайдеров
    "MarketDataBundle",
    "MarketDataProvider",
    "BundledProvider",
    "SyntheticProvider",
    "build_provider",
]


def build_provider(config: DataConfig, root, seed: int, n_steps: int = 90) -> MarketDataProvider:
    """Factory selecting a provider by configuration (Open-Closed principle)."""
    provider = config.provider.lower()                   # имя провайдера в нижнем регистре
    if provider == "synthetic":                          # синтетический источник…
        return SyntheticProvider(config, seed=seed, n_steps=n_steps)  #   детерминированная генерация
    if provider == "bundled":                            # версионированный оффлайн-датасет…
        return BundledProvider(config, root=root, seed=seed, n_steps=n_steps)  #   чтение с диска
    if provider == "binance":                            # живой источник Binance…
        from cryptohedge.services.providers.binance import BinanceProvider  # ленивый импорт (сеть/зависимости)

        return BinanceProvider(config, seed=seed, n_steps=n_steps)  #   REST-провайдер
    raise ValueError(f"Unknown data provider: {config.provider}")  # неизвестный провайдер → ошибка
