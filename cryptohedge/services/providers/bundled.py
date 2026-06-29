"""Bundled provider: reads the pre-generated, version-controlled dataset.

This is the default, fully reproducible source. If the files are missing it
transparently regenerates them with the :class:`SyntheticProvider` (same seed),
so a fresh checkout always works offline.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services.providers (источник данных по умолчанию).
НАЗНАЧЕНИЕ:  читает версионированный датасет из data/raw (parquet + csv). Если
             файлов нет — прозрачно генерирует их SyntheticProvider'ом с тем же
             seed, поэтому свежий клон всегда работает оффлайн и воспроизводимо.
ИМПОРТИРУЕТ: pathlib, pandas; core.config.DataConfig; base.* ; SyntheticProvider.
ЭКСПОРТИРУЕТ: BundledProvider (load/exists/materialize).
КЕМ ИСПОЛЬЗУЕТСЯ: фабрика build_provider при config.data.provider == "bundled".
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from pathlib import Path                                 # пути к файлам датасета

import pandas as pd                                       # чтение/запись parquet/csv

from cryptohedge.core.config import DataConfig           # типизированный конфиг данных
from cryptohedge.services.providers.base import MarketDataBundle, MarketDataProvider  # контракт
from cryptohedge.services.providers.synthetic import SyntheticProvider  # фолбэк-генератор


class BundledProvider(MarketDataProvider):
    name = "bundled"                                     # имя провайдера

    def __init__(self, config: DataConfig, root: str | Path, seed: int, n_steps: int = 90) -> None:
        self.config = config                             # конфиг данных
        self.root = Path(root)                           # корень проекта
        self.seed = seed                                 # seed (для фолбэк-генерации)
        self.n_steps = n_steps                           # число шагов истории
        self.raw_dir = self.root / "data" / "raw"        # каталог сырого датасета

    def _paths(self):
        return (                                         # кортеж путей файлов датасета:
            self.raw_dir / "spot_bars.parquet",          #   OHLCV-бары
            self.raw_dir / "spot_close.parquet",         #   цены закрытия (широкая таблица)
            self.raw_dir / "market_data.parquet",        #   опционные данные
            self.raw_dir / "universe.csv",               #   список символов
        )

    def exists(self) -> bool:
        return all(p.exists() for p in self._paths())    # все ли файлы датасета на месте

    def materialize(self) -> MarketDataBundle:
        """Generate the synthetic dataset and persist it to ``data/raw``."""
        bundle = SyntheticProvider(self.config, seed=self.seed, n_steps=self.n_steps).load()  # генерируем синтетику
        self.raw_dir.mkdir(parents=True, exist_ok=True)  # создаём каталог
        bars, close, md, universe = self._paths()        # пути для сохранения
        bundle.spot_bars.to_parquet(bars)                # сохраняем бары
        bundle.spot_close.to_parquet(close)              # сохраняем цены закрытия
        bundle.option_market_data.to_parquet(md)         # сохраняем опционы
        pd.Series(bundle.symbols, name="symbol").to_csv(universe, index=False)  # сохраняем символы
        pd.Series(bundle.meta).to_json(self.raw_dir / "meta.json")  # сохраняем метаданные
        return bundle                                    # возвращаем сгенерированный пакет

    def load(self) -> MarketDataBundle:
        if not self.exists():                            # если датасета нет на диске…
            return self.materialize()                    #   → генерируем и сохраняем
        bars, close, md, universe = self._paths()        # пути файлов
        spot_bars = pd.read_parquet(bars)                # читаем бары
        spot_close = pd.read_parquet(close)              # читаем цены закрытия
        option_market_data = pd.read_parquet(md)         # читаем опционы
        symbols = pd.read_csv(universe)["symbol"].tolist()  # читаем список символов
        return MarketDataBundle(                         # собираем пакет данных
            spot_bars=spot_bars,
            spot_close=spot_close,
            option_market_data=option_market_data,
            symbols=symbols,
            primary_symbol=self.config.primary_symbol,
            meta={"provider": self.name, "source": str(self.raw_dir)},
        ).validate()                                     # валидируем перед возвратом
