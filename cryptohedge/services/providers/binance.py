"""Live spot provider using Binance public REST endpoints (no API key required).

Spot OHLCV for the universe is fetched from ``/api/v3/klines``. Liquid 30-day
option chains for 100 assets are not freely available without registration, so
the option chain for the primary symbol is generated Heston-consistently around
the *live* spot (clearly documented). If the network is unavailable the provider
degrades gracefully to the fully synthetic dataset, preserving reproducibility.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services.providers (живой источник данных).
НАЗНАЧЕНИЕ:  тянет спотовые OHLCV вселенной через публичный REST Binance
             (/api/v3/klines, без ключа). Ликвидной бесплатной опционной цепочки
             на 100 активов нет — поэтому опционы BTC генерируются согласованно с
             Хестоном вокруг ЖИВОГО спота. При недоступности сети — детерминированный
             фолбэк на SyntheticProvider (воспроизводимость сохраняется).
ИМПОРТИРУЕТ: time, numpy, pandas, requests; core.config.DataConfig; base.*;
             synthetic.DEFAULT_UNIVERSE/SyntheticProvider.
ЭКСПОРТИРУЕТ: BinanceProvider.
КЕМ ИСПОЛЬЗУЕТСЯ: build_provider при config.data.provider == "binance".
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import time                                              # пауза между запросами (rate-limit)
from typing import List                                  # аннотации

import numpy as np                                       # вычисления
import pandas as pd                                       # таблицы
import requests                                          # HTTP-запросы к Binance

from cryptohedge.core.config import DataConfig           # конфиг данных
from cryptohedge.services.providers.base import MarketDataBundle, MarketDataProvider  # контракт
from cryptohedge.services.providers.synthetic import DEFAULT_UNIVERSE, SyntheticProvider  # фолбэк/опционы


class BinanceProvider(MarketDataProvider):
    name = "binance"                                     # имя провайдера

    def __init__(self, config: DataConfig, seed: int, n_steps: int = 90) -> None:
        self.config = config                             # конфиг данных
        self.seed = seed                                 # seed (для опционов/фолбэка)
        self.n_steps = n_steps                           # число баров истории

    def _symbols(self) -> List[str]:
        primary = self.config.primary_symbol             # первичный актив
        source = self.config.symbols or DEFAULT_UNIVERSE  # источник тикеров
        symbols = [s for s in dict.fromkeys(source) if s != primary]  # уникальные без первичного
        return [primary] + symbols[: max(0, self.config.universe_size - 1)]  # первичный + до размера-1

    def _fetch_klines(self, symbol: str, limit: int) -> pd.DataFrame:
        url = f"{self.config.binance_base_url}/api/v3/klines"  # endpoint свечей
        params = {"symbol": symbol, "interval": self.config.bar_interval, "limit": limit}  # параметры запроса
        resp = requests.get(url, params=params, timeout=self.config.request_timeout_s)  # GET-запрос
        resp.raise_for_status()                          # ошибка при HTTP != 2xx
        raw = resp.json()                                # массив свечей (JSON)
        df = pd.DataFrame(                               # в DataFrame по схеме Binance
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "qav", "trades", "tbav", "tqav", "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")  # время открытия → datetime
        for col in ("open", "high", "low", "close", "volume"):  # числовые колонки…
            df[col] = df[col].astype(float)              #   приводим к float
        df["symbol"] = symbol                            # добавляем символ
        return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]  # нормализованный OHLCV

    def load(self) -> MarketDataBundle:
        symbols = self._symbols()                        # список символов
        limit = self.n_steps + 1                         # число баров (шаги + 1)
        frames = []                                      # успешно загруженные фреймы
        good: List[str] = []                             # символы с успешной загрузкой
        try:
            for sym in symbols:                          # по каждому символу…
                try:
                    frames.append(self._fetch_klines(sym, limit))  # тянем свечи
                    good.append(sym)                     # помечаем успех
                    time.sleep(0.05)                     # пауза (вежливый rate-limit)
                except Exception:                        # сбой по одному символу…
                    continue                             #   пропускаем
        except Exception:                                # глобальный сбой сети…
            frames = []                                  #   сбрасываем результаты

        if len(good) < 5 or self.config.primary_symbol not in good:  # данных мало / нет первичного…
            # network unavailable / too few symbols -> deterministic fallback
            return SyntheticProvider(self.config, seed=self.seed, n_steps=self.n_steps).load()  # → фолбэк

        spot_bars = pd.concat(frames, ignore_index=True)  # объединяем все бары
        # align on common timestamps                      # выравниваем по общим меткам времени
        spot_close = spot_bars.pivot_table(index="timestamp", columns="symbol", values="close")  # широкая таблица цен
        spot_close = spot_close.dropna(axis=1, how="any").dropna(axis=0, how="any")  # убираем пропуски
        good = [s for s in good if s in spot_close.columns]  # оставляем символы без пропусков

        # Heston-consistent option chain around the live primary spot.  # опционы вокруг живого спота
        synth = SyntheticProvider(self.config, seed=self.seed, n_steps=len(spot_close) - 1)  # генератор опционов
        live_primary = spot_close[self.config.primary_symbol].to_numpy()  # живой путь первичного актива
        option_market_data = self._synthetic_options(synth, spot_close.index, live_primary)  # строим опционы

        return MarketDataBundle(                         # собираем и валидируем пакет
            spot_bars=spot_bars[spot_bars["symbol"].isin(good)],
            spot_close=spot_close[good],
            option_market_data=option_market_data,
            symbols=good,
            primary_symbol=self.config.primary_symbol,
            meta={"provider": self.name, "live_symbols": len(good)},
        ).validate()

    def _synthetic_options(self, synth: SyntheticProvider, index, spot_path: np.ndarray) -> pd.DataFrame:
        from cryptohedge.services.providers.base import INSTR_ASSET, MARKET_DATA_COLUMNS  # коды/схема (локальный импорт)

        ts_ns = index.astype("int64").to_numpy()         # метки времени в наносекундах
        n = len(spot_path)                               # число срезов
        expiry_ns = int(ts_ns[-1] + self.config.option_expiry_days * 86_400_000_000_000)  # время экспирации
        strikes = synth._strike_grid(float(spot_path[0]))  # сетка страйков от стартовой цены
        # approximate instantaneous variance from realised returns  # дисперсия из реализованных доходностей
        rets = np.diff(np.log(spot_path))                # лог-доходности живого спота
        var = np.concatenate([[np.var(rets) * 365], pd.Series(rets).rolling(10, min_periods=1).var().to_numpy() * 365])  # годовая дисперсия
        records = []                                     # строки опционных данных
        for t in range(n):                               # по каждому срезу…
            spot = float(spot_path[t])                   #   живой спот
            records.append(                              #   строка базового актива (спот)
                {
                    "sample_idx": t, "timestamp": int(ts_ns[t]), "instrument_type": INSTR_ASSET,
                    "strike": 0.0, "expiry_ts": 0, "time_to_maturity": 0.0, "price": spot,
                    "best_bid_price": spot * 0.9999, "best_ask_price": spot * 1.0001,
                    "bid_amount_total": 100.0, "ask_amount_total": 100.0,
                    "bid_vwap": spot * 0.9999, "ask_vwap": spot * 1.0001,
                }
            )
            records.extend(                              #   + опционы вокруг живого спота
                synth._option_rows(t, int(ts_ns[t]), spot, float(max(var[t], 1e-3)), t, n - 1, expiry_ns, strikes)
            )
        md = pd.DataFrame.from_records(records, columns=MARKET_DATA_COLUMNS)  # опционы → DataFrame
        md["timestamp"] = pd.to_datetime(md["timestamp"])  # метки → datetime
        return md                                        # опционные данные
