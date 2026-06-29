"""Deterministic synthetic market generator.

Produces a fully self-contained, reproducible dataset:

* a 100-asset spot universe with a realistic correlation structure (positively
  correlated, inversely correlated, neutral and a few cointegrated names), driven
  by a Heston-simulated BTC factor;
* a Heston-consistent option chain (calls + puts, single 30-day-ish expiry) for
  the primary symbol at every daily slice, so that Heston re-calibration recovers
  meaningful, slowly drifting parameters.

Everything is a pure function of the configured ``seed``.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services.providers (генератор данных).
НАЗНАЧЕНИЕ:  ДЕТЕРМИНИРОВАННО (из seed) генерирует весь датасет: вселенную из 100
             активов с реалистичной структурой корреляций (положительные,
             обратные, нейтральные, коинтегрированные), управляемую BTC-фактором
             по Хестону; и Хестон-согласованную опционную цепочку BTC на каждом
             дневном срезе. Всё — чистая функция от seed.
ИМПОРТИРУЕТ: math, numpy, pandas; domain.market.HestonParameters;
             services.heston_pricing.heston_premiums; base.* (коды/схема/контракт).
ЭКСПОРТИРУЕТ: SyntheticProvider.
КОНСТАНТЫ:
  - _NS_PER_DAY/_YEAR_DAYS : перевод дней↔наносекунд/годы.
  - DEFAULT_UNIVERSE       : курируемый список тикеров (дополняется синтетикой).
  - _BASE_PRICES           : стартовые ценовые уровни для реализма.
КЕМ ИСПОЛЬЗУЕТСЯ: build_provider (provider="synthetic") и BundledProvider (фолбэк).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import math                                              # элементарная математика (exp/sqrt/sin)
from typing import List, Tuple                           # аннотации

import numpy as np                                       # генерация случайностей и массивы
import pandas as pd                                       # таблицы данных

from cryptohedge.core.config import DataConfig           # конфиг данных
from cryptohedge.domain.market import HestonParameters   # параметры Хестона для опционов
from cryptohedge.services.heston_pricing import heston_premiums  # цены опционов по Хестону
from cryptohedge.services.providers.base import (        # контракт и схема данных:
    INSTR_ASSET,                                         #   код «актив»
    INSTR_CALL,                                          #   код «call»
    INSTR_PUT,                                           #   код «put»
    MARKET_DATA_COLUMNS,                                 #   схема колонок опционов
    MarketDataBundle,                                    #   контейнер данных
    MarketDataProvider,                                  #   абстрактный провайдер
)

_NS_PER_DAY = 86_400_000_000_000                         # наносекунд в сутках
_YEAR_DAYS = 365.0                                       # дней в году (крипта 24/7)

# A curated set of real tickers; padded with synthetic names up to universe_size.
DEFAULT_UNIVERSE: List[str] = [                          # базовый список реальных тикеров
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "TRXUSDT", "BCHUSDT",
    "ATOMUSDT", "XLMUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "NEARUSDT", "ICPUSDT", "INJUSDT", "SUIUSDT", "AAVEUSDT", "UNIUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "EGLDUSDT", "FTMUSDT", "THETAUSDT", "ALGOUSDT", "XTZUSDT",
    "EOSUSDT", "GALAUSDT", "GRTUSDT", "CHZUSDT", "ZECUSDT", "DASHUSDT", "MKRUSDT",
    "SNXUSDT", "COMPUSDT", "CRVUSDT", "1INCHUSDT", "ENJUSDT", "BATUSDT", "ZILUSDT",
    "RUNEUSDT",
]

# Approximate starting price levels (USD) for realism.
_BASE_PRICES = {                                         # стартовые цены (USD) для известных тикеров
    "BTCUSDT": 45_000.0, "ETHUSDT": 2_400.0, "BNBUSDT": 310.0, "SOLUSDT": 100.0,
    "XRPUSDT": 0.6, "ADAUSDT": 0.55, "DOGEUSDT": 0.08, "AVAXUSDT": 35.0,
    "DOTUSDT": 7.5, "LINKUSDT": 15.0, "MATICUSDT": 0.9, "LTCUSDT": 70.0,
}


class SyntheticProvider(MarketDataProvider):
    name = "synthetic"                                   # имя провайдера

    def __init__(self, config: DataConfig, seed: int, n_steps: int = 90) -> None:
        self.config = config                             # конфиг данных
        self.seed = seed                                 # seed генерации
        self.n_steps = int(n_steps)                      # число шагов (дней)
        self.rng = np.random.default_rng(seed)           # детерминированный генератор RNG

    # ------------------------------------------------------------------ universe
    def _universe(self) -> List[str]:
        primary = self.config.primary_symbol             # первичный актив
        source = list(dict.fromkeys(self.config.symbols)) if self.config.symbols else list(DEFAULT_UNIVERSE)  # источник тикеров
        symbols = [s for s in source if s != primary]    # убираем первичный из списка (добавим в начало)
        i = 0                                            # счётчик синтетических имён
        while len(symbols) + 1 < self.config.universe_size:  # пока не набрали нужный размер…
            i += 1                                       #   следующий номер
            cand = f"SYN{i:03d}USDT"                     #   синтетический тикер
            if cand != primary and cand not in symbols:  #   если уникален…
                symbols.append(cand)                     #     добавляем
        symbols = symbols[: max(0, self.config.universe_size - 1)]  # обрезаем до размера-1
        return [primary] + symbols                       # первичный актив первым, затем остальные

    # --------------------------------------------------------------- simulation
    def _simulate_btc(self, n_steps: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Full-truncation Euler Heston path for the BTC factor."""
        kappa, theta, eps, rho, mu, v0 = 3.0, 0.04 * 4, 0.6, -0.6, 0.05, 0.04  # параметры процесса Хестона
        S = np.empty(n_steps + 1)                        # массив цен
        v = np.empty(n_steps + 1)                        # массив дисперсий
        S[0], v[0] = _BASE_PRICES["BTCUSDT"], v0         # начальные цена и дисперсия
        z = self.rng.standard_normal((n_steps, 2))       # пары стандартных нормальных шоков
        for t in range(n_steps):                         # схема Эйлера (full truncation):
            w1 = z[t, 0]                                 #   шок цены
            w2 = rho * z[t, 0] + math.sqrt(1.0 - rho**2) * z[t, 1]  # коррелированный шок дисперсии
            vt = max(v[t], 0.0)                          #   усечённая (неотрицательная) дисперсия
            S[t + 1] = S[t] * math.exp((mu - 0.5 * vt) * dt + math.sqrt(vt * dt) * w1)  # шаг цены
            v[t + 1] = max(v[t] + kappa * (theta - vt) * dt + eps * math.sqrt(vt * dt) * w2, 1e-4)  # шаг дисперсии
        return S, v                                      # путь цены и дисперсии

    def _simulate_universe(self, symbols: List[str], btc_close: np.ndarray) -> pd.DataFrame:
        """Factor model: each asset loads on BTC returns plus idiosyncratic noise."""
        n_steps = len(btc_close) - 1                     # число шагов
        btc_ret = np.diff(np.log(btc_close))             # лог-доходности BTC (общий фактор)
        n_assets = len(symbols)                          # число активов

        # relationship buckets: ~65% positive, ~15% inverse, ~12% neutral, ~8% cointegrated
        betas = np.empty(n_assets)                       # бета к BTC-фактору
        idio = np.empty(n_assets)                        # идиосинкратическая волатильность
        relationship = np.empty(n_assets, dtype=object)  # тип связи с BTC
        for i, sym in enumerate(symbols):                # по каждому активу…
            if sym == self.config.primary_symbol:        #   сам BTC…
                betas[i], idio[i], relationship[i] = 1.0, 0.0, "primary"  #   бета=1, без шума
                continue
            u = self.rng.random()                        #   случайная корзина связи
            if u < 0.65:                                 #   ~65% — положительная связь
                betas[i] = self.rng.uniform(0.4, 1.6)
                idio[i] = self.rng.uniform(0.005, 0.02)
                relationship[i] = "positive"
            elif u < 0.80:                               #   ~15% — обратная связь
                betas[i] = self.rng.uniform(-1.3, -0.3)
                idio[i] = self.rng.uniform(0.005, 0.02)
                relationship[i] = "inverse"
            elif u < 0.92:                               #   ~12% — нейтральная
                betas[i] = self.rng.uniform(-0.05, 0.05)
                idio[i] = self.rng.uniform(0.02, 0.05)
                relationship[i] = "neutral"
            else:                                        #   ~8% — коинтегрированная
                betas[i] = self.rng.uniform(0.8, 1.2)
                idio[i] = self.rng.uniform(0.002, 0.008)
                relationship[i] = "cointegrated"

        eps_mat = self.rng.standard_normal((n_assets, n_steps)) * idio[:, None]  # идиосинкратические шоки
        log_ret = betas[:, None] * btc_ret[None, :] + eps_mat  # доходности = бета·BTC + шум

        base = np.array([_BASE_PRICES.get(s, float(self.rng.uniform(0.2, 250.0))) for s in symbols])  # стартовые цены
        closes = np.empty((n_assets, n_steps + 1))       # матрица цен закрытия
        closes[:, 0] = base                              # стартовые цены в столбце 0
        closes[:, 1:] = base[:, None] * np.exp(np.cumsum(log_ret, axis=1))  # накопление доходностей
        closes[0] = btc_close  # keep exact BTC path     # точный путь BTC в строке 0

        # overlay genuine cointegration for the 'cointegrated' bucket (OU residual)
        for i, rel in enumerate(relationship):           # для коинтегрированных активов…
            if rel == "cointegrated":
                resid = self._ou_residual(n_steps + 1, sigma=base[i] * 0.01)  # OU-остаток (возврат к среднему)
                closes[i] = base[i] / btc_close[0] * btc_close + resid  # цена = масштаб·BTC + стационарный остаток
                closes[i] = np.clip(closes[i], base[i] * 0.2, None)  # отсекаем слишком низкие значения

        return pd.DataFrame(closes.T, columns=symbols)   # таблица цен [время × символ]

    def _ou_residual(self, n: int, sigma: float, kappa: float = 8.0) -> np.ndarray:
        dt = 1.0 / _YEAR_DAYS                            # шаг времени (год)
        x = np.empty(n)                                 # массив остатка
        x[0] = 0.0                                       # старт с нуля
        shocks = self.rng.standard_normal(n - 1)         # шоки OU-процесса
        for t in range(n - 1):                           # процесс Орнштейна-Уленбека:
            x[t + 1] = x[t] - kappa * x[t] * dt + sigma * math.sqrt(dt) * shocks[t]  # возврат к среднему + шум
        return x                                         # стационарный остаток

    @staticmethod
    def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        n = len(close)                                   # число баров
        prev = np.concatenate([[close[0]], close[:-1]])  # цена предыдущего бара (для open)
        open_ = prev                                     # open = close предыдущего бара
        rng_amp = np.abs(rng.normal(0.0, 0.01, n))       # случайная амплитуда для high
        high = np.maximum(open_, close) * (1.0 + rng_amp)  # high выше max(open,close)
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, n)))  # low ниже min(open,close)
        volume = np.abs(rng.normal(1.0, 0.3, n)) * 1_000.0  # синтетический объём
        return open_, high, low, close, volume           # OHLCV

    # ----------------------------------------------------------------- options
    def _drifting_params(self, t: int, n_steps: int, v0: float) -> HestonParameters:
        phase = 2 * math.pi * t / max(n_steps, 1)        # фаза медленного дрейфа параметров во времени
        kappa = 3.0 * (1.0 + 0.10 * math.sin(phase))     # дрейфующая каппа
        theta = 0.16 * (1.0 + 0.15 * math.sin(phase + 0.7))  # дрейфующая тета
        eps = 0.6 * (1.0 + 0.10 * math.cos(phase))       # дрейфующий эпсилон
        rho = -0.6 + 0.05 * math.sin(phase + 1.3)        # дрейфующая ро
        return HestonParameters(v0=float(max(v0, 1e-3)), kappa=kappa, theta=theta, eps=eps, rho=rho,  # параметры среза
                                flat_yield=0.0)

    def _option_rows(
        self,
        sample_idx: int,                                 # индекс среза
        ts_ns: int,                                      # метка времени, нс
        spot: float,                                     # спот на срезе
        v0: float,                                       # дисперсия на срезе
        t_index: int,                                    # индекс времени (для дрейфа параметров)
        n_steps: int,                                    # всего шагов
        expiry_ns: int,                                  # время экспирации, нс
        strikes: np.ndarray,                             # сетка страйков
    ) -> List[dict]:
        ttm = (expiry_ns - ts_ns) / (_NS_PER_DAY * _YEAR_DAYS)  # срок до экспирации в годах
        if ttm <= 0:                                     # опцион уже истёк…
            return []                                    #   → нет строк
        params = self._drifting_params(t_index, n_steps, v0)  # параметры Хестона на срезе
        rows: List[dict] = []                            # строки опционов
        for is_call, instr in ((True, INSTR_CALL), (False, INSTR_PUT)):  # для call и put…
            calls = np.full(len(strikes), is_call)       # массив флагов call/put
            premia_usd = heston_premiums(spot, strikes, np.full(len(strikes), ttm), calls, params)  # цены опционов (USD)
            coin_px = premia_usd / spot  # quote in units of the underlying (Deribit convention)  # цена в монетах
            for k, px in zip(strikes, coin_px):          # по каждому страйку…
                if not np.isfinite(px) or px <= 0:       #   некорректная цена…
                    continue                             #     пропускаем
                spread = 0.02 + 0.01 * self.rng.random()  # синтетический спред
                bid = px * (1.0 - spread / 2)            #   цена покупки
                ask = px * (1.0 + spread / 2)            #   цена продажи
                liq = float(np.abs(self.rng.normal(50.0, 15.0)))  # синтетическая ликвидность bid
                rows.append(                             # строка опциона (схема MARKET_DATA_COLUMNS)
                    {
                        "sample_idx": sample_idx,
                        "timestamp": ts_ns,
                        "instrument_type": instr,
                        "strike": float(k),
                        "expiry_ts": expiry_ns,
                        "time_to_maturity": float(ttm),
                        "price": float(px),
                        "best_bid_price": float(bid),
                        "best_ask_price": float(ask),
                        "bid_amount_total": liq,
                        "ask_amount_total": float(np.abs(self.rng.normal(50.0, 15.0))),
                        "bid_vwap": float(bid),
                        "ask_vwap": float(ask),
                    }
                )
        return rows                                      # строки опционов среза

    # -------------------------------------------------------------------- build
    def load(self) -> MarketDataBundle:
        symbols = self._universe()                       # строим вселенную символов
        n_steps = int(self.n_steps)                      # число шагов
        dt = 1.0 / _YEAR_DAYS                            # шаг времени (год)

        btc_close, btc_var = self._simulate_btc(n_steps, dt)  # симулируем путь BTC (цена + дисперсия)
        closes = self._simulate_universe(symbols, btc_close)  # симулируем цены всей вселенной

        start = pd.Timestamp("2024-01-02 00:00:00")      # стартовая дата
        timestamps = pd.date_range(start, periods=n_steps + 1, freq="D")  # дневные метки времени
        ts_ns = timestamps.astype("int64").to_numpy()    # метки в наносекундах

        # ---- spot bars (long OHLCV) + wide close matrix  # --- OHLCV-бары + широкая таблица цен ---
        bar_frames = []                                  # фреймы баров по символам
        ohlc_rng = np.random.default_rng(self.seed + 7)  # отдельный RNG для OHLC (детерминирован)
        for sym in symbols:                              # по каждому символу…
            o, h, l, c, vol = self._ohlc_from_close(closes[sym].to_numpy(), ohlc_rng)  # OHLCV из цен закрытия
            bar_frames.append(                           # фрейм баров символа
                pd.DataFrame(
                    {"timestamp": timestamps, "symbol": sym, "open": o, "high": h,
                     "low": l, "close": c, "volume": vol}
                )
            )
        spot_bars = pd.concat(bar_frames, ignore_index=True)  # объединяем все бары (длинный формат)
        spot_close = closes.copy()                       # широкая таблица цен закрытия
        spot_close.index = timestamps                    # индекс по времени
        spot_close.index.name = "timestamp"              # имя индекса

        # ---- option market data for the primary symbol  # --- опционные данные первичного актива ---
        primary = self.config.primary_symbol             # первичный актив
        spot_path = closes[primary].to_numpy()           # путь цены первичного актива
        expiry_ns = int(ts_ns[-1] + self.config.option_expiry_days * _NS_PER_DAY)  # время экспирации опционов
        strikes = self._strike_grid(spot_path[0])        # сетка страйков от стартовой цены

        records: List[dict] = []                         # все строки опционных данных
        for t in range(n_steps + 1):                     # по каждому срезу…
            spot = float(spot_path[t])                   #   спот на срезе
            records.append(                              #   строка базового актива (спот)
                {
                    "sample_idx": t, "timestamp": int(ts_ns[t]), "instrument_type": INSTR_ASSET,
                    "strike": 0.0, "expiry_ts": 0, "time_to_maturity": 0.0, "price": spot,
                    "best_bid_price": spot * 0.9999, "best_ask_price": spot * 1.0001,
                    "bid_amount_total": 100.0, "ask_amount_total": 100.0,
                    "bid_vwap": spot * 0.9999, "ask_vwap": spot * 1.0001,
                }
            )
            records.extend(                              #   + строки опционов среза
                self._option_rows(t, int(ts_ns[t]), spot, float(btc_var[t]), t, n_steps, expiry_ns, strikes)
            )

        option_market_data = pd.DataFrame.from_records(records, columns=MARKET_DATA_COLUMNS)  # опционы → DataFrame
        option_market_data["timestamp"] = pd.to_datetime(option_market_data["timestamp"])  # метки → datetime

        meta = {                                         # метаданные датасета
            "provider": self.name,
            "seed": self.seed,
            "n_samples": n_steps + 1,
            "expiry_ts": expiry_ns,
            "strikes": strikes.tolist(),
            "universe_size": len(symbols),
        }
        return MarketDataBundle(                         # собираем и валидируем пакет
            spot_bars=spot_bars,
            spot_close=spot_close,
            option_market_data=option_market_data,
            symbols=symbols,
            primary_symbol=primary,
            meta=meta,
        ).validate()

    def _strike_grid(self, spot0: float) -> np.ndarray:
        n = self.config.n_strikes_per_expiry             # число страйков
        width = self.config.strike_width_pct             # полуширина сетки (доля от спота)
        lo, hi = spot0 * (1 - width), spot0 * (1 + width)  # границы сетки страйков
        raw = np.linspace(lo, hi, n)                     # равномерная сетка
        step = 10 ** max(0, int(math.floor(math.log10(spot0))) - 2)  # шаг округления (по порядку цены)
        return np.unique(np.round(raw / step) * step)    # округлённые уникальные страйки
