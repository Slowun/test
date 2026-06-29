"""Adaptive stop-loss and trailing-stop logic.

The stop distance blends three risk signals - ATR (price action), parametric VaR
(tail risk) and the Heston instantaneous volatility (model risk) - and is clamped
to a configured band. The :class:`TrailingStop` ratchets the level as the trade
moves in the holder's favour and is re-calibrated on each update.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (логика стоп-лоссов).
НАЗНАЧЕНИЕ:  адаптивный стоп-лосс и трейлинг-стоп. Дистанция стопа — максимум из
             трёх сигналов риска: ATR (ценовое движение), параметрический VaR
             (хвостовой риск) и мгновенная волатильность Хестона (модельный риск),
             зажатый в заданный коридор [min_stop_pct, max_stop_pct]. TrailingStop
             «подтягивает» уровень при движении в пользу позиции.
ИМПОРТИРУЕТ: numpy; core.config.StopLossConfig; domain.decisions.StopLevel.
ЭКСПОРТИРУЕТ: average_true_range, adaptive_stop, TrailingStop.
КЕМ ИСПОЛЬЗУЕТСЯ: агент risk_management.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import numpy as np                                       # вычисления

from cryptohedge.core.config import StopLossConfig       # конфиг стоп-лоссов
from cryptohedge.domain.decisions import StopLevel       # domain-объект уровня стопа


def average_true_range(high, low, close, window: int = 14) -> np.ndarray:
    """Wilder's Average True Range (NaN-padded to the input length)."""
    high = np.asarray(high, float)                       # максимумы → массив
    low = np.asarray(low, float)                         # минимумы → массив
    close = np.asarray(close, float)                     # закрытия → массив
    prev_close = np.concatenate([[close[0]], close[:-1]])  # предыдущие закрытия
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])  # True Range
    atr = np.full_like(tr, np.nan)                       # массив ATR (изначально NaN)
    if len(tr) >= window:                               # если данных хватает…
        atr[window - 1] = tr[:window].mean()           #   первое значение = среднее TR
        for i in range(window, len(tr)):                #   сглаживание Уайлдера:
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr                                          # ряд ATR


def adaptive_stop(
    reference_price: float,                             # опорная цена
    atr_value: float,                                  # значение ATR
    daily_var: float,                                  # дневной VaR
    heston_vol: float,                                 # мгновенная волатильность Хестона
    side: str,                                         # сторона позиции: long/short
    config: StopLossConfig,                            # конфиг стопов
) -> StopLevel:
    """Compute an adaptive stop level for a long/short position."""
    atr_pct = (config.atr_multiplier * atr_value / reference_price) if reference_price > 0 else 0.0  # дистанция по ATR
    var_pct = config.var_multiplier * float(daily_var)  # дистанция по VaR
    vol_pct = config.var_multiplier * float(heston_vol)  # heston daily vol = sqrt(v0/365)  # дистанция по вол Хестона

    distance = max(atr_pct, var_pct, vol_pct)           # берём наибольший (консервативный) сигнал
    distance = float(np.clip(distance, config.min_stop_pct, config.max_stop_pct))  # зажимаем в коридор

    if side == "long":                                 # длинная позиция…
        stop_price = reference_price * (1.0 - distance)  #   стоп ниже цены
    else:                                              # короткая позиция…
        stop_price = reference_price * (1.0 + distance)  #   стоп выше цены

    return StopLevel(                                  # собираем уровень стопа:
        instrument="primary",
        side=side,  # type: ignore[arg-type]
        stop_price=float(stop_price),
        reference_price=float(reference_price),
        distance_pct=distance,
        method="atr+var+heston",
        components={"atr_pct": atr_pct, "var_pct": var_pct, "heston_vol_pct": vol_pct},  # вклад каждого сигнала
    )


class TrailingStop:
    """Stateful trailing stop that re-calibrates against live volatility."""

    def __init__(self, side: str, entry_price: float, config: StopLossConfig) -> None:
        self.side = side                               # сторона позиции
        self.config = config                           # конфиг стопов
        self.entry_price = entry_price                 # цена входа
        self.extreme = entry_price  # highest (long) / lowest (short) price seen  # экстремум цены
        self.stop_price = entry_price                  # текущий уровень стопа
        self.triggered = False                         # сработал ли стоп

    def update(self, price: float, atr_value: float, daily_var: float, heston_vol: float) -> StopLevel:
        if self.side == "long":                        # обновляем экстремум:
            self.extreme = max(self.extreme, price)    #   максимум для long
        else:
            self.extreme = min(self.extreme, price)    #   минимум для short

        atr_pct = (self.config.trailing_atr_multiplier * atr_value / price) if price > 0 else 0.0  # дистанция по ATR
        distance = float(np.clip(max(atr_pct, self.config.var_multiplier * max(daily_var, heston_vol)),  # дистанция
                                 self.config.min_stop_pct, self.config.max_stop_pct))

        if self.side == "long":                        # длинная позиция…
            new_stop = self.extreme * (1.0 - distance)  #   стоп от максимума
            self.stop_price = max(self.stop_price, new_stop)  #   стоп только растёт (ratchet)
            self.triggered = price <= self.stop_price  #   срабатывает при падении ниже стопа
        else:                                          # короткая позиция…
            new_stop = self.extreme * (1.0 + distance)  #   стоп от минимума
            self.stop_price = min(self.stop_price, new_stop)  #   стоп только снижается
            self.triggered = price >= self.stop_price  #   срабатывает при росте выше стопа

        return StopLevel(                              # собираем уровень стопа:
            instrument="primary",
            side=self.side,  # type: ignore[arg-type]
            stop_price=float(self.stop_price),
            reference_price=float(price),
            distance_pct=distance,
            method="trailing",
            components={"extreme": float(self.extreme), "atr_pct": atr_pct, "triggered": float(self.triggered)},
        )
