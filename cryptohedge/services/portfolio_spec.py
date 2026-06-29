"""Helpers to derive the liability portfolio and the vega-hedge option from data.

Shared by the greeks, hedging and backtesting agents so they all reference the
exact same instruments.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (спецификация хедж-инструментов).
НАЗНАЧЕНИЕ:  выводит из рыночных данных «портфель обязательств» (защитный пут) и
             опцион для вега-хеджа (около-ATM колл). Используется агентами greeks,
             hedging и backtesting, чтобы все ссылались на ОДНИ И ТЕ ЖЕ инструменты.
ИМПОРТИРУЕТ: numpy, pandas; domain.market.OptionContract; base.{INSTR_CALL,
             INSTR_PUT}.
ЭКСПОРТИРУЕТ: option_expiry, available_strikes, nearest, build_hedge_setup.
КЕМ ИСПОЛЬЗУЕТСЯ: агенты greeks_calculation, hedging_decision, backtesting.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import List, Tuple                           # аннотации

import numpy as np                                       # вычисления
import pandas as pd                                       # таблицы

from cryptohedge.domain.market import OptionContract     # domain-объект опциона
from cryptohedge.services.providers.base import INSTR_CALL, INSTR_PUT  # коды инструментов


def option_expiry(market_data: pd.DataFrame) -> int:
    opts = market_data[market_data["instrument_type"].isin([INSTR_CALL, INSTR_PUT])]  # только опционы
    if opts.empty:                                      # опционов нет…
        raise ValueError("No option rows found in market data")
    return int(opts["expiry_ts"].astype("int64").mode().iloc[0])  # наиболее частая экспирация


def available_strikes(market_data: pd.DataFrame, is_call: bool) -> np.ndarray:
    instr = INSTR_CALL if is_call else INSTR_PUT         # выбираем тип инструмента
    return np.sort(market_data[market_data["instrument_type"] == instr]["strike"].unique())  # отсортир. страйки


def nearest(strikes: np.ndarray, target: float) -> float:
    return float(strikes[np.argmin(np.abs(strikes - target))])  # страйк, ближайший к цели


def build_hedge_setup(
    market_data: pd.DataFrame,                          # рыночные данные
    spot0: float,                                       # начальный спот
    quantity_to_hedge: float,                           # количество к хеджированию
    primary_symbol: str,                                # первичный актив
    put_moneyness: float = 0.95,                        # денежность защитного пута
    call_moneyness: float = 1.0,                        # денежность вега-хедж колла
) -> Tuple[List[OptionContract], OptionContract]:
    """Return (liability_portfolio, vega_hedge_option).

    Liability = a protective put on the primary asset, sized to the BTC notional
    that must be hedged. Vega is hedged with a near-ATM call at the same expiry.
    """
    expiry = option_expiry(market_data)                 # единая экспирация
    call_strikes = available_strikes(market_data, True)  # доступные страйки коллов
    put_strikes = available_strikes(market_data, False)  # доступные страйки путов
    if len(call_strikes) == 0 or len(put_strikes) == 0:  # нет какой-то стороны…
        raise ValueError("Need both call and put strikes to build hedge setup")

    vega_strike = nearest(call_strikes, spot0 * call_moneyness)  # страйк около-ATM колла
    put_strike = nearest(put_strikes, spot0 * put_moneyness)     # страйк защитного пута

    vega_option = OptionContract(primary_symbol, vega_strike, expiry, True, notional=1.0)  # опцион вега-хеджа
    liability = [OptionContract(primary_symbol, put_strike, expiry, False, notional=max(quantity_to_hedge, 1.0))]  # обязательство (пут)
    return liability, vega_option                       # (портфель обязательств, вега-хедж)
