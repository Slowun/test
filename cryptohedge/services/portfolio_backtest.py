"""Portfolio diversification analytics and a rebalanced backtest engine.

This module turns a *one-shot* set of optimiser weights into a realistic, time
evolving portfolio: between rebalancing dates the weights drift with asset
returns, and on each rebalancing date the target weights are recomputed from a
trailing window and applied subject to transaction costs. It also exposes the
standard diversification diagnostics used to *confirm* that the resulting
portfolio is well diversified (diversification ratio, effective number of bets,
Herfindahl-Hirschman concentration).

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (бэктест портфеля с ребалансировкой).
НАЗНАЧЕНИЕ:  превращает одномоментный набор весов в реалистичный портфель во
             времени: между датами ребалансировки веса дрейфуют с доходностями, а
             на каждой дате пересчитываются по скользящему окну и применяются с
             учётом издержек. Плюс диагностика диверсификации (коэф. диверсификации,
             эффективное число активов, концентрация HHI).
ИМПОРТИРУЕТ: dataclass; numpy, pandas.
ЭКСПОРТИРУЕТ: diversification_ratio, effective_number_of_bets, herfindahl_index,
             diversification_report, PortfolioBacktest, backtest_rebalanced.
КЕМ ИСПОЛЬЗУЕТСЯ: агент portfolio_optimization.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass, asdict                # для PortfolioBacktest
from typing import Callable, Dict, List, Optional        # аннотации

import numpy as np                                       # вычисления
import pandas as pd                                       # таблицы


# --------------------------------------------------------------- diversification
def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Weighted average volatility divided by portfolio volatility (>= 1).

    A value of 1 means no diversification (a single bet); the higher the ratio,
    the larger the variance-reduction benefit of combining the assets.
    """
    w = np.asarray(weights, float)                      # веса → массив
    cov = np.asarray(cov, float)                        # ковариация → массив
    sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))   # волатильности активов
    port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))    # волатильность портфеля
    if port_vol <= 1e-12:                               # вырожденный портфель…
        return 1.0
    return float((w @ sigma) / port_vol)                # коэффициент диверсификации


def effective_number_of_bets(weights: np.ndarray) -> float:
    """Inverse Herfindahl index: number of *equally weighted* equivalent holdings."""
    w = np.asarray(weights, float)                      # веса → массив
    hhi = float(np.sum(w**2))                           # индекс Херфиндаля
    return float(1.0 / hhi) if hhi > 0 else 0.0         # эффективное число активов


def herfindahl_index(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)                      # веса → массив
    return float(np.sum(w**2))                          # индекс концентрации HHI


def diversification_report(weights: np.ndarray, cov: np.ndarray) -> Dict[str, float]:
    w = np.asarray(weights, float)                      # веса → массив
    n = int(np.sum(w > 1e-6))                           # число активных позиций
    return {                                            # сводка диверсификации:
        "diversification_ratio": diversification_ratio(w, cov),
        "effective_n": effective_number_of_bets(w),
        "n_active": n,
        "max_weight": float(np.max(w)) if len(w) else 0.0,
        "hhi": herfindahl_index(w),
    }


# ------------------------------------------------------------------- backtest
@dataclass(frozen=True)
class PortfolioBacktest:
    """Result of a rebalanced portfolio backtest."""

    equity: pd.Series                # portfolio value, base 1.0  # кривая капитала (база 1.0)
    weights_path: pd.DataFrame       # drifting weights per asset over time  # эволюция весов
    rebalance_dates: List[pd.Timestamp]  # даты ребалансировок
    returns: pd.Series               # net daily portfolio returns  # дневные доходности (нетто)
    turnover: pd.Series              # per-day turnover at rebalances  # оборот по дням
    cum_cost: pd.Series              # cumulative transaction cost (fraction of capital)  # накопл. издержки
    metrics: Dict[str, float]        # итоговые метрики

    def to_summary(self) -> Dict[str, float]:
        return dict(self.metrics)                       # копия метрик


def backtest_rebalanced(
    prices: pd.DataFrame,                              # широкая таблица цен (даты × активы)
    weight_fn: Callable[[pd.DataFrame], np.ndarray],   # функция весов от окна доходностей
    rebalance_days: int = 5,                          # частота ребалансировки (дни)
    fee_pct: float = 0.0003,                          # пропорциональная издержка
    lookback: int = 30,                               # окно оценки весов (дни)
    periods_per_year: int = 365,                      # периодов в году
) -> PortfolioBacktest:
    """Backtest a periodically rebalanced long portfolio.

    Parameters
    ----------
    prices: wide price matrix (index = dates, columns = assets).
    weight_fn: maps a trailing *returns* window to a target weight vector
        (aligned to ``prices.columns``). Called at every rebalancing date.
    rebalance_days: rebalance cadence in days.
    fee_pct: proportional transaction cost charged on traded weight.
    lookback: trailing window (in days) used to estimate the weights.
    """
    prices = prices.dropna(axis=1, how="any")          # убираем активы с пропусками
    cols = list(prices.columns)                        # список активов
    n = len(cols)                                      # число активов
    rets = prices.pct_change().fillna(0.0)             # дневные доходности
    dates = prices.index                               # даты

    if n == 0 or len(dates) < 3:                       # данных недостаточно…
        empty = pd.Series(dtype=float)
        return PortfolioBacktest(empty, pd.DataFrame(), [], empty, empty, empty, {})

    w = np.ones(n) / n                      # start equally weighted  # старт: равные веса
    first = min(lookback, len(dates) - 2)              # первая возможная ребалансировка
    equity = [1.0]                                     # кривая капитала
    weights_path = [w.copy()]                          # история весов
    port_rets = [0.0]                                  # доходности портфеля
    turnover_series = [0.0]                            # оборот
    cum_cost = [0.0]                                   # накопленные издержки
    rebalance_dates: List[pd.Timestamp] = [dates[0]]   # даты ребалансировок
    total_cost = 0.0                                   # суммарные издержки

    for t in range(1, len(dates)):                     # по каждому дню…
        r_t = rets.iloc[t].to_numpy()                  #   доходности дня
        # drift weights with realised returns, then renormalise  # дрейф весов:
        w_drift = w * (1.0 + r_t)                      #   рост каждой позиции
        gross = float(np.sum(w_drift))                 #   общая стоимость
        w_drift = w_drift / gross if gross > 1e-12 else np.ones(n) / n  #   нормировка
        port_ret = float(np.sum(w * r_t))   # return earned over [t-1, t] with start weights  # доходность портфеля

        cost = 0.0                                     #   издержки дня
        do_rebalance = (t >= first) and ((t - first) % max(rebalance_days, 1) == 0)  # день ребалансировки?
        if do_rebalance:                               #   ребалансировка:
            train = rets.iloc[max(0, t - lookback): t]  #     скользящее окно доходностей
            try:
                w_target = np.asarray(weight_fn(train), float)  #     целевые веса от стратегии
                if w_target.shape[0] != n or not np.all(np.isfinite(w_target)):  # некорректны…
                    w_target = w_drift                  #       → оставляем дрейф
            except Exception:                           #     ошибка стратегии…
                w_target = w_drift                      #       → оставляем дрейф
            s = np.sum(w_target)                       #     сумма весов
            w_target = w_target / s if s > 1e-12 else np.ones(n) / n  #     нормировка
            trade = float(np.sum(np.abs(w_target - w_drift)))  #     оборот (изменение весов)
            cost = trade * fee_pct                      #     издержки сделки
            total_cost += cost                          #     накапливаем издержки
            turnover_series.append(trade)               #     лог оборота
            rebalance_dates.append(dates[t])            #     лог даты
            w = w_target                                #     новые веса
        else:                                          #   без ребалансировки:
            turnover_series.append(0.0)                 #     оборота нет
            w = w_drift                                 #     веса = дрейф

        equity.append(equity[-1] * (1.0 + port_ret - cost))  #   обновляем капитал
        port_rets.append(port_ret - cost)              #   доходность за вычетом издержек
        weights_path.append(w.copy())                  #   лог весов
        cum_cost.append(total_cost)                    #   лог накопленных издержек

    equity_s = pd.Series(equity, index=dates, name="equity")  # кривая капитала
    weights_df = pd.DataFrame(weights_path, index=dates, columns=cols)  # эволюция весов
    rets_s = pd.Series(port_rets, index=dates, name="returns")  # доходности
    turnover_s = pd.Series(turnover_series, index=dates, name="turnover")  # оборот
    cost_s = pd.Series(cum_cost, index=dates, name="cum_cost")  # издержки

    final_w = weights_df.iloc[-1].to_numpy()           # финальные веса
    cov = rets.cov().to_numpy() * periods_per_year      # годовая ковариация
    metrics = _equity_metrics(equity_s, rets_s, periods_per_year)  # метрики капитала
    metrics.update(diversification_report(final_w, cov))  # + диверсификация
    metrics["n_rebalances"] = int(max(0, len(rebalance_dates) - 1))  # число ребалансировок
    metrics["total_cost"] = float(total_cost)          # суммарные издержки
    metrics["avg_diversification_ratio"] = _avg_div_ratio(weights_df, cov)  # средний коэф. диверсификации

    return PortfolioBacktest(equity_s, weights_df, rebalance_dates, rets_s,  # результат бэктеста
                             turnover_s, cost_s, metrics)


def _avg_div_ratio(weights_df: pd.DataFrame, cov: np.ndarray) -> float:
    vals = [diversification_ratio(weights_df.iloc[i].to_numpy(), cov)  # коэф. диверсификации по точкам
            for i in range(0, len(weights_df), max(1, len(weights_df) // 20))]  # ~20 равномерных точек
    return float(np.mean(vals)) if vals else 1.0       # средний коэффициент


def _equity_metrics(equity: pd.Series, rets: pd.Series, ppy: int) -> Dict[str, float]:
    r = rets.to_numpy()[1:]                            # доходности (без стартового 0)
    if len(r) == 0:                                    # данных нет…
        return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "volatility": 0.0, "max_drawdown": 0.0}
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)  # совокупная доходность
    n_years = len(r) / ppy                             # длительность в годах
    cagr = float((1.0 + total_return) ** (1.0 / n_years) - 1.0) if n_years > 0 and total_return > -1 else 0.0  # CAGR
    vol = float(np.std(r, ddof=1) * np.sqrt(ppy)) if len(r) > 1 else 0.0  # годовая волатильность
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(ppy)) if len(r) > 1 and np.std(r, ddof=1) > 0 else 0.0  # Шарп
    running_max = np.maximum.accumulate(equity.to_numpy())  # текущий максимум
    dd = (equity.to_numpy() - running_max) / running_max  # просадки
    mdd = float(np.min(dd)) if len(dd) else 0.0        # максимальная просадка
    return {"total_return": total_return, "cagr": cagr, "sharpe": sharpe,  # сводка метрик
            "volatility": vol, "max_drawdown": mdd}
