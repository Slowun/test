"""Volatility estimation and primary-instrument hedge sizing.

Computes daily volatility, volatility-of-volatility and a confidence interval for
the variance (chi-square based), then converts the investor's capital and risk
budget into the BTC notional that must be hedged.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (вычислительный use-case волатильности).
НАЗНАЧЕНИЕ:  оценивает дневную волатильность, вол-оф-вол и доверительный интервал
             для дисперсии (на основе хи-квадрат), затем переводит капитал инвестора
             и его риск-бюджет в нотионал BTC, который необходимо захеджировать.
ИМПОРТИРУЕТ: dataclass; numpy; scipy.stats; domain.market.VolatilityEstimate.
ЭКСПОРТИРУЕТ: log_returns, estimate_volatility, HedgeSizing, size_primary_hedge.
КЕМ ИСПОЛЬЗУЕТСЯ: агент market_analysis (оценка волатильности и сайзинг хеджа).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass                        # для HedgeSizing (value object)
from typing import Dict                                  # аннотации

import numpy as np                                       # вычисления
from scipy import stats                                  # хи-квадрат и нормальные квантили

from cryptohedge.domain.market import VolatilityEstimate  # domain-объект оценки волатильности


def log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)             # цены → массив float
    return np.diff(np.log(prices))                       # лог-доходности


def estimate_volatility(
    prices: np.ndarray,                                  # ряд цен
    window: int = 30,                                    # окно реализованной волатильности
    vov_window: int = 30,                                # окно вол-оф-вол
    confidence_level: float = 0.95,                      # уровень доверия для CI
    horizon_days: int = 1,                               # горизонт (дней) для масштабирования
    trading_days: int = 365,                             # дней в году (крипта 24/7)
) -> VolatilityEstimate:
    """Estimate daily vol, vol-of-vol and a chi-square CI for the daily vol."""
    rets = log_returns(prices)                           # лог-доходности
    if len(rets) < 2:                                    # данных слишком мало…
        raise ValueError("Need at least 3 prices to estimate volatility")

    daily_vol = float(np.std(rets, ddof=1))              # дневная волатильность (выборочная)
    annualized = daily_vol * np.sqrt(trading_days)       # годовая волатильность

    # rolling realised vol series -> vol of vol           # скользящая реализ. вол → вол-оф-вол
    n = len(rets)                                         # число доходностей
    win = min(window, max(2, n // 2))                    # адаптивное окно
    rolling = np.array([np.std(rets[max(0, i - win):i], ddof=1) for i in range(win, n + 1)])  # скользящая вол
    vol_of_vol = float(np.std(rolling, ddof=1)) if len(rolling) > 1 else 0.0  # вол-оф-вол

    # chi-square CI for the standard deviation            # доверительный интервал (хи-квадрат)
    dof = len(rets) - 1                                   # степени свободы
    alpha = 1.0 - confidence_level                        # уровень значимости
    chi2_low = stats.chi2.ppf(alpha / 2, dof)            # нижний квантиль хи-квадрат
    chi2_high = stats.chi2.ppf(1 - alpha / 2, dof)       # верхний квантиль хи-квадрат
    var = daily_vol**2                                   # дисперсия
    ci_low = float(np.sqrt(dof * var / chi2_high))       # нижняя граница вол
    ci_high = float(np.sqrt(dof * var / chi2_low))       # верхняя граница вол

    scale = np.sqrt(horizon_days)                        # масштаб по горизонту (√t)
    return VolatilityEstimate(                            # собираем domain-оценку:
        daily_vol=daily_vol * scale,
        annualized_vol=annualized,
        vol_of_vol=vol_of_vol,
        ci_low=ci_low * scale,
        ci_high=ci_high * scale,
        confidence_level=confidence_level,
        horizon_days=horizon_days,
    )


@dataclass(frozen=True)
class HedgeSizing:                                        # результат сайзинга хеджа (неизменяемый)
    capital_usd: float                                   # капитал, USD
    spot: float                                          # спот базового актива
    daily_vol_used: float                                # использованная дневная вол
    confidence_z: float                                  # z-квантиль уровня доверия
    unhedged_var_pct: float                              # VaR без хеджа (доля)
    target_var_pct: float                                # целевой VaR (риск-бюджет)
    hedge_ratio: float                                   # доля капитала под хедж
    notional_to_hedge_usd: float                         # нотионал хеджа, USD
    quantity_to_hedge: float                             # количество базового актива

    def to_dict(self) -> Dict[str, float]:
        return {                                         # сериализация в словарь (для блэкборда/логов)
            "capital_usd": self.capital_usd,
            "spot": self.spot,
            "daily_vol_used": self.daily_vol_used,
            "confidence_z": self.confidence_z,
            "unhedged_var_pct": self.unhedged_var_pct,
            "target_var_pct": self.target_var_pct,
            "hedge_ratio": self.hedge_ratio,
            "notional_to_hedge_usd": self.notional_to_hedge_usd,
            "quantity_to_hedge": self.quantity_to_hedge,
        }


def size_primary_hedge(
    capital_usd: float,                                  # капитал инвестора
    spot: float,                                         # спот
    vol: VolatilityEstimate,                             # оценка волатильности
    risk_budget_pct: float,                              # целевой риск-бюджет (доля)
    confidence_level: float = 0.95,                      # уровень доверия для VaR
    use_ci_high: bool = True,                            # использовать верхнюю границу вол (консервативно)
) -> HedgeSizing:
    """Determine how much BTC notional must be hedged to respect the risk budget.

    Assumes the capital is exposed to BTC. The unhedged one-day parametric VaR is
    ``z * sigma``; if it exceeds the risk budget the excess fraction of capital is
    hedged.
    """
    z = float(stats.norm.ppf(confidence_level))          # z-квантиль (например, 1.645 для 95%)
    sigma = vol.ci_high if use_ci_high else vol.daily_vol  # консервативная или точечная вол
    unhedged = z * sigma                                 # параметрический VaR без хеджа
    target = float(risk_budget_pct)                      # целевой риск
    hedge_ratio = float(np.clip(1.0 - target / unhedged, 0.0, 1.0)) if unhedged > 0 else 0.0  # доля под хедж
    notional = hedge_ratio * capital_usd                 # нотионал хеджа
    return HedgeSizing(                                  # собираем результат:
        capital_usd=capital_usd,
        spot=spot,
        daily_vol_used=sigma,
        confidence_z=z,
        unhedged_var_pct=unhedged,
        target_var_pct=target,
        hedge_ratio=hedge_ratio,
        notional_to_hedge_usd=notional,
        quantity_to_hedge=notional / spot if spot > 0 else 0.0,
    )
