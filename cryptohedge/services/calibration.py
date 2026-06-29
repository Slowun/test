"""Heston calibration via filtering + MLE, with Black-Scholes and SABR benchmarks.

* :func:`calibrate_mle` estimates the Heston parameters from the spot time series
  using an EWMA variance *filter* combined with maximum-likelihood estimation of
  the Euler-discretised dynamics (suited to time series, no look-ahead).
* :func:`sabr_calibrate` / :func:`black_scholes_benchmark` provide the two
  benchmark models required for model-risk comparison.
* :func:`parameter_stability` monitors the temporal stability of the calibrated
  parameters.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (калибровка моделей по временному ряду).
НАЗНАЧЕНИЕ:  калибровка Хестона по ряду спота (EWMA-фильтр дисперсии + MLE на
             эйлеровой дискретизации, без заглядывания вперёд); бенчмарки
             Black-Scholes и SABR (для оценки модельного риска); мониторинг
             стабильности параметров во времени.
ИМПОРТИРУЕТ: dataclass; numpy; scipy.optimize.minimize;
             domain.market.HestonParameters; heston_pricing.bs_implied_vol.
ЭКСПОРТИРУЕТ: ewma_variance, calibrate_mle, sabr_lognormal_vol, SABRParameters,
             sabr_calibrate, black_scholes_benchmark, parameter_stability.
КЕМ ИСПОЛЬЗУЕТСЯ: агент heston_calibration.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass                        # для SABRParameters
from typing import Dict, List, Optional, Sequence        # аннотации

import numpy as np                                       # вычисления
from scipy.optimize import minimize                      # оптимизатор MLE/SABR

from cryptohedge.domain.market import HestonParameters   # domain-объект параметров
from cryptohedge.services.heston_pricing import bs_implied_vol  # инверсия в implied vol


# ----------------------------------------------------------------- variance filter
def ewma_variance(returns: np.ndarray, lam: float = 0.94, trading_days: int = 365) -> np.ndarray:
    """Filtered annualised instantaneous variance from returns (RiskMetrics EWMA)."""
    r = np.nan_to_num(np.asarray(returns, float))        # доходности → массив (NaN→0)
    var = np.empty(len(r))                                # массив дисперсий
    var[0] = np.var(r) if np.var(r) > 0 else 1e-6         # стартовая дисперсия
    for t in range(1, len(r)):                            # EWMA-рекурсия (RiskMetrics):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return var * trading_days                             # годовая мгновенная дисперсия


# --------------------------------------------------------------------------- MLE
def calibrate_mle(
    prices: np.ndarray,                                  # ряд цен
    dt: float = 1.0 / 365.0,                             # шаг времени (год)
    flat_yield: float = 0.0,                             # плоская доходность
    init: Optional[Sequence[float]] = None,             # стартовые параметры
    trading_days: int = 365,                            # дней в году
) -> HestonParameters:
    """Maximum-likelihood Heston calibration on a price time series.

    The latent variance is filtered with EWMA; the (return, variance-increment)
    pairs are then modelled as conditionally bivariate-normal under the Euler
    discretisation, and the parameters maximise the joint log-likelihood.
    """
    prices = np.asarray(prices, float)                   # цены → массив
    rets = np.diff(np.log(prices))                       # лог-доходности
    v = ewma_variance(rets, trading_days=trading_days)   # фильтрованная дисперсия
    v = np.clip(v, 1e-6, None)                           # отсекаем нули
    dv = np.diff(v)                                      # приращения дисперсии
    r = rets[1:]                                         # доходности (сдвиг под dv)
    vt = v[:-1]                                          # дисперсия в начале интервала

    if init is None:                                     # стартовые параметры по умолчанию:
        init = [2.0, float(np.mean(v)), 0.5, -0.5, float(np.mean(r) / dt)]

    def neg_ll(p):                                       # отрицательная лог-правдоподобность:
        kappa, theta, eps, rho, mu = p
        if kappa <= 0 or theta <= 0 or eps <= 0 or abs(rho) >= 0.999:  # вне допустимой области…
            return 1e8                                   #   штраф
        m_r = (mu - 0.5 * vt) * dt                       # условное среднее доходности
        m_v = kappa * (theta - vt) * dt                  # условное среднее приращения дисперсии
        var_r = vt * dt                                  # дисперсия доходности
        var_v = eps**2 * vt * dt                         # дисперсия приращения дисперсии
        cov = rho * eps * vt * dt                        # ковариация (через rho)
        det = var_r * var_v - cov**2                     # детерминант ковариационной матрицы
        det = np.where(det <= 0, 1e-12, det)             # защита от вырождения
        dr = r - m_r                                     # отклонение доходности
        dvv = dv - m_v                                   # отклонение дисперсии
        quad = (var_v * dr**2 - 2 * cov * dr * dvv + var_r * dvv**2) / det  # квадратичная форма
        ll = -0.5 * (np.log(det) + quad + 2 * np.log(2 * np.pi))  # лог-плотность би-нормали
        return float(-np.sum(ll))                        # -сумма LL

    res = minimize(neg_ll, init, method="Nelder-Mead",   # минимизация -LL
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    kappa, theta, eps, rho, _mu = res.x                  # оценённые параметры
    v0 = float(v[-1])                                    # текущая дисперсия (последняя)
    return HestonParameters(                             # результат → domain-объект:
        v0=v0,
        kappa=float(abs(kappa)),
        theta=float(abs(theta)),
        eps=float(abs(eps)),
        rho=float(np.clip(rho, -0.999, 0.999)),
        flat_yield=flat_yield,
        calibration_error=float(res.fun),
        feller_satisfied=bool(2 * abs(kappa) * abs(theta) - eps**2 >= 0),  # условие Феллера
    )


# -------------------------------------------------------------------------- SABR
def sabr_lognormal_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """Hagan (2002) lognormal SABR implied volatility approximation."""
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:         # некорректный вход…
        return float("nan")
    if abs(F - K) < 1e-12:                               # ATM-случай (F≈K):
        term = (
            ((1 - beta) ** 2 / 24) * alpha**2 / F ** (2 - 2 * beta)
            + 0.25 * rho * beta * nu * alpha / F ** (1 - beta)
            + (2 - 3 * rho**2) / 24 * nu**2
        )
        return float(alpha / F ** (1 - beta) * (1 + term * T))  # ATM-приближение Хагана
    logFK = np.log(F / K)                               # log(F/K)
    fk_beta = (F * K) ** ((1 - beta) / 2)               # (F·K)^((1-β)/2)
    z = (nu / alpha) * fk_beta * logFK                  # вспомогательная z
    xz = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))  # x(z)
    denom = fk_beta * (1 + ((1 - beta) ** 2 / 24) * logFK**2 + ((1 - beta) ** 4 / 1920) * logFK**4)  # знаменатель
    term = (
        ((1 - beta) ** 2 / 24) * alpha**2 / fk_beta**2
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2 - 3 * rho**2) / 24 * nu**2
    )
    factor = z / xz if abs(xz) > 1e-12 else 1.0         # поправка z/x(z)
    return float((alpha / denom) * factor * (1 + term * T))  # SABR implied vol


@dataclass(frozen=True)
class SABRParameters:                                    # параметры SABR (неизменяемые)
    alpha: float                                        # уровень волатильности
    beta: float                                         # экспонента (фиксируется)
    rho: float                                          # корреляция спот-вол
    nu: float                                           # вол-оф-вол
    rmse: float                                         # ошибка подгонки улыбки

    def to_dict(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "rho": self.rho, "nu": self.nu, "rmse": self.rmse}  # сериализация


def sabr_calibrate(
    forward: float, strikes: np.ndarray, ttm: float, market_iv: np.ndarray, beta: float = 0.5
) -> SABRParameters:
    """Calibrate SABR (alpha, rho, nu) to a market implied-vol smile (beta fixed)."""
    strikes = np.asarray(strikes, float)                # страйки → массив
    market_iv = np.asarray(market_iv, float)            # рыночные IV → массив
    mask = np.isfinite(market_iv) & (market_iv > 0)     # маска валидных IV
    strikes, market_iv = strikes[mask], market_iv[mask] # фильтрация
    if len(strikes) < 3:                                # точек мало…
        return SABRParameters(np.nan, beta, np.nan, np.nan, np.nan)

    atm_iv = market_iv[np.argmin(np.abs(strikes - forward))]  # IV ближайшего к форварду страйка

    def loss(p):                                        # MSE подгонки улыбки:
        alpha, rho, nu = p
        if alpha <= 0 or abs(rho) >= 0.999 or nu < 0:   #   вне допустимой области…
            return 1e6
        model = np.array([sabr_lognormal_vol(forward, k, ttm, alpha, beta, rho, nu) for k in strikes])  # модельные IV
        return float(np.nanmean((model - market_iv) ** 2))  # средний квадрат ошибки

    res = minimize(loss, [atm_iv * forward ** (1 - beta), -0.3, 0.5], method="Nelder-Mead",  # минимизация MSE
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8})
    alpha, rho, nu = res.x                              # оценённые параметры
    return SABRParameters(float(alpha), beta, float(np.clip(rho, -0.999, 0.999)), float(abs(nu)), float(np.sqrt(res.fun)))  # результат


# --------------------------------------------------------------- BS benchmark
def black_scholes_benchmark(
    spot: float, strikes: np.ndarray, ttm: float, market_prices: np.ndarray, is_call: np.ndarray, flat_yield: float = 0.0
) -> Dict[str, float]:
    """Flat-vol Black-Scholes benchmark: ATM IV and smile RMSE vs market."""
    strikes = np.asarray(strikes, float)                # страйки → массив
    ivs = np.array([                                    # implied vol по всем страйкам:
        bs_implied_vol(spot, k, ttm, flat_yield, p, bool(c))
        for k, p, c in zip(strikes, market_prices, is_call)
    ])
    valid = np.isfinite(ivs)                            # валидные IV
    if valid.sum() == 0:                                # инверсия нигде не сошлась…
        return {"atm_iv": float("nan"), "flat_vol": float("nan"), "iv_rmse": float("nan")}
    atm_iv = float(ivs[valid][np.argmin(np.abs(strikes[valid] - spot))])  # ATM IV
    flat_vol = float(np.nanmean(ivs[valid]))            # плоская волатильность (среднее)
    rmse = float(np.sqrt(np.nanmean((ivs[valid] - flat_vol) ** 2)))  # RMSE улыбки относительно плоской
    return {"atm_iv": atm_iv, "flat_vol": flat_vol, "iv_rmse": rmse}


# ----------------------------------------------------------------- stability
def parameter_stability(history: List[HestonParameters], max_rel_change: float = 0.5) -> Dict[str, object]:
    """Quantify temporal stability of calibrated Heston parameters."""
    if len(history) < 2:                                # истории мало…
        return {"stable": True, "max_rel_change": 0.0, "per_param": {}}
    keys = ["v0", "kappa", "theta", "eps", "rho"]       # отслеживаемые параметры
    arr = np.array([[getattr(h, k) for k in keys] for h in history], float)  # матрица [время × параметр]
    rel = np.abs(np.diff(arr, axis=0)) / (np.abs(arr[:-1]) + 1e-8)  # относительные изменения
    per_param = {k: float(np.nanmean(rel[:, i])) for i, k in enumerate(keys)}  # среднее изменение по параметрам
    max_change = float(np.nanmax(rel))                  # макс. относительное изменение
    return {                                            # сводка стабильности:
        "stable": bool(max_change < max_rel_change),
        "max_rel_change": max_change,
        "mean_rel_change": float(np.nanmean(rel)),
        "per_param": per_param,
    }
