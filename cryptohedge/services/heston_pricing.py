"""Thin, NumPy-friendly wrapper around the numba ``pyquant`` Heston engine.

This isolates the rest of the codebase from the low-level jitclass API. It exposes
vectorised premium pricing, implied-vol inversion and implied-volatility-surface
calibration (Levenberg-Marquardt least squares on option premiums), returning the
plain :class:`cryptohedge.domain.HestonParameters` value object.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (вычислительный use-case ценообразования).
НАЗНАЧЕНИЕ:  тонкая NumPy-обёртка над numba-движком pyquant (Heston/Black-Scholes).
             Скрывает низкоуровневый jitclass-API: даёт векторное ценообразование
             премий, инверсию implied-vol и калибровку по поверхности IV
             (Левенберг-Марквардт, МНК по премиям), возвращая domain-объект
             HestonParameters.
ИМПОРТИРУЕТ: numpy; domain.market.HestonParameters; примитивы pyquant.common,
             pyquant.black_scholes.BSCalc, pyquant.heston.*, pyquant.vol_surface.
ЭКСПОРТИРУЕТ: heston_premiums, bs_implied_vol, calibrate_iv_surface, heston_atm_iv.
КОНСТАНТЫ:   _HESTON / _BS — переиспользуемые синглтоны движков pyquant.
КЕМ ИСПОЛЬЗУЕТСЯ: агенты heston_calibration, greeks_calculation, hedging_decision;
             провайдеры synthetic/binance (генерация опционов).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import Optional, Sequence, Tuple             # аннотации

import numpy as np                                       # массивы и векторизация

from cryptohedge.domain.market import HestonParameters   # domain-объект параметров Хестона

# pyquant numba primitives ----------------------------------------------------
from pyquant.common import (                             # обёртки над величинами (типобезопасность):
    ForwardRates,                                        #   форвардные ставки
    ForwardYield,                                        #   форвардная доходность
    DiscountYield,                                       #   дисконтная доходность
    Forward,                                             #   форвард
    Spot,                                                #   спот
    Strike,                                              #   страйк
    Strikes,                                             #   массив страйков
    StrikesMaturitiesGrid,                               #   сетка страйк×срок
    OptionTypes,                                         #   типы опционов (call/put)
    Premium,                                             #   премия
    Premiums,                                            #   массив премий
    TimeToMaturity,                                      #   срок до экспирации
    TimesToMaturity,                                     #   массив сроков
    CalibrationWeights,                                  #   веса калибровки
    forward_curve_from_forward_rates,                    #   построение форвардной кривой
)
from pyquant.black_scholes import BSCalc                 # движок Black-Scholes (implied vol)
from pyquant.heston import (                             # движок Хестона и его параметры:
    HestonCalc,                                          #   калькулятор Хестона
    HestonParams,                                        #   параметры Хестона (pyquant)
    Variance,                                            #   v0
    VarReversion,                                        #   kappa
    AverageVar,                                          #   theta
    VolOfVar,                                            #   eps
    Correlation,                                         #   rho
    FlatForwardYield,                                    #   плоская доходность
)
from pyquant.vol_surface import VolSurfaceChainSpace     # пространство опционной цепочки для калибровки

_HESTON = HestonCalc()                                   # синглтон движка Хестона (переиспользуем)
_BS = BSCalc()                                           # синглтон движка Black-Scholes


def _heston_params(p: HestonParameters) -> HestonParams:
    return HestonParams(                                 # конвертация domain → pyquant-параметры:
        Variance(p.v0),                                  #   v0
        VarReversion(p.kappa),                           #   kappa
        AverageVar(p.theta),                             #   theta
        VolOfVar(p.eps),                                 #   eps
        Correlation(p.rho),                              #   rho
        FlatForwardYield(p.flat_yield),                  #   доходность
    )


def heston_premiums(
    spot: float,                                         # спот базового актива
    strikes: Sequence[float],                            # страйки
    ttm: Sequence[float],                                # сроки до экспирации
    is_call: Sequence[bool],                             # флаги call/put
    params: HestonParameters,                            # параметры Хестона
) -> np.ndarray:
    """Vectorised Heston premiums for a set of (strike, maturity, type) points."""
    Ks = np.asarray(strikes, dtype=np.float64)           # страйки → массив
    Ts = np.asarray(ttm, dtype=np.float64)               # сроки → массив
    calls = np.asarray(is_call, dtype=np.bool_)          # флаги → булев массив
    grid = StrikesMaturitiesGrid(Spot(float(spot)), TimesToMaturity(Ts), Strikes(Ks))  # сетка точек
    return np.asarray(_HESTON._grid_premiums(_heston_params(params), grid, OptionTypes(calls)))  # премии


def bs_implied_vol(
    spot: float, strike: float, ttm: float, rate: float, premium: float, is_call: bool
) -> float:
    """Black-Scholes implied volatility from a premium (NaN if it cannot invert)."""
    if premium <= 0 or ttm <= 0:                         # некорректный вход…
        return float("nan")                              #   → NaN
    fwd = Forward(Spot(float(spot)), ForwardYield(float(rate)), DiscountYield(float(rate)), TimeToMaturity(float(ttm)))  # форвард
    try:
        return float(_BS.implied_vol(fwd, Strike(float(strike)), Premium(float(premium))).sigma)  # инверсия в IV
    except Exception:                                    # инверсия не сошлась…
        return float("nan")                              #   → NaN


def calibrate_iv_surface(
    spot: float,                                         # спот
    strikes: Sequence[float],                            # страйки
    ttm: Sequence[float],                                # сроки
    is_call: Sequence[bool],                             # флаги call/put
    premiums: Sequence[float],                           # рыночные премии
    flat_yield: float = 0.0,                             # плоская доходность
    init_params: Optional[Sequence[float]] = None,       # стартовые параметры (тёплый старт)
    num_iter: int = 50,                                  # число итераций LM
    tol: float = 1e-8,                                   # допуск сходимости
) -> HestonParameters:
    """Calibrate Heston to an option chain by least squares on premiums.

    This is the implied-volatility-surface calibration route (the LM optimiser
    fits model premiums, equivalently the IV surface, to the market quotes).
    """
    Ks = np.asarray(strikes, dtype=np.float64)           # страйки → массив
    Ts = np.asarray(ttm, dtype=np.float64)               # сроки → массив
    calls = np.asarray(is_call, dtype=np.bool_)          # флаги → массив
    pvs = np.asarray(premiums, dtype=np.float64)         # премии → массив

    order = np.argsort(Ts, kind="stable")                # сортируем по сроку (устойчиво)
    Ks, Ts, calls, pvs = Ks[order], Ts[order], calls[order], pvs[order]  # применяем порядок

    valid = (pvs > 0) & np.isfinite(pvs) & (Ts > 0)      # маска валидных котировок
    Ks, Ts, calls, pvs = Ks[valid], Ts[valid], calls[valid], pvs[valid]  # фильтруем
    if len(pvs) < 6:                                      # котировок мало для калибровки…
        raise ValueError("Need at least 6 valid option quotes to calibrate Heston")

    unique_T = np.unique(Ts)                              # уникальные сроки
    fwd_rates = float(spot) * np.exp(flat_yield * unique_T)  # форвардные цены по срокам
    fwd_curve = forward_curve_from_forward_rates(        # форвардная кривая
        Spot(float(spot)), ForwardRates(fwd_rates), TimesToMaturity(unique_T)
    )
    chain = VolSurfaceChainSpace(                         # опционная цепочка для калибровки
        fwd_curve, TimesToMaturity(Ts), Strikes(Ks), OptionTypes(calls), Premiums(pvs)
    )
    if len(chain.pvs) < 6:                               # после OTM-фильтрации осталось мало…
        raise ValueError("Too few out-of-the-money quotes survived filtering")

    hc = HestonCalc()                                    # локальный калькулятор (со своими настройками)
    hc.num_iter = int(num_iter)                          # число итераций
    hc.tol = float(tol)                                  # допуск
    if init_params is not None:                          # если задан тёплый старт…
        ip = np.asarray(init_params, dtype=np.float64)   #   стартовый вектор
        hc.update_cached_params(                         #   кэшируем стартовые параметры
            HestonParams(
                Variance(ip[0]), VarReversion(ip[1]), AverageVar(ip[2]),
                VolOfVar(ip[3]), Correlation(ip[4]), FlatForwardYield(flat_yield),
            )
        )
    weights = CalibrationWeights(np.ones_like(chain.pvs))  # равные веса котировок
    params, err = hc.calibrate(chain, FlatForwardYield(float(flat_yield)), weights)  # запуск калибровки

    return HestonParameters(                             # результат → domain-объект:
        v0=float(params.v0),                             #   начальная дисперсия
        kappa=float(params.kappa),                       #   скорость возврата
        theta=float(params.theta),                       #   долгосрочная дисперсия
        eps=float(params.eps),                           #   вол-оф-вол
        rho=float(params.rho),                           #   корреляция
        flat_yield=float(flat_yield),                    #   доходность
        calibration_error=float(err.v),                  #   ошибка калибровки
        feller_satisfied=bool(2.0 * params.kappa * params.theta - params.eps**2 >= 0.0),  # условие Феллера
    )


def heston_atm_iv(spot: float, ttm: float, params: HestonParameters) -> float:
    """ATM implied vol implied by Heston params (used for benchmarks / smiles)."""
    k = float(spot)                                      # страйк ATM = спот
    prem = heston_premiums(spot, [k], [ttm], [True], params)[0]  # цена ATM-call по Хестону
    return bs_implied_vol(spot, k, ttm, params.flat_yield, float(prem), True)  # → implied vol
