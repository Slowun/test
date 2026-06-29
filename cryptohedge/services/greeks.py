"""Greeks computation and aggregation.

The default :class:`HestonGreeksEngine` computes first- and second-order
sensitivities by finite differences on the semi-analytical Heston price (fast,
deterministic, torch-free): delta, gamma, vega, theta, rho, vanna, volga and
charm. Greeks are consistent across instruments (vega in vol space), so the
delta/vega hedge ratios used by the hedging agent are well-defined. An optional
Monte-Carlo engine mirrors the reference autograd approach when ``engine='mc'``.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (вычислительный use-case греков).
НАЗНАЧЕНИЕ:  считает чувствительности 1-го и 2-го порядка методом конечных
             разностей по полу-аналитической цене Хестона: delta, gamma, vega,
             theta, rho, vanna, volga, charm. Греки согласованы между инструментами
             (vega в пространстве волатильности), поэтому коэффициенты Δ/ν-хеджа
             корректно определены.
ИМПОРТИРУЕТ: math, numpy; core.config.GreeksConfig; domain.greeks.{Greeks,
             PortfolioGreeks}; domain.market.{HestonParameters,OptionContract};
             services.heston_pricing.heston_premiums.
ЭКСПОРТИРУЕТ: HestonGreeksEngine, aggregate, greeks_to_frame.
КЕМ ИСПОЛЬЗУЕТСЯ: агент greeks_calculation; косвенно hedging_decision.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import math                                              # sqrt для перевода дисперсии в волатильность
from typing import Dict, Iterable, List                  # аннотации

import numpy as np                                       # массивы

from cryptohedge.core.config import GreeksConfig         # конфиг бампов/движка греков
from cryptohedge.domain.greeks import Greeks, PortfolioGreeks  # domain-объекты греков
from cryptohedge.domain.market import HestonParameters, OptionContract  # параметры и контракт
from cryptohedge.services.heston_pricing import heston_premiums  # цена опциона по Хестону


class HestonGreeksEngine:
    """Finite-difference greeks on the analytical Heston price."""

    def __init__(self, config: GreeksConfig) -> None:
        self.config = config                             # конфиг бампов (шаги конечных разностей)

    # ------------------------------------------------------------------ pricing
    def price(self, spot: float, params: HestonParameters, K: float, T: float, is_call: bool) -> float:
        return float(heston_premiums(spot, [K], [T], [is_call], params)[0])  # цена одного опциона

    def _with_v0(self, params: HestonParameters, v0: float) -> HestonParameters:
        return HestonParameters(v0=max(v0, 1e-6), kappa=params.kappa, theta=params.theta,  # копия с новым v0
                                eps=params.eps, rho=params.rho, flat_yield=params.flat_yield)

    def _with_rate(self, params: HestonParameters, r: float) -> HestonParameters:
        return HestonParameters(v0=params.v0, kappa=params.kappa, theta=params.theta,  # копия с новой ставкой
                                eps=params.eps, rho=params.rho, flat_yield=r)

    def _delta(self, spot, params, K, T, is_call, dS) -> float:
        up = self.price(spot + dS, params, K, T, is_call)  # цена при споте +dS
        dn = self.price(spot - dS, params, K, T, is_call)  # цена при споте -dS
        return (up - dn) / (2 * dS)                      # центральная разность → дельта

    # ------------------------------------------------------------------- greeks
    def compute(self, spot: float, params: HestonParameters, contract: OptionContract, ttm: float) -> Greeks:
        K, is_call = contract.strike, contract.is_call   # страйк и тип опциона
        cfg = self.config                                # конфиг бампов
        dS = max(spot * cfg.spot_bump_pct, 1e-8)         # шаг по споту
        dsig = cfg.vol_bump                              # шаг по волатильности
        dr = cfg.rate_bump                               # шаг по ставке
        dT = cfg.time_bump_days / 365.0                  # шаг по времени (в годах)

        sigma0 = math.sqrt(max(params.v0, 1e-8))         # текущая волатильность из v0
        base = self.price(spot, params, K, ttm, is_call)  # базовая цена

        # spot greeks                                     # греки по споту:
        p_su = self.price(spot + dS, params, K, ttm, is_call)  # цена при +dS
        p_sd = self.price(spot - dS, params, K, ttm, is_call)  # цена при -dS
        delta = (p_su - p_sd) / (2 * dS)                 # дельта
        gamma = (p_su - 2 * base + p_sd) / (dS**2)       # гамма

        # vol greeks (bump in volatility space; v0 = sigma^2)  # греки по волатильности:
        params_vu = self._with_v0(params, (sigma0 + dsig) ** 2)  # параметры при σ+
        params_vd = self._with_v0(params, (sigma0 - dsig) ** 2)  # параметры при σ-
        p_vu = self.price(spot, params_vu, K, ttm, is_call)  # цена при σ+
        p_vd = self.price(spot, params_vd, K, ttm, is_call)  # цена при σ-
        vega = (p_vu - p_vd) / (2 * dsig)                # вега
        volga = (p_vu - 2 * base + p_vd) / (dsig**2)     # волга (∂vega/∂σ)

        # rho                                             # ро (чувствительность к ставке):
        if dr > 0:                                        # если бамп ставки задан…
            p_ru = self.price(spot, self._with_rate(params, params.flat_yield + dr), K, ttm, is_call)  # ставка +dr
            p_rd = self.price(spot, self._with_rate(params, params.flat_yield - dr), K, ttm, is_call)  # ставка -dr
            rho = (p_ru - p_rd) / (2 * dr)               # ро
        else:
            rho = 0.0                                     # иначе ро = 0

        # theta (per day): value lost as maturity shortens by one day  # тета (за день):
        theta = 0.0
        if ttm - dT > 1e-6:                              # если можно сократить срок…
            p_tm = self.price(spot, params, K, ttm - dT, is_call)  # цена при сокращённом сроке
            theta = (p_tm - base) / cfg.time_bump_days   # тета (изменение за день)

        # vanna: d^2P / (dS dsigma)                       # ванна (∂²P/∂S∂σ):
        p_su_vu = self.price(spot + dS, params_vu, K, ttm, is_call)  # S+, σ+
        p_su_vd = self.price(spot + dS, params_vd, K, ttm, is_call)  # S+, σ-
        p_sd_vu = self.price(spot - dS, params_vu, K, ttm, is_call)  # S-, σ+
        p_sd_vd = self.price(spot - dS, params_vd, K, ttm, is_call)  # S-, σ-
        vanna = (p_su_vu - p_su_vd - p_sd_vu + p_sd_vd) / (4 * dS * dsig)  # смешанная производная

        # charm: d delta / d(time) per day                # чарм (∂delta/∂t за день):
        charm = 0.0
        if ttm - dT > 1e-6:                              # если можно сократить срок…
            delta_tm = self._delta(spot, params, K, ttm - dT, is_call, dS)  # дельта при сокращённом сроке
            charm = (delta_tm - delta) / cfg.time_bump_days  # чарм

        greeks = Greeks(                                 # собираем греки на единицу нотионала:
            premium=base, delta=delta, gamma=gamma, vega=vega, theta=theta,
            rho=rho, vanna=vanna, volga=volga, charm=charm,
        )
        return greeks.scaled(contract.notional)          # масштабируем на нотионал контракта

    # ---------------------------------------------------------------- chain grid
    def chain_greeks(self, spot, params, strikes, ttm, is_call=True) -> List[Greeks]:
        return [                                         # греки по сетке страйков:
            self.compute(spot, params, OptionContract("primary", float(k), 0, is_call), ttm)
            for k in strikes
        ]


def aggregate(per_instrument: Iterable[Greeks]) -> PortfolioGreeks:
    """Aggregate per-instrument greeks into a single portfolio-level object."""
    total = Greeks()                                     # нулевые греки
    for g in per_instrument:                             # по каждому инструменту…
        total = total + g                                #   суммируем греки
    return PortfolioGreeks.from_greeks(total)            # → агрегированные греки портфеля


def greeks_to_frame(per_instrument: Dict[str, Greeks]):
    import pandas as pd                                  # локальный импорт pandas

    return pd.DataFrame({name: g.to_dict() for name, g in per_instrument.items()}).T  # греки → DataFrame
