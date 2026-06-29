"""Portfolio construction / rebalancing.

Implements five optimisers - Mean-Variance, Risk Parity, Minimum Variance,
Maximum Diversification and CVaR (Rockafellar-Uryasev LP) - all honouring
long-only / max-weight bounds and an optional turnover budget that captures the
cost and frequency of rebalancing.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (оптимизация портфеля).
НАЗНАЧЕНИЕ:  реализует пять оптимизаторов весов — Mean-Variance, Risk Parity,
             Minimum Variance, Maximum Diversification и CVaR (ЛП Рокафеллара-
             Урясева). Все учитывают long-only/максимальный вес и опциональный
             бюджет оборота (turnover) — стоимость и частоту ребалансировки.
ИМПОРТИРУЕТ: numpy; scipy.optimize.{linprog, minimize}.
ЭКСПОРТИРУЕТ: mean_variance, min_variance, risk_parity, max_diversification,
             cvar_optimization, optimize (диспетчер «Стратегия»), turnover,
             transaction_cost.
КЕМ ИСПОЛЬЗУЕТСЯ: агент portfolio_optimization; сервис portfolio_backtest.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from typing import Dict, List, Optional                  # аннотации

import numpy as np                                       # вычисления
from scipy.optimize import linprog, minimize             # ЛП (CVaR) и НЛП (остальные)


def _bounds(n: int, long_only: bool, max_weight: float):
    lo = 0.0 if long_only else -max_weight               # нижняя граница веса (0 или -max)
    return [(lo, max_weight)] * n                        # границы для всех активов


def _normalize(w: np.ndarray) -> np.ndarray:
    s = np.sum(w)                                        # сумма весов
    return w / s if abs(s) > 1e-12 else np.ones_like(w) / len(w)  # нормировка к 1 (иначе равные веса)


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # ограничение: сумма весов = 1


def _turnover_constraint(w_prev: Optional[np.ndarray], max_turnover: float):
    if w_prev is None:                                  # нет предыдущих весов…
        return []                                        #   нет ограничения
    return [{"type": "ineq", "fun": lambda w: max_turnover - np.sum(np.abs(w - w_prev))}]  # оборот ≤ лимита


def mean_variance(mu, Sigma, risk_aversion, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    n = len(mu)                                          # число активов
    mu = np.asarray(mu, float)                           # ожидаемые доходности
    Sigma = np.asarray(Sigma, float)                    # ковариационная матрица

    def neg_util(w):                                    # -полезность (доходность - риск):
        return -(w @ mu - 0.5 * risk_aversion * w @ Sigma @ w)

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])  # ограничения
    res = minimize(neg_util, np.ones(n) / n, method="SLSQP",  # SLSQP-оптимизация
                   bounds=_bounds(n, long_only, max_weight), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return _normalize(res.x) if res.success else np.ones(n) / n  # успех → веса, иначе равные


def min_variance(Sigma, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    n = Sigma.shape[0]                                  # число активов
    Sigma = np.asarray(Sigma, float)                   # ковариационная матрица

    def variance(w):                                    # дисперсия портфеля:
        return w @ Sigma @ w

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])  # ограничения
    res = minimize(variance, np.ones(n) / n, method="SLSQP",  # минимизация дисперсии
                   bounds=_bounds(n, long_only, max_weight), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return _normalize(res.x) if res.success else np.ones(n) / n  # успех → веса, иначе равные


def risk_parity(Sigma, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    """Equalise marginal risk contributions across assets."""
    n = Sigma.shape[0]                                  # число активов
    Sigma = np.asarray(Sigma, float)                   # ковариационная матрица
    target = np.ones(n) / n                             # цель: равные вклады в риск

    def objective(w):                                   # отклонение вкладов риска от равных:
        port_var = w @ Sigma @ w                        #   дисперсия портфеля
        if port_var <= 0:
            return 1e6
        mrc = Sigma @ w                                 #   маржинальные вклады
        rc = w * mrc / np.sqrt(port_var)                #   вклады в риск
        rc = rc / np.sum(rc)                            #   нормировка
        return np.sum((rc - target) ** 2)              #   квадрат отклонения от равных

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])  # ограничения
    res = minimize(objective, target, method="SLSQP",   # минимизация отклонения
                   bounds=_bounds(n, max(long_only, True), max_weight), constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return _normalize(res.x) if res.success else target  # успех → веса, иначе равные


def max_diversification(Sigma, long_only=True, max_weight=1.0, w_prev=None, max_turnover=None):
    """Maximise the diversification ratio (weighted avg vol / portfolio vol)."""
    n = Sigma.shape[0]                                  # число активов
    Sigma = np.asarray(Sigma, float)                   # ковариационная матрица
    sigma = np.sqrt(np.diag(Sigma))                    # волатильности активов

    def neg_dr(w):                                      # -коэффициент диверсификации:
        pv = np.sqrt(w @ Sigma @ w)                     #   волатильность портфеля
        return -(w @ sigma) / pv if pv > 0 else 1e6    #   взвеш. средняя вол / вол портфеля

    cons = [_sum_to_one()] + (_turnover_constraint(w_prev, max_turnover) if max_turnover else [])  # ограничения
    res = minimize(neg_dr, np.ones(n) / n, method="SLSQP",  # максимизация диверсификации
                   bounds=_bounds(n, long_only, max_weight), constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return _normalize(res.x) if res.success else np.ones(n) / n  # успех → веса, иначе равные


def cvar_optimization(scenarios, alpha=0.95, long_only=True, max_weight=1.0, target_return=None, mu=None):
    """Minimise portfolio CVaR via the Rockafellar-Uryasev linear program.

    Variables: ``[w (n), var (1), u (S)]`` where ``u_s >= -scenario_s.w - var``.
    Objective: ``var + 1/((1-alpha) S) * sum(u)``.
    """
    R = np.asarray(scenarios, float)  # (S, n) scenario returns  # сценарии доходностей
    S, n = R.shape                                      # число сценариев и активов
    nvars = n + 1 + S                                   # всего переменных ЛП

    c = np.zeros(nvars)                                 # вектор целевой функции
    c[n] = 1.0                                          # коэффициент при VaR
    c[n + 1:] = 1.0 / ((1.0 - alpha) * S)              # коэффициенты при «хвостовых» u

    # u_s >= -R_s . w - var  ->  -R_s.w - var - u_s <= 0  # ограничения неравенств:
    A_ub = np.zeros((S, nvars))
    A_ub[:, :n] = -R                                    #   -R·w
    A_ub[:, n] = -1.0                                   #   -var
    A_ub[np.arange(S), n + 1 + np.arange(S)] = -1.0    #   -u_s
    b_ub = np.zeros(S)

    A_eq = np.zeros((1, nvars))                         # равенство:
    A_eq[0, :n] = 1.0                                  #   сумма весов
    b_eq = [1.0]                                        #   = 1

    if target_return is not None and mu is not None:    # опциональное ограничение доходности:
        add = np.zeros((1, nvars))
        add[0, :n] = -np.asarray(mu, float)            #   -mu·w ≤ -target → mu·w ≥ target
        A_ub = np.vstack([A_ub, add])
        b_ub = np.concatenate([b_ub, [-target_return]])

    lo = 0.0 if long_only else -max_weight              # нижняя граница весов
    bounds = [(lo, max_weight)] * n + [(None, None)] + [(0, None)] * S  # границы всех переменных
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")  # решаем ЛП
    if res.success:                                    # успех…
        return _normalize(res.x[:n])                    #   → нормированные веса
    return np.ones(n) / n                              # иначе равные веса


def optimize(
    method: str,                                        # метод оптимизации
    mu: Optional[np.ndarray],                           # ожидаемые доходности
    Sigma: np.ndarray,                                  # ковариационная матрица
    scenarios: Optional[np.ndarray] = None,             # сценарии (для CVaR)
    risk_aversion: float = 5.0,                         # неприятие риска (MV)
    cvar_alpha: float = 0.95,                           # уровень CVaR
    long_only: bool = True,                             # только длинные позиции
    max_weight: float = 1.0,                            # максимальный вес актива
    w_prev: Optional[np.ndarray] = None,                # предыдущие веса (turnover)
    max_turnover: Optional[float] = None,               # лимит оборота
) -> np.ndarray:
    """Dispatch to the requested optimiser (Strategy pattern)."""
    method = method.lower()                             # имя метода в нижнем регистре
    if method == "mean_variance":                       # → Mean-Variance
        return mean_variance(mu, Sigma, risk_aversion, long_only, max_weight, w_prev, max_turnover)
    if method == "min_variance":                        # → Minimum Variance
        return min_variance(Sigma, long_only, max_weight, w_prev, max_turnover)
    if method == "risk_parity":                         # → Risk Parity
        return risk_parity(Sigma, long_only, max_weight, w_prev, max_turnover)
    if method == "max_diversification":                 # → Maximum Diversification
        return max_diversification(Sigma, long_only, max_weight, w_prev, max_turnover)
    if method == "cvar":                                # → CVaR
        if scenarios is None:
            raise ValueError("CVaR optimisation requires return scenarios")
        return cvar_optimization(scenarios, cvar_alpha, long_only, max_weight, mu=mu)
    raise ValueError(f"Unknown optimisation method: {method}")  # неизвестный метод


def turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(w_new) - np.asarray(w_old))))  # суммарный оборот весов


def transaction_cost(w_new: np.ndarray, w_old: np.ndarray, fee_pct: float, capital: float) -> float:
    return float(turnover(w_new, w_old) * fee_pct * capital)  # издержки ребалансировки
