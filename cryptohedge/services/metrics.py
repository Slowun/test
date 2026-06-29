"""Portfolio performance and risk metrics.

A single :func:`compute_metrics` returns ROI, Sharpe, Sortino, Calmar, Maximum
Drawdown, Profit Factor, Win Rate, VaR, CVaR, Expected Shortfall, Beta, Alpha and
Information Ratio. All ratios are annualised with the configured period count.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (метрики доходности и риска).
НАЗНАЧЕНИЕ:  единая функция compute_metrics возвращает ROI, CAGR, Sharpe, Sortino,
             Calmar, Max Drawdown, Profit Factor, Win Rate, VaR, CVaR, Expected
             Shortfall, Beta, Alpha, Information Ratio, волатильность. Все
             коэффициенты годовые (через periods_per_year). VaR — три метода:
             historical / gaussian / cornish_fisher.
ИМПОРТИРУЕТ: dataclass; numpy; (опц.) scipy.stats.
ЭКСПОРТИРУЕТ: PerformanceMetrics, equity_to_returns, max_drawdown, value_at_risk,
             conditional_var, compute_metrics.
КЕМ ИСПОЛЬЗУЕТСЯ: агенты risk_management, backtesting, portfolio_optimization.
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

from dataclasses import dataclass, asdict                # для PerformanceMetrics
from typing import Dict, Optional                        # аннотации

import numpy as np                                       # вычисления


@dataclass(frozen=True)
class PerformanceMetrics:                                # сводка метрик (неизменяемая)
    roi: float                                          # совокупная доходность
    cagr: float                                         # среднегодовая доходность
    sharpe: float                                       # коэффициент Шарпа
    sortino: float                                      # коэффициент Сортино
    calmar: float                                       # коэффициент Кальмара
    max_drawdown: float                                 # максимальная просадка
    profit_factor: float                                # профит-фактор
    win_rate: float                                     # доля прибыльных периодов
    var: float                                          # VaR
    cvar: float                                         # CVaR
    expected_shortfall: float                           # ожидаемые потери (= CVaR)
    beta: float                                         # бета к бенчмарку
    alpha: float                                        # альфа (годовая)
    information_ratio: float                            # информационный коэффициент
    volatility: float                                   # годовая волатильность

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)                             # сериализация в словарь


def equity_to_returns(equity: np.ndarray) -> np.ndarray:
    equity = np.asarray(equity, dtype=float)            # кривая капитала → массив
    if len(equity) < 2:                                # данных мало…
        return np.array([])
    base = np.where(np.abs(equity[:-1]) < 1e-12, np.nan, equity[:-1])  # защита от деления на 0
    return np.nan_to_num((equity[1:] - equity[:-1]) / base)  # простые доходности


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)            # кривая капитала → массив
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)         # текущий исторический максимум
    drawdown = (equity - running_max) / np.where(np.abs(running_max) < 1e-12, np.nan, running_max)  # просадки
    return float(np.nanmin(drawdown)) if len(drawdown) else 0.0  # минимальная (самая глубокая)


def value_at_risk(returns: np.ndarray, confidence: float = 0.95, method: str = "historical") -> float:
    """One-period VaR as a positive loss fraction."""
    r = np.asarray(returns, dtype=float)                # доходности → массив
    r = r[np.isfinite(r)]                               # только конечные
    if len(r) == 0:
        return 0.0
    alpha = 1.0 - confidence                            # уровень хвоста
    if method == "gaussian":                            # параметрический (нормальный) VaR:
        from scipy import stats

        return float(-(np.mean(r) + stats.norm.ppf(alpha) * np.std(r, ddof=1)))
    if method == "cornish_fisher":                      # Корниш-Фишер (учёт асимметрии/эксцесса):
        from scipy import stats

        mu, sigma = np.mean(r), np.std(r, ddof=1)
        s = stats.skew(r)                               #   асимметрия
        k = stats.kurtosis(r)                           #   эксцесс
        z = stats.norm.ppf(alpha)                       #   нормальный квантиль
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36  # поправленный квантиль
        return float(-(mu + z_cf * sigma))
    return float(-np.quantile(r, alpha))               # исторический VaR (квантиль)


def conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall (historical)."""
    r = np.asarray(returns, dtype=float)                # доходности → массив
    r = r[np.isfinite(r)]                               # только конечные
    if len(r) == 0:
        return 0.0
    alpha = 1.0 - confidence                            # уровень хвоста
    threshold = np.quantile(r, alpha)                   # порог VaR
    tail = r[r <= threshold]                            # хвост потерь
    return float(-np.mean(tail)) if len(tail) else float(-threshold)  # средние потери в хвосте


def compute_metrics(
    returns: np.ndarray,                               # доходности стратегии
    benchmark: Optional[np.ndarray] = None,            # доходности бенчмарка (для beta/alpha)
    risk_free: float = 0.0,                            # безрисковая ставка (годовая)
    periods_per_year: int = 365,                       # периодов в году
    var_confidence: float = 0.95,                      # уровень доверия для VaR
    var_method: str = "historical",                    # метод VaR
    equity: Optional[np.ndarray] = None,               # кривая капитала (для MDD)
) -> PerformanceMetrics:
    r = np.asarray(returns, dtype=float)               # доходности → массив
    r = r[np.isfinite(r)]                              # только конечные
    if len(r) == 0:                                    # данных нет…
        return PerformanceMetrics(*([0.0] * 15))       #   все метрики = 0

    rf_per = risk_free / periods_per_year              # безрисковая на период
    excess = r - rf_per                                # избыточная доходность
    mean, std = float(np.mean(r)), float(np.std(r, ddof=1)) if len(r) > 1 else 0.0  # среднее и std
    ann_factor = np.sqrt(periods_per_year)             # годовой множитель (√N)

    total_return = float(np.prod(1.0 + r) - 1.0)       # совокупная доходность
    n_years = len(r) / periods_per_year                # длительность в годах
    cagr = float((1.0 + total_return) ** (1.0 / n_years) - 1.0) if n_years > 0 and total_return > -1 else 0.0  # CAGR
    volatility = std * ann_factor                      # годовая волатильность

    sharpe = float(np.mean(excess) / std * ann_factor) if std > 0 else 0.0  # коэффициент Шарпа
    downside = r[r < 0]                                # отрицательные доходности
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0  # downside-волатильность
    sortino = float(np.mean(excess) / dstd * ann_factor) if dstd > 0 else 0.0  # коэффициент Сортино

    eq = np.asarray(equity, dtype=float) if equity is not None else np.cumprod(1.0 + r)  # кривая капитала
    mdd = max_drawdown(eq)                             # максимальная просадка
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0  # коэффициент Кальмара

    gains = r[r > 0].sum()                             # суммарные прибыли
    losses = -r[r < 0].sum()                           # суммарные убытки
    profit_factor = float(gains / losses) if losses > 1e-12 else float("inf") if gains > 0 else 0.0  # профит-фактор
    win_rate = float(np.mean(r > 0))                   # доля прибыльных периодов

    var = value_at_risk(r, var_confidence, var_method)  # VaR
    cvar = conditional_var(r, var_confidence)          # CVaR

    beta = alpha = information_ratio = 0.0             # метрики относительно бенчмарка
    if benchmark is not None:                          # если бенчмарк задан…
        b = np.asarray(benchmark, dtype=float)
        m = min(len(r), len(b))                        #   общая длина
        rr, bb = r[-m:], b[-m:]                        #   выравниваем по хвосту
        var_b = float(np.var(bb, ddof=1)) if m > 1 else 0.0  # дисперсия бенчмарка
        if var_b > 0:
            beta = float(np.cov(rr, bb, ddof=1)[0, 1] / var_b)  # бета
            alpha = float((np.mean(rr) - beta * np.mean(bb)) * periods_per_year)  # альфа (годовая)
        active = rr - bb                               #   активная доходность
        astd = float(np.std(active, ddof=1)) if m > 1 else 0.0  # tracking error
        information_ratio = float(np.mean(active) / astd * ann_factor) if astd > 0 else 0.0  # IR

    return PerformanceMetrics(                         # собираем сводку метрик:
        roi=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        profit_factor=profit_factor,
        win_rate=win_rate,
        var=var,
        cvar=cvar,
        expected_shortfall=cvar,
        beta=beta,
        alpha=alpha,
        information_ratio=information_ratio,
        volatility=volatility,
    )
