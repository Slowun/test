"""Dependence analysis and multi-criteria hedge-instrument ranking.

Implements linear (Pearson), rank (Spearman, Kendall) correlations, dynamic
DCC-GARCH correlation and cointegration tests (Engle-Granger & Johansen), then
ranks candidate instruments for hedging the primary asset by a weighted blend of
correlation strength, relationship stability, liquidity, hedging cost and risk
reduction.

================================ КАРТА МОДУЛЯ ================================
СЛОЙ:        cryptohedge.services (анализ зависимостей и ранжирование).
НАЗНАЧЕНИЕ:  считает линейные (Pearson) и ранговые (Spearman, Kendall) корреляции,
             динамическую DCC-GARCH корреляцию и тесты коинтеграции (Engle-Granger,
             Johansen); затем ранжирует кандидаты на хедж по взвешенной комбинации
             силы связи, стабильности, ликвидности, стоимости хеджа и снижения риска.
ИМПОРТИРУЕТ: warnings, numpy, pandas, scipy.stats; (опц.) arch, statsmodels,
             scipy.optimize; core.config.{CorrelationConfig,RankingWeights};
             domain.market.InstrumentRanking.
ЭКСПОРТИРУЕТ: static_correlations, rolling_stability, dcc_garch_correlations,
             cointegration, classify_relationship, rank_instruments.
КЕМ ИСПОЛЬЗУЕТСЯ: агент market_analysis (выбор инструментов хеджирования).
=============================================================================
"""

from __future__ import annotations                       # отложенные аннотации типов

import warnings                                          # подавление шумных предупреждений
from typing import Dict, List, Optional, Sequence        # аннотации

import numpy as np                                       # вычисления
import pandas as pd                                       # таблицы
from scipy import stats                                  # spearman/kendall

from cryptohedge.core.config import CorrelationConfig, RankingWeights  # конфиг корреляций и веса
from cryptohedge.domain.market import InstrumentRanking  # domain-объект ранжирования

warnings.filterwarnings("ignore")                        # глушим предупреждения статпакетов


# ----------------------------------------------------------------- static corr
def static_correlations(returns: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Pearson, Spearman and Kendall of every column against ``primary``."""
    base = returns[primary].to_numpy()                   # доходности первичного актива
    rows = []                                            # строки результата
    for col in returns.columns:                          # по каждому активу…
        if col == primary:                               #   сам первичный пропускаем
            continue
        other = returns[col].to_numpy()                  #   доходности кандидата
        mask = np.isfinite(base) & np.isfinite(other)    #   общие валидные точки
        if mask.sum() < 5:                               #   слишком мало точек…
            rows.append({"symbol": col, "pearson": np.nan, "spearman": np.nan, "kendall": np.nan})
            continue
        x, y = base[mask], other[mask]                   #   очищенные ряды
        rows.append(                                     #   три коэффициента корреляции:
            {
                "symbol": col,
                "pearson": float(np.corrcoef(x, y)[0, 1]),
                "spearman": float(stats.spearmanr(x, y).correlation),
                "kendall": float(stats.kendalltau(x, y).correlation),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")        # таблица корреляций (индекс — символ)


def rolling_stability(returns: pd.DataFrame, primary: str, window: int = 30) -> pd.Series:
    """Stability of the relationship = 1 / (1 + std of rolling Pearson corr)."""
    base = returns[primary]                              # ряд первичного актива
    out = {}                                             # результат по символам
    for col in returns.columns:                          # по каждому активу…
        if col == primary:                               #   сам первичный пропускаем
            continue
        roll = base.rolling(window).corr(returns[col]).dropna()  # скользящая корреляция
        out[col] = float(1.0 / (1.0 + np.std(roll))) if len(roll) > 1 else 0.0  # стабильность
    return pd.Series(out, name="stability")              # ряд стабильности


# ------------------------------------------------------------------ DCC-GARCH
def _garch_standardized(series: np.ndarray) -> Optional[np.ndarray]:
    """Standardised residuals from a GARCH(1,1) fit (returns in % for stability)."""
    try:
        from arch.univariate import arch_model           # импорт arch (может отсутствовать)

        r = series[np.isfinite(series)] * 100.0          # доходности в % (стабильность оценки)
        if len(r) < 20:                                  # данных мало…
            return None
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)  # модель GARCH(1,1)
        res = am.fit(disp="off", show_warning=False)     # обучение
        z = res.resid / res.conditional_volatility       # стандартизованные остатки
        return np.asarray(z, dtype=float)                # → массив
    except Exception:                                    # arch недоступен / не сошлось…
        return None


def _ewma_standardized(series: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA standardisation fallback when GARCH is unavailable."""
    r = np.nan_to_num(series)                            # заполняем NaN нулями
    var = np.empty_like(r)                               # массив дисперсий
    var[0] = np.var(r) if np.var(r) > 0 else 1e-8        # стартовая дисперсия
    for t in range(1, len(r)):                           # EWMA-рекурсия (RiskMetrics):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return r / np.sqrt(np.maximum(var, 1e-12))           # стандартизованные доходности


def dcc_garch_correlations(
    returns: pd.DataFrame,                               # доходности
    primary: str,                                        # первичный актив
    candidates: Sequence[str],                           # кандидаты
    a: float = 0.02,                                     # параметр DCC a
    b: float = 0.95,                                     # параметр DCC b
    estimate: bool = True,                               # оценивать (a,b) по QML
    max_iter: int = 50,                                  # лимит итераций оценки
) -> Dict[str, float]:
    """Mean dynamic conditional correlation between ``primary`` and each candidate.

    Two-step DCC: (1) GARCH(1,1) standardisation per series; (2) DCC(1,1)
    recursion. ``(a, b)`` are pooled-QML estimated across all candidate pairs when
    ``estimate`` is True, otherwise the configured constants are used.
    """
    series = {primary: returns[primary].to_numpy()}      # словарь рядов: первичный
    for c in candidates:                                 # + кандидаты
        series[c] = returns[c].to_numpy()

    std: Dict[str, np.ndarray] = {}                      # стандартизованные ряды
    for sym, arr in series.items():                      # по каждому ряду…
        z = _garch_standardized(arr)                     #   стандартизация через GARCH
        std[sym] = z if z is not None else _ewma_standardized(arr)  # фолбэк на EWMA

    zp = std[primary]                                    # стандартизованный первичный
    pairs = [(zp, std[c]) for c in candidates if len(std[c]) == len(zp)]  # валидные пары

    if estimate and pairs:                               # если нужно оценить (a,b)…
        a, b = _estimate_dcc_ab(pairs, a, b, max_iter)   #   QML-оценка по всем парам

    result: Dict[str, float] = {}                        # средние DCC-корреляции
    for c in candidates:                                 # по каждому кандидату…
        zc = std[c]
        if len(zc) != len(zp):                           #   длины не совпадают…
            result[c] = float("nan")
            continue
        corr_path = _dcc_pair_corr(zp, zc, a, b)         #   путь динамической корреляции
        result[c] = float(np.nanmean(corr_path))         #   среднее по пути
    return result


def _dcc_pair_corr(z1: np.ndarray, z2: np.ndarray, a: float, b: float) -> np.ndarray:
    Z = np.column_stack([z1, z2])                        # матрица стандартизованных рядов
    Qbar = np.cov(Z, rowvar=False)                       # безусловная ковариация
    T = len(z1)                                          # длина ряда
    Q = Qbar.copy()                                      # стартовая Q
    out = np.empty(T)                                    # путь корреляции
    for t in range(T):                                   # DCC(1,1)-рекурсия:
        if t > 0:
            zz = np.outer(Z[t - 1], Z[t - 1])            #   внешнее произведение прошлых остатков
            Q = (1 - a - b) * Qbar + a * zz + b * Q      #   обновление Q
        d = np.sqrt(np.diag(Q))                          #   стандартные отклонения
        out[t] = Q[0, 1] / (d[0] * d[1]) if d[0] > 0 and d[1] > 0 else np.nan  # корреляция
    return out


def _estimate_dcc_ab(pairs, a0: float, b0: float, max_iter: int):
    from scipy.optimize import minimize                  # импорт оптимизатора

    def neg_ll(params):                                  # отрицательная лог-правдоподобность DCC:
        a, b = params
        if a < 0 or b < 0 or a + b >= 0.999:             #   нарушены ограничения…
            return 1e6                                   #     штраф
        total = 0.0
        for z1, z2 in pairs:                             #   по всем парам…
            Z = np.column_stack([z1, z2])
            Qbar = np.cov(Z, rowvar=False)
            Q = Qbar.copy()
            for t in range(len(z1)):                     #     DCC-рекурсия + накопление LL:
                if t > 0:
                    zz = np.outer(Z[t - 1], Z[t - 1])
                    Q = (1 - a - b) * Qbar + a * zz + b * Q
                d = np.sqrt(np.diag(Q))
                R = Q / np.outer(d, d)                    #       корреляционная матрица
                detR = R[0, 0] * R[1, 1] - R[0, 1] ** 2  #       детерминант 2×2
                if detR <= 0:                            #       вырождена…
                    return 1e6
                zt = Z[t]
                quad = (zt @ np.linalg.inv(R) @ zt)      #       квадратичная форма
                total += 0.5 * (np.log(detR) + quad)     #       вклад в -LL
        return total

    try:
        res = minimize(neg_ll, x0=[a0, b0], method="Nelder-Mead",  # минимизация -LL
                       options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-2})
        a, b = float(res.x[0]), float(res.x[1])          # оценённые (a,b)
        if a < 0 or b < 0 or a + b >= 0.999:             # вне допустимой области…
            return a0, b0                                #   → дефолт
        return a, b
    except Exception:                                    # оптимизация упала…
        return a0, b0                                    #   → дефолт


# ---------------------------------------------------------------- cointegration
def cointegration(
    prices: pd.DataFrame,                                # уровни цен
    primary: str,                                        # первичный актив
    candidates: Sequence[str],                           # кандидаты
    method: str = "both",                                # метод: engle_granger / johansen / both
    pvalue: float = 0.05,                                # порог значимости
    det_order: int = 0,                                  # детерминированный тренд (Johansen)
    k_ar_diff: int = 1,                                  # число лагов (Johansen)
) -> Dict[str, bool]:
    """Test each candidate for cointegration with the primary price level."""
    from statsmodels.tsa.stattools import coint          # тест Энгла-Грейнджера
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # тест Йохансена

    base = prices[primary].to_numpy()                    # цена первичного актива
    out: Dict[str, bool] = {}                            # результат по кандидатам
    for c in candidates:                                 # по каждому кандидату…
        other = prices[c].to_numpy()
        eg = jo = False                                  # флаги двух тестов
        if method in ("engle_granger", "both"):          #   тест Энгла-Грейнджера:
            try:
                _, pval, _ = coint(base, other)
                eg = bool(pval < pvalue)                 #     p < порог → коинтеграция
            except Exception:
                eg = False
        if method in ("johansen", "both"):               #   тест Йохансена:
            try:
                jres = coint_johansen(np.column_stack([base, other]), det_order, k_ar_diff)
                jo = bool(jres.lr1[0] > jres.cvt[0, 1])  # trace stat vs 95% crit, r=0
            except Exception:
                jo = False
        out[c] = (eg or jo) if method == "both" else (eg if method == "engle_granger" else jo)  # комбинируем
    return out


# ---------------------------------------------------------------------- ranking
def classify_relationship(pearson: float, pos: float, neg: float, zero_band: float) -> str:
    if np.isnan(pearson):                                # нет корреляции…
        return "neutral"
    if pearson >= pos:                                   # сильная положительная
        return "positive"
    if pearson <= neg:                                   # сильная обратная
        return "inverse"
    if abs(pearson) <= zero_band:                        # около нуля
        return "neutral"
    return "weak"                                        # слабая связь


def _minmax(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)                  # вход → массив
    finite = v[np.isfinite(v)]                           # только конечные значения
    if len(finite) == 0:                                 # все NaN…
        return np.zeros_like(v)                          #   → нули
    lo, hi = np.nanmin(v), np.nanmax(v)                  # границы
    if hi - lo < 1e-12:                                  # все значения равны…
        return np.nan_to_num(np.ones_like(v) * 0.5)      #   → 0.5
    return np.nan_to_num((v - lo) / (hi - lo))           # нормировка в [0,1]


def rank_instruments(
    static: pd.DataFrame,                                # статические корреляции
    stability: pd.Series,                                # стабильность связи
    dcc: Dict[str, float],                               # средние DCC-корреляции
    cointegrated: Dict[str, bool],                       # флаги коинтеграции
    liquidity: pd.Series,                                # ликвидность
    hedge_cost: pd.Series,                               # стоимость хеджа
    config: CorrelationConfig,                           # конфиг порогов
    weights: RankingWeights,                             # веса критериев
) -> List[InstrumentRanking]:
    """Combine all criteria into a weighted score and rank candidates.

    A good hedge has strong (positive or inverse) and *stable* dependence, deep
    liquidity, low hedging cost and high risk-reduction potential
    (``|corr|`` is a proxy for variance reduction of the hedged book).
    """
    syms = list(static.index)                            # список символов
    abs_corr = static["pearson"].abs().to_numpy()        # |Pearson|
    dcc_arr = np.array([abs(dcc.get(s, np.nan)) for s in syms])  # |DCC|
    corr_strength = np.nanmax(np.column_stack([abs_corr, dcc_arr]), axis=1)  # макс. сила связи
    risk_reduction = 1.0 - np.sqrt(np.clip(1.0 - corr_strength**2, 0.0, 1.0))  # потенциал снижения риска

    n_corr = _minmax(corr_strength)                      # нормированная сила связи
    n_stab = _minmax(stability.reindex(syms).to_numpy())  # нормированная стабильность
    n_liq = _minmax(liquidity.reindex(syms).to_numpy())  # нормированная ликвидность
    n_cost = 1.0 - _minmax(hedge_cost.reindex(syms).to_numpy())  # lower cost is better  # дешевле = лучше
    n_rr = _minmax(risk_reduction)                       # нормированное снижение риска
    coint_bonus = np.array([0.1 if cointegrated.get(s, False) else 0.0 for s in syms])  # бонус за коинтеграцию

    score = (                                            # итоговый взвешенный балл:
        weights.correlation * n_corr
        + weights.stability * n_stab
        + weights.liquidity * n_liq
        + weights.hedge_cost * n_cost
        + weights.risk_reduction * n_rr
        + coint_bonus
    )

    rankings: List[InstrumentRanking] = []               # список результатов
    for i, s in enumerate(syms):                         # по каждому символу…
        rankings.append(                                 #   собираем объект ранжирования:
            InstrumentRanking(
                symbol=s,
                pearson=float(static.loc[s, "pearson"]),
                spearman=float(static.loc[s, "spearman"]),
                kendall=float(static.loc[s, "kendall"]),
                dcc_mean=float(dcc.get(s, np.nan)),
                cointegrated=bool(cointegrated.get(s, False)),
                stability=float(stability.get(s, 0.0)),
                liquidity=float(liquidity.get(s, 0.0)),
                hedge_cost=float(hedge_cost.get(s, 0.0)),
                risk_reduction=float(risk_reduction[i]),
                score=float(score[i]),
                relationship=classify_relationship(
                    float(static.loc[s, "pearson"]),
                    config.positive_threshold,
                    config.negative_threshold,
                    config.zero_band,
                ),
            )
        )
    rankings.sort(key=lambda r: r.score, reverse=True)   # сортируем по убыванию балла
    return rankings
